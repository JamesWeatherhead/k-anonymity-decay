"""
Orchestrates k-anonymity simulations across patient population and models.

Coordinates disclosure simulations using AnonymityEngine and DisclosureModels.
Outputs k values per turn, threshold crossing turns, and summary statistics.

Author: James Weatherhead, UTMB (jacweath@utmb.edu)
"""

import logging
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import time


def stable_hash(s: str) -> int:
    """
    Deterministic hash that's consistent across Python sessions.

    Python's built-in hash() uses randomized seeding (PYTHONHASHSEED) for
    security, making results non-reproducible across process invocations.
    This function uses SHA-256 to provide deterministic hashing.

    Args:
        s: String to hash

    Returns:
        Deterministic integer hash value
    """
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from .anonymity_engine import AnonymityEngine
from .disclosure_models import (
    DisclosureModel,
    ProgressiveRefinementModel,
    RandomOrderingModel,
    RarityOrderedModel,
    create_disclosure_model
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """
    Container for single-patient simulation results.

    Attributes:
        patient_id: Unique patient identifier
        model_name: Name of disclosure model used
        k_by_turn: List of k values after each turn
        turns_to_thresholds: Dict mapping threshold to turn number (or None)
        final_k: k value after all QIs disclosed
        n_turns: Total number of disclosure turns
        disclosure_sequence: List of (qi_name, qi_value) tuples
    """
    patient_id: str
    model_name: str
    k_by_turn: List[int]
    turns_to_thresholds: Dict[int, Optional[int]]
    final_k: int
    n_turns: int
    disclosure_sequence: List[Tuple[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        result = {
            'patient_id': self.patient_id,
            'model_name': self.model_name,
            'final_k': self.final_k,
            'n_turns': self.n_turns,
        }
        # Add k values by turn
        for i, k in enumerate(self.k_by_turn, 1):
            result[f'k_turn_{i}'] = k

        # Add threshold crossing turns
        for threshold, turn in self.turns_to_thresholds.items():
            result[f'turns_to_k{threshold}'] = turn

        return result


@dataclass
class AggregatedResults:
    """
    Container for aggregated simulation results across all patients.

    Attributes:
        model_name: Name of disclosure model
        n_simulations: Number of simulation runs (may differ from n_unique_patients
                       for models with multiple permutations per patient)
        n_unique_patients: Number of unique patients (for random model, this differs
                          from n_simulations)
        n_permutations_per_patient: Number of permutations per patient (1 for
                                    deterministic models, >1 for random model)
        k_by_turn_stats: DataFrame with k statistics by turn
        threshold_stats: DataFrame with threshold crossing statistics
        patient_level_stats: Optional dict with patient-level (not permutation) stats
        raw_results: List of individual SimulationResult objects

    Note:
        For the Random Ordering model, n_simulations = n_unique_patients * n_permutations.
        Statistics in k_by_turn_stats and threshold_stats are computed across all
        simulations (permutations), not unique patients. Use patient_level_stats for
        patient-level aggregations.
    """
    model_name: str
    n_simulations: int
    n_unique_patients: int
    n_permutations_per_patient: int
    k_by_turn_stats: pd.DataFrame
    threshold_stats: pd.DataFrame
    raw_results: List[SimulationResult]
    patient_level_stats: Optional[Dict[str, Any]] = None


class SimulationRunner:
    """
    Orchestrates disclosure simulations across patients and models.

    This class manages the simulation workflow:
        1. Initializes AnonymityEngine with patient population
        2. Runs simulations for each patient using specified models
        3. Collects and aggregates results
        4. Exports results for visualization

    Attributes:
        population (pd.DataFrame): Patient profiles DataFrame
        engine (AnonymityEngine): Anonymity computation engine
        thresholds (List[int]): Small-cell thresholds to track

    Example:
        >>> runner = SimulationRunner(patient_profiles, thresholds=[5, 11, 20])
        >>> results = runner.run_progressive_simulation()
        >>> runner.export_results(results, "results/progressive.csv")
    """

    def __init__(
        self,
        population: pd.DataFrame,
        thresholds: List[int] = [5, 11, 20],
        n_jobs: int = -1,
        random_seed: int = 42
    ):
        """
        Initialize the simulation runner.

        Args:
            population: Patient profiles DataFrame (index = patient_id)
            thresholds: k thresholds to track crossing turns
            n_jobs: Number of parallel workers (-1 = all cores)
            random_seed: Seed for reproducibility
        """
        self.population = population
        self.thresholds = thresholds
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        # Initialize anonymity engine
        logger.info("Initializing AnonymityEngine...")
        self.engine = AnonymityEngine(population)

        logger.info(f"SimulationRunner initialized with {len(population)} patients")

    def _simulate_single_patient(
        self,
        patient_id: str,
        model: DisclosureModel,
        value_frequencies: Optional[Dict] = None
    ) -> SimulationResult:
        """
        Run simulation for a single patient.

        Args:
            patient_id: Patient identifier
            model: Disclosure model to use
            value_frequencies: QI value frequencies (required for rarity model)

        Returns:
            SimulationResult for this patient
        """
        patient_profile = self.population.loc[patient_id]

        # Get disclosure sequence from model
        if isinstance(model, RarityOrderedModel):
            sequence = model.get_disclosure_sequence(patient_profile, value_frequencies)
        else:
            sequence = model.get_disclosure_sequence(patient_profile)

        # Compute k after each turn
        k_values = self.engine.compute_k_sequence(sequence)

        # Determine threshold crossings
        turns_to_thresholds = {t: None for t in self.thresholds}
        for i, k in enumerate(k_values, 1):
            for threshold in self.thresholds:
                if turns_to_thresholds[threshold] is None and k < threshold:
                    turns_to_thresholds[threshold] = i

        return SimulationResult(
            patient_id=patient_id,
            model_name=model.name,
            k_by_turn=k_values,
            turns_to_thresholds=turns_to_thresholds,
            final_k=k_values[-1] if k_values else self.engine.n_patients,
            n_turns=len(k_values),
            disclosure_sequence=sequence
        )

    def run_simulation(
        self,
        model: DisclosureModel,
        patient_sample: Optional[pd.Index] = None,
        show_progress: bool = True
    ) -> AggregatedResults:
        """
        Run simulation for all patients using specified model.

        Args:
            model: Disclosure model to use
            patient_sample: Optional subset of patient IDs to simulate.
                          If None, simulates all patients.
            show_progress: Whether to show progress bar

        Returns:
            AggregatedResults with all simulation data
        """
        # Determine patients to simulate
        patient_ids = patient_sample if patient_sample is not None else self.population.index
        n_patients = len(patient_ids)

        logger.info(f"Running {model.name} simulation for {n_patients} patients...")

        # Get value frequencies for rarity model
        value_frequencies = None
        if isinstance(model, RarityOrderedModel):
            value_frequencies = self.engine._value_frequencies

        # Run simulations
        results = []
        iterator = tqdm(patient_ids, desc=f"{model.name}") if show_progress else patient_ids

        for patient_id in iterator:
            result = self._simulate_single_patient(patient_id, model, value_frequencies)
            results.append(result)

        # Aggregate results
        aggregated = self._aggregate_results(results, model.name)

        logger.info(f"Completed {model.name} simulation")

        return aggregated

    def run_random_model_simulation(
        self,
        n_permutations: int = 100,
        patient_sample: Optional[pd.Index] = None,
        show_progress: bool = True
    ) -> AggregatedResults:
        """
        Run random ordering simulation with multiple permutations per patient.

        Args:
            n_permutations: Number of random permutations per patient
            patient_sample: Optional subset of patient IDs
            show_progress: Whether to show progress bar

        Returns:
            AggregatedResults with all simulation data
        """
        patient_ids = patient_sample if patient_sample is not None else self.population.index
        n_patients = len(patient_ids)

        logger.info(f"Running Random Ordering simulation ({n_permutations} permutations) "
                    f"for {n_patients} patients...")

        results = []
        iterator = tqdm(patient_ids, desc="Random Ordering") if show_progress else patient_ids

        for patient_id in iterator:
            patient_profile = self.population.loc[patient_id]

            # Run multiple permutations
            for perm_idx in range(n_permutations):
                model = RandomOrderingModel(seed=self.random_seed + stable_hash(patient_id) + perm_idx)
                sequence = model.get_disclosure_sequence(patient_profile)

                k_values = self.engine.compute_k_sequence(sequence)

                turns_to_thresholds = {t: None for t in self.thresholds}
                for i, k in enumerate(k_values, 1):
                    for threshold in self.thresholds:
                        if turns_to_thresholds[threshold] is None and k < threshold:
                            turns_to_thresholds[threshold] = i

                result = SimulationResult(
                    patient_id=f"{patient_id}_perm{perm_idx}",
                    model_name="Random Ordering",
                    k_by_turn=k_values,
                    turns_to_thresholds=turns_to_thresholds,
                    final_k=k_values[-1] if k_values else self.engine.n_patients,
                    n_turns=len(k_values),
                    disclosure_sequence=sequence
                )
                results.append(result)

        # Note: Pass n_unique_patients and n_permutations to distinguish from total simulations
        aggregated = self._aggregate_results(
            results,
            "Random Ordering",
            n_unique_patients=n_patients,
            n_permutations_per_patient=n_permutations
        )

        logger.info(f"Completed Random Ordering simulation: {n_patients} patients × "
                    f"{n_permutations} permutations = {len(results)} total simulations")

        return aggregated

    def run_all_models(
        self,
        patient_sample: Optional[pd.Index] = None,
        n_random_permutations: int = 100,
        show_progress: bool = True
    ) -> Dict[str, AggregatedResults]:
        """
        Run simulations for all three disclosure models.

        Args:
            patient_sample: Optional subset of patient IDs
            n_random_permutations: Permutations for random model
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping model names to AggregatedResults
        """
        results = {}

        # Progressive Refinement
        progressive = ProgressiveRefinementModel()
        results['progressive'] = self.run_simulation(
            progressive, patient_sample, show_progress
        )

        # Random Ordering
        results['random'] = self.run_random_model_simulation(
            n_random_permutations, patient_sample, show_progress
        )

        # Rarity-Ordered
        rarity = RarityOrderedModel()
        results['rarity'] = self.run_simulation(
            rarity, patient_sample, show_progress
        )

        return results

    def _aggregate_results(
        self,
        results: List[SimulationResult],
        model_name: str,
        n_unique_patients: Optional[int] = None,
        n_permutations_per_patient: int = 1
    ) -> AggregatedResults:
        """
        Aggregate individual results into summary statistics.

        Args:
            results: List of SimulationResult objects
            model_name: Name of the disclosure model
            n_unique_patients: Number of unique patients (defaults to len(results))
            n_permutations_per_patient: Number of permutations per patient

        Returns:
            AggregatedResults with summary statistics
        """
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame([r.to_dict() for r in results])

        # Determine patient counts
        n_simulations = len(results)
        if n_unique_patients is None:
            n_unique_patients = n_simulations

        # Compute k statistics by turn
        k_columns = [c for c in results_df.columns if c.startswith('k_turn_')]
        # Forward-fill k values for patients with fewer than max disclosure steps
        # (carry forward final k for patients with incomplete QI sequences)
        k_columns_sorted = sorted(k_columns, key=lambda x: int(x.split('_')[-1]))
        results_df[k_columns_sorted] = results_df[k_columns_sorted].ffill(axis=1)
        k_stats_list = []

        for col in sorted(k_columns, key=lambda x: int(x.split('_')[-1])):
            turn = int(col.split('_')[-1])
            k_values = results_df[col].dropna()

            if len(k_values) > 0:
                stats = {
                    'turn': turn,
                    'mean_k': k_values.mean(),
                    'median_k': k_values.median(),
                    'std_k': k_values.std(),
                    'q25_k': k_values.quantile(0.25),
                    'q75_k': k_values.quantile(0.75),
                    'min_k': k_values.min(),
                    'max_k': k_values.max(),
                    'pct_k_lt_5': (k_values < 5).mean() * 100,
                    'pct_k_lt_11': (k_values < 11).mean() * 100,
                    'pct_unique': (k_values == 1).mean() * 100,
                }
                k_stats_list.append(stats)

        k_by_turn_stats = pd.DataFrame(k_stats_list)

        # Compute threshold crossing statistics
        threshold_stats_list = []
        for threshold in self.thresholds:
            col = f'turns_to_k{threshold}'
            if col in results_df.columns:
                turns = results_df[col].dropna()

                stats = {
                    'threshold': threshold,
                    'pct_reached': (results_df[col].notna()).mean() * 100,
                    'mean_turns': turns.mean() if len(turns) > 0 else None,
                    'median_turns': turns.median() if len(turns) > 0 else None,
                    'std_turns': turns.std() if len(turns) > 0 else None,
                    'q25_turns': turns.quantile(0.25) if len(turns) > 0 else None,
                    'q75_turns': turns.quantile(0.75) if len(turns) > 0 else None,
                }
                threshold_stats_list.append(stats)

        threshold_stats = pd.DataFrame(threshold_stats_list)

        # Compute patient-level statistics for random model (multiple permutations)
        patient_level_stats = None
        if n_permutations_per_patient > 1:
            patient_level_stats = self._compute_patient_level_stats(
                results_df, n_unique_patients, n_permutations_per_patient
            )

        return AggregatedResults(
            model_name=model_name,
            n_simulations=n_simulations,
            n_unique_patients=n_unique_patients,
            n_permutations_per_patient=n_permutations_per_patient,
            k_by_turn_stats=k_by_turn_stats,
            threshold_stats=threshold_stats,
            raw_results=results,
            patient_level_stats=patient_level_stats
        )

    def _compute_patient_level_stats(
        self,
        results_df: pd.DataFrame,
        n_unique_patients: int,
        n_permutations_per_patient: int
    ) -> Dict[str, Any]:
        """
        Compute patient-level statistics from permutation-level results.

        For the random model, each patient has multiple permutation results.
        This method aggregates to provide patient-level statistics:
        - A patient "reaches threshold" if ANY of their permutations does
        - Median turns is computed among permutations that reached threshold

        Args:
            results_df: DataFrame with permutation-level results
            n_unique_patients: Number of unique patients
            n_permutations_per_patient: Number of permutations per patient

        Returns:
            Dict with patient-level statistics
        """
        # Extract base patient ID (remove _permN suffix)
        results_df = results_df.copy()
        results_df['base_patient_id'] = results_df['patient_id'].str.rsplit('_perm', n=1).str[0]

        patient_stats = {}

        for threshold in self.thresholds:
            col = f'turns_to_k{threshold}'
            if col not in results_df.columns:
                continue

            # Group by patient and check if ANY permutation reached threshold
            patient_reached = results_df.groupby('base_patient_id')[col].apply(
                lambda x: x.notna().any()
            )
            n_patients_reached = patient_reached.sum()
            pct_patients_reached = n_patients_reached / n_unique_patients * 100

            # Compute median turns across all permutations that reached threshold
            all_turns = results_df[col].dropna()
            median_turns = all_turns.median() if len(all_turns) > 0 else None

            patient_stats[f'pct_patients_reaching_k{threshold}'] = pct_patients_reached
            patient_stats[f'n_patients_reaching_k{threshold}'] = int(n_patients_reached)
            patient_stats[f'median_turns_to_k{threshold}_across_permutations'] = median_turns

        patient_stats['n_unique_patients'] = n_unique_patients
        patient_stats['n_permutations_per_patient'] = n_permutations_per_patient
        patient_stats['n_total_permutations'] = len(results_df)

        return patient_stats

    def export_results(
        self,
        results: AggregatedResults,
        output_dir: str,
        prefix: str = ""
    ) -> Dict[str, str]:
        """
        Export simulation results to CSV files.

        Args:
            results: AggregatedResults to export
            output_dir: Directory for output files
            prefix: Optional prefix for filenames

        Returns:
            Dict mapping result type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_prefix = f"{prefix}_" if prefix else ""
        model_slug = results.model_name.lower().replace(' ', '_')

        exported_files = {}

        # Export k by turn statistics
        k_stats_path = output_dir / f"{file_prefix}{model_slug}_k_by_turn.csv"
        results.k_by_turn_stats.to_csv(k_stats_path, index=False)
        exported_files['k_by_turn'] = str(k_stats_path)

        # Export threshold statistics
        threshold_path = output_dir / f"{file_prefix}{model_slug}_threshold_stats.csv"
        results.threshold_stats.to_csv(threshold_path, index=False)
        exported_files['threshold_stats'] = str(threshold_path)

        # Export raw results (without disclosure sequences for space)
        raw_df = pd.DataFrame([r.to_dict() for r in results.raw_results])
        raw_path = output_dir / f"{file_prefix}{model_slug}_raw_results.csv"
        raw_df.to_csv(raw_path, index=False)
        exported_files['raw_results'] = str(raw_path)

        logger.info(f"Exported results to {output_dir}")

        return exported_files

    def get_combined_k_by_turn(
        self,
        all_results: Dict[str, AggregatedResults]
    ) -> pd.DataFrame:
        """
        Combine k-by-turn statistics from multiple models.

        Args:
            all_results: Dict mapping model names to AggregatedResults

        Returns:
            DataFrame with k statistics for all models
        """
        combined = []

        for model_name, results in all_results.items():
            df = results.k_by_turn_stats.copy()
            df['model'] = model_name
            combined.append(df)

        return pd.concat(combined, ignore_index=True)

    def get_combined_threshold_stats(
        self,
        all_results: Dict[str, AggregatedResults]
    ) -> pd.DataFrame:
        """
        Combine threshold statistics from multiple models.

        Args:
            all_results: Dict mapping model names to AggregatedResults

        Returns:
            DataFrame with threshold stats for all models
        """
        combined = []

        for model_name, results in all_results.items():
            df = results.threshold_stats.copy()
            df['model'] = model_name
            combined.append(df)

        return pd.concat(combined, ignore_index=True)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create test population
    np.random.seed(42)
    n_test = 100

    test_pop = pd.DataFrame({
        'patient_id': [f'P{i:04d}' for i in range(n_test)],
        'age_decade': np.random.choice(['20-29', '30-39', '40-49', '50-59', '60-69'], n_test),
        'gender': np.random.choice(['M', 'F'], n_test),
        'race': np.random.choice(['white', 'black', 'asian', 'other'], n_test, p=[0.6, 0.2, 0.15, 0.05]),
        'ethnicity': np.random.choice(['nonhispanic', 'hispanic'], n_test, p=[0.85, 0.15]),
        'marital_status': np.random.choice(['M', 'S', 'W', 'D'], n_test),
        'primary_condition': np.random.choice([100, 200, 300, 400, 500], n_test, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'secondary_condition': np.random.choice([100, 200, 300, 'none'], n_test),
        'primary_medication': np.random.choice([1000, 2000, 3000, 'none'], n_test),
        'has_procedure': np.random.choice([True, False], n_test, p=[0.7, 0.3]),
        'has_allergy': np.random.choice([True, False], n_test, p=[0.4, 0.6]),
        'first_encounter_year': np.random.choice([2010, 2011, 2012, 2013], n_test),
    }).set_index('patient_id')

    print(f"Test population: {len(test_pop)} patients")
    print()

    # Initialize runner
    runner = SimulationRunner(test_pop, thresholds=[5, 11, 20])

    # Run progressive simulation
    print("=" * 60)
    print("Progressive Refinement Simulation")
    print("=" * 60)
    progressive_model = ProgressiveRefinementModel()
    results = runner.run_simulation(progressive_model)

    print(f"\nk by turn statistics:")
    print(results.k_by_turn_stats)

    print(f"\nThreshold crossing statistics:")
    print(results.threshold_stats)

    # Run all models
    print("\n" + "=" * 60)
    print("All Models Simulation")
    print("=" * 60)
    all_results = runner.run_all_models(n_random_permutations=10)

    print("\nCombined k by turn:")
    combined_k = runner.get_combined_k_by_turn(all_results)
    print(combined_k[['model', 'turn', 'median_k', 'pct_k_lt_5', 'pct_k_lt_11']])

    print("\nCombined threshold stats:")
    combined_thresh = runner.get_combined_threshold_stats(all_results)
    print(combined_thresh)
