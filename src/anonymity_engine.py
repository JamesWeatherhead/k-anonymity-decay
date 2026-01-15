# =============================================================================
# anonymity_engine.py
# =============================================================================
# Core module for computing k-anonymity set sizes.
#
# This module implements the anonymity set size (k) computation, which is
# central to measuring disclosure risk in de-identified clinical data.
#
# Key Concept:
#   Given a set of disclosed quasi-identifiers (QIs), k is the number of
#   individuals in the population who share ALL the same QI values. Lower k
#   indicates higher disclosure risk; k=1 means unique identification.
#
# References:
#   - Sweeney, L. (2002). k-anonymity: A model for protecting privacy.
#   - El Emam, K. (2013). Guide to the De-Identification of Personal Health
#     Information.
#
# Author: James Weatherhead
# Institution: University of Texas Medical Branch (UTMB)
# Contact: jacweath@utmb.edu
# =============================================================================

import logging
from typing import Dict, Any, List, Optional, Tuple, Set, FrozenSet
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)


class AnonymityEngine:
    """
    Engine for computing k-anonymity set sizes across a patient population.

    This class pre-computes indices and frequency tables to enable efficient
    k computation during simulation. The key optimization is pre-building
    indexes for common constraint patterns to avoid repeated DataFrame filtering.

    Attributes:
        population (pd.DataFrame): Patient population with quasi-identifiers
        qi_columns (List[str]): List of quasi-identifier column names
        frequency_cache (Dict): Cache of pre-computed value frequencies

    Example:
        >>> engine = AnonymityEngine(patient_profiles)
        >>> k = engine.compute_k({'age_decade': '40-49', 'gender': 'F'})
        >>> print(f"k = {k}")
    """

    def __init__(self, population: pd.DataFrame, qi_columns: Optional[List[str]] = None):
        """
        Initialize the AnonymityEngine.

        Args:
            population: DataFrame with patient profiles (one row per patient)
            qi_columns: List of column names to consider as quasi-identifiers.
                       If None, uses all columns except index.
        """
        self.population = population.copy()
        self.n_patients = len(population)

        # Determine quasi-identifier columns
        if qi_columns is None:
            self.qi_columns = population.columns.tolist()
        else:
            self.qi_columns = qi_columns

        # Pre-compute frequency tables for each QI
        self._frequency_tables: Dict[str, pd.Series] = {}
        self._build_frequency_tables()

        # Pre-compute value counts for rarity ranking
        self._value_frequencies: Dict[str, Dict[Any, float]] = {}
        self._compute_value_frequencies()

        logger.info(f"AnonymityEngine initialized with {self.n_patients} patients "
                    f"and {len(self.qi_columns)} quasi-identifiers")

    def _build_frequency_tables(self) -> None:
        """
        Pre-compute frequency tables for each quasi-identifier.

        This enables O(1) lookup of how many patients have a specific QI value.
        """
        logger.info("Building frequency tables...")

        for col in self.qi_columns:
            if col in self.population.columns:
                # Handle frozenset columns specially
                if self.population[col].apply(lambda x: isinstance(x, (set, frozenset))).any():
                    # For set columns, we can't do simple value_counts
                    # Instead, track which patients have each value
                    continue
                else:
                    self._frequency_tables[col] = self.population[col].value_counts()

    def _compute_value_frequencies(self) -> None:
        """
        Compute population frequency for each QI value.

        Frequency is the proportion of patients having that value.
        Used for rarity-ordered disclosure model.
        """
        logger.info("Computing value frequencies...")

        for col in self.qi_columns:
            if col not in self.population.columns:
                continue

            self._value_frequencies[col] = {}

            # Check if this is a set-valued column
            sample_val = self.population[col].iloc[0] if len(self.population) > 0 else None
            if isinstance(sample_val, (set, frozenset)):
                # For set columns, compute frequency of each individual code
                all_codes = set()
                for codes in self.population[col]:
                    if isinstance(codes, (set, frozenset)):
                        all_codes.update(codes)

                for code in all_codes:
                    count = self.population[col].apply(
                        lambda x: code in x if isinstance(x, (set, frozenset)) else False
                    ).sum()
                    self._value_frequencies[col][code] = count / self.n_patients
            else:
                # For scalar columns, use value_counts
                counts = self.population[col].value_counts()
                for val, count in counts.items():
                    self._value_frequencies[col][val] = count / self.n_patients

    def get_value_frequency(self, qi_name: str, qi_value: Any) -> float:
        """
        Get the population frequency of a specific QI value.

        Args:
            qi_name: Name of the quasi-identifier
            qi_value: Value to look up

        Returns:
            Frequency as proportion (0.0 to 1.0)
        """
        if qi_name not in self._value_frequencies:
            return 0.0

        if isinstance(qi_value, (set, frozenset)):
            # For set constraints, compute joint frequency
            # This is the frequency of patients having ALL codes in the set
            if not qi_value:
                return 1.0
            freqs = [self._value_frequencies[qi_name].get(v, 0.0) for v in qi_value]
            # Return minimum as upper bound (actual joint frequency may be lower)
            return min(freqs) if freqs else 0.0
        else:
            return self._value_frequencies[qi_name].get(qi_value, 0.0)

    def compute_k(self, constraints: Dict[str, Any]) -> int:
        """
        Compute anonymity set size k for a given set of constraints.

        k is defined as the number of patients in the population who match
        ALL the disclosed quasi-identifier constraints.

        Algorithm:
            1. Start with all patients (k = N)
            2. For each constraint, filter to patients matching that constraint
            3. Return count of remaining patients

        Args:
            constraints: Dictionary mapping QI column names to disclosed values.
                        Values can be:
                        - Scalar: exact match required
                        - Set/frozenset: all elements must be present in patient's set
                        - None: constraint is skipped

        Returns:
            k: Number of patients matching all constraints

        Example:
            >>> constraints = {
            ...     'age_decade': '40-49',
            ...     'gender': 'F',
            ...     'primary_condition': 38341003  # Hypertension
            ... }
            >>> k = engine.compute_k(constraints)
        """
        if not constraints:
            return self.n_patients

        # Start with boolean mask of all True
        mask = pd.Series(True, index=self.population.index)

        for qi_name, qi_value in constraints.items():
            # Skip None or empty constraints
            if qi_value is None:
                continue

            if qi_name not in self.population.columns:
                logger.warning(f"QI column '{qi_name}' not found in population")
                continue

            # Apply constraint based on value type
            if isinstance(qi_value, (set, frozenset)):
                # Set membership: patient's set must contain ALL values in constraint
                if qi_value:  # Only filter if non-empty set
                    mask &= self.population[qi_name].apply(
                        lambda x: qi_value.issubset(x) if isinstance(x, (set, frozenset)) else False
                    )
            elif isinstance(qi_value, bool):
                # Boolean exact match
                mask &= (self.population[qi_name] == qi_value)
            else:
                # Scalar exact match
                mask &= (self.population[qi_name] == qi_value)

        return mask.sum()

    def compute_k_batch(
        self,
        constraint_list: List[Dict[str, Any]],
        show_progress: bool = False
    ) -> List[int]:
        """
        Compute k for multiple constraint sets efficiently.

        Args:
            constraint_list: List of constraint dictionaries
            show_progress: Whether to show progress bar

        Returns:
            List of k values corresponding to each constraint set
        """
        iterator = constraint_list
        if show_progress:
            iterator = tqdm(constraint_list, desc="Computing k values")

        return [self.compute_k(constraints) for constraints in iterator]

    def compute_k_sequence(
        self,
        qi_sequence: List[Tuple[str, Any]]
    ) -> List[int]:
        """
        Compute k after each turn of a disclosure sequence.

        This simulates multi-turn disclosure where each turn adds a new
        quasi-identifier constraint. Returns k values after each turn.

        Args:
            qi_sequence: List of (qi_name, qi_value) tuples in disclosure order

        Returns:
            List of k values [k_1, k_2, ..., k_n] where k_t is anonymity set
            size after turn t

        Example:
            >>> sequence = [
            ...     ('age_decade', '40-49'),
            ...     ('gender', 'F'),
            ...     ('primary_condition', 38341003)
            ... ]
            >>> k_values = engine.compute_k_sequence(sequence)
            >>> # k_values[0] = k after disclosing age
            >>> # k_values[1] = k after disclosing age + gender
            >>> # k_values[2] = k after disclosing age + gender + condition
        """
        k_values = []
        accumulated_constraints = {}

        for qi_name, qi_value in qi_sequence:
            # Accumulate constraints
            if qi_value is not None:
                accumulated_constraints[qi_name] = qi_value

            # Compute k with all constraints so far
            k = self.compute_k(accumulated_constraints)
            k_values.append(k)

        return k_values

    def compute_delta_k(
        self,
        base_constraints: Dict[str, Any],
        new_qi_name: str,
        new_qi_value: Any
    ) -> Tuple[int, int, int]:
        """
        Compute the change in k when adding a new constraint.

        Args:
            base_constraints: Existing constraints
            new_qi_name: Name of new QI to add
            new_qi_value: Value of new QI

        Returns:
            Tuple of (k_before, k_after, delta_k)
            where delta_k = k_before - k_after (positive means k decreased)
        """
        k_before = self.compute_k(base_constraints)

        extended_constraints = base_constraints.copy()
        extended_constraints[new_qi_name] = new_qi_value

        k_after = self.compute_k(extended_constraints)
        delta_k = k_before - k_after

        return k_before, k_after, delta_k

    def find_uniquely_identified(
        self,
        constraints: Dict[str, Any]
    ) -> pd.Index:
        """
        Find patients who would be uniquely identified (k=1) given constraints.

        Args:
            constraints: QI constraints to apply

        Returns:
            Index of uniquely identified patients
        """
        mask = pd.Series(True, index=self.population.index)

        for qi_name, qi_value in constraints.items():
            if qi_value is None:
                continue
            if qi_name not in self.population.columns:
                continue

            if isinstance(qi_value, (set, frozenset)):
                if qi_value:
                    mask &= self.population[qi_name].apply(
                        lambda x: qi_value.issubset(x) if isinstance(x, (set, frozenset)) else False
                    )
            else:
                mask &= (self.population[qi_name] == qi_value)

        # Get matching patients
        matching_patients = self.population[mask].index

        # Find those who are unique among the matching set
        unique_patients = []
        for patient_id in matching_patients:
            patient_profile = self.population.loc[patient_id]
            patient_constraints = {
                qi: patient_profile[qi] for qi in constraints.keys()
                if qi in self.population.columns
            }
            if self.compute_k(patient_constraints) == 1:
                unique_patients.append(patient_id)

        return pd.Index(unique_patients)

    def get_qi_value_distribution(self, qi_name: str) -> pd.Series:
        """
        Get the distribution of values for a quasi-identifier.

        Args:
            qi_name: Name of the quasi-identifier

        Returns:
            Series with value counts, sorted by frequency
        """
        if qi_name in self._frequency_tables:
            return self._frequency_tables[qi_name].sort_values(ascending=False)
        else:
            return self.population[qi_name].value_counts()

    def get_rare_values(self, qi_name: str, threshold: float = 0.001) -> List[Any]:
        """
        Get values that appear in less than threshold proportion of population.

        Args:
            qi_name: Name of the quasi-identifier
            threshold: Frequency threshold (default 0.1% of population)

        Returns:
            List of rare values
        """
        if qi_name not in self._value_frequencies:
            return []

        rare_values = [
            val for val, freq in self._value_frequencies[qi_name].items()
            if freq < threshold
        ]

        return sorted(rare_values, key=lambda x: self._value_frequencies[qi_name][x])

    def get_population_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the population and QI distributions.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_patients': self.n_patients,
            'n_quasi_identifiers': len(self.qi_columns),
            'qi_columns': self.qi_columns,
            'qi_cardinality': {},
            'rare_value_counts': {}
        }

        for col in self.qi_columns:
            if col in self.population.columns:
                # Count unique values
                sample = self.population[col].iloc[0] if len(self.population) > 0 else None
                if isinstance(sample, (set, frozenset)):
                    # For set columns, count unique individual codes
                    all_codes = set()
                    for codes in self.population[col]:
                        if isinstance(codes, (set, frozenset)):
                            all_codes.update(codes)
                    summary['qi_cardinality'][col] = len(all_codes)
                else:
                    summary['qi_cardinality'][col] = self.population[col].nunique()

                # Count rare values
                rare = self.get_rare_values(col)
                summary['rare_value_counts'][col] = len(rare)

        return summary


def compute_k(
    population: pd.DataFrame,
    constraints: Dict[str, Any]
) -> int:
    """
    Convenience function to compute k without instantiating AnonymityEngine.

    Note: For repeated computations, instantiate AnonymityEngine once and reuse.

    Args:
        population: Patient population DataFrame
        constraints: QI constraints dictionary

    Returns:
        k value (anonymity set size)
    """
    engine = AnonymityEngine(population)
    return engine.compute_k(constraints)


if __name__ == "__main__":
    # Simple test with synthetic data
    import argparse

    logging.basicConfig(level=logging.INFO)

    # Create test population
    np.random.seed(42)
    n_test = 1000

    test_pop = pd.DataFrame({
        'patient_id': [f'P{i:04d}' for i in range(n_test)],
        'age_decade': np.random.choice(['20-29', '30-39', '40-49', '50-59', '60-69'], n_test),
        'gender': np.random.choice(['M', 'F'], n_test),
        'race': np.random.choice(['white', 'black', 'asian', 'other'], n_test, p=[0.6, 0.2, 0.15, 0.05]),
        'condition': np.random.choice([100, 200, 300, 400, 500], n_test, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    }).set_index('patient_id')

    print("Test population:")
    print(test_pop.head())
    print(f"\nTotal patients: {len(test_pop)}")

    # Create engine
    engine = AnonymityEngine(test_pop)

    # Test k computation
    print("\n--- Testing k computation ---")

    # No constraints
    k0 = engine.compute_k({})
    print(f"k with no constraints: {k0} (should be {n_test})")

    # Single constraint
    k1 = engine.compute_k({'gender': 'F'})
    print(f"k with gender=F: {k1}")

    # Multiple constraints
    k2 = engine.compute_k({'gender': 'F', 'age_decade': '40-49'})
    print(f"k with gender=F, age_decade=40-49: {k2}")

    # Rare constraint
    k3 = engine.compute_k({'condition': 500})
    print(f"k with rare condition (500): {k3}")

    # Sequence
    print("\n--- Testing k sequence ---")
    sequence = [
        ('age_decade', '40-49'),
        ('gender', 'F'),
        ('race', 'asian'),
        ('condition', 500)
    ]
    k_values = engine.compute_k_sequence(sequence)
    print(f"k sequence: {k_values}")

    # Summary
    print("\n--- Population summary ---")
    summary = engine.get_population_summary()
    print(f"QI cardinalities: {summary['qi_cardinality']}")
