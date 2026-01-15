"""
Disclosure models for multi-turn LLM chat simulation.

Defines how quasi-identifiers are revealed across conversation turns:
  - Progressive Refinement: Demographics first, then clinical details
  - Random Ordering: Random permutation of available QIs
  - Rarity-Ordered: Rarest QIs first (worst-case scenario)

Author: James Weatherhead, UTMB (jacweath@utmb.edu)
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import random

import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class DisclosureModel(ABC):
    """
    Abstract base class for disclosure models.

    A disclosure model defines the order in which quasi-identifiers are
    revealed across conversation turns. Subclasses implement different
    disclosure patterns to simulate various clinician behaviors.

    Attributes:
        name (str): Human-readable name of the model
        description (str): Brief description of the disclosure pattern
    """

    def __init__(self, name: str, description: str):
        """
        Initialize the disclosure model.

        Args:
            name: Model name for display and logging
            description: Brief description of disclosure pattern
        """
        self.name = name
        self.description = description

    @abstractmethod
    def get_disclosure_sequence(
        self,
        patient_profile: pd.Series,
        value_frequencies: Optional[Dict[str, Dict[Any, float]]] = None
    ) -> List[Tuple[str, Any]]:
        """
        Generate the sequence of QI disclosures for a patient.

        Args:
            patient_profile: Series containing patient's QI values
            value_frequencies: Optional dict mapping QI names to value frequency dicts

        Returns:
            List of (qi_name, qi_value) tuples in disclosure order
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ProgressiveRefinementModel(DisclosureModel):
    """
    Progressive refinement disclosure model.

    This model simulates a clinician who starts with general demographic
    context and progressively adds more specific clinical details. This
    is a common pattern in clinical LLM conversations where the clinician
    establishes patient context before asking clinical questions.

    Disclosure Sequence:
        Turn 1: age_decade + gender (basic demographics)
        Turn 2: race + ethnicity (extended demographics)
        Turn 3: marital_status (social context)
        Turn 4: primary_condition (main diagnosis)
        Turn 5: secondary_condition (comorbidity)
        Turn 6: primary_medication (treatment)
        Turn 7: has_procedure (intervention history)
        Turn 8: has_allergy (safety information)
        Turn 9: first_encounter_year (temporal context)

    Example:
        >>> model = ProgressiveRefinementModel()
        >>> sequence = model.get_disclosure_sequence(patient_profile)
        >>> # sequence[0] = ('age_decade', '40-49'), ('gender', 'F')
    """

    # Default turn structure: list of QI names per turn
    DEFAULT_TURN_STRUCTURE = [
        ['age_decade', 'gender'],           # Turn 1: Basic demographics
        ['race', 'ethnicity'],              # Turn 2: Extended demographics
        ['marital_status'],                 # Turn 3: Social context
        ['primary_condition'],              # Turn 4: Main diagnosis
        ['secondary_condition'],            # Turn 5: Comorbidity
        ['primary_medication'],             # Turn 6: Treatment
        ['has_procedure'],                  # Turn 7: Intervention history
        ['has_allergy'],                    # Turn 8: Safety information
        ['first_encounter_year'],           # Turn 9: Temporal context
    ]

    def __init__(self, turn_structure: Optional[List[List[str]]] = None):
        """
        Initialize the progressive refinement model.

        Args:
            turn_structure: Optional custom turn structure. If None, uses default.
        """
        super().__init__(
            name="Progressive Refinement",
            description="Demographics first, then clinical details progressively"
        )
        self.turn_structure = turn_structure or self.DEFAULT_TURN_STRUCTURE

    def get_disclosure_sequence(
        self,
        patient_profile: pd.Series,
        value_frequencies: Optional[Dict[str, Dict[Any, float]]] = None
    ) -> List[Tuple[str, Any]]:
        """
        Generate progressive disclosure sequence for a patient.

        Args:
            patient_profile: Series containing patient's QI values
            value_frequencies: Not used by this model

        Returns:
            List of (qi_name, qi_value) tuples in disclosure order
        """
        sequence = []

        for turn_qis in self.turn_structure:
            for qi_name in turn_qis:
                if qi_name in patient_profile.index:
                    qi_value = patient_profile[qi_name]
                    # Skip "none" or empty values
                    if qi_value is not None and qi_value != "none":
                        if isinstance(qi_value, (set, frozenset)) and len(qi_value) == 0:
                            continue
                        sequence.append((qi_name, qi_value))

        return sequence

    def get_turn_count(self) -> int:
        """Get the number of turns in the disclosure sequence."""
        return len(self.turn_structure)


class RandomOrderingModel(DisclosureModel):
    """
    Random ordering disclosure model.

    This model simulates unpredictable disclosure patterns where quasi-
    identifiers are revealed in random order. This is useful for estimating
    the distribution of turns required to reach small-cell territory across
    different disclosure patterns.

    Multiple random permutations can be generated per patient to estimate
    the variance in turns-to-threshold.

    Example:
        >>> model = RandomOrderingModel(seed=42)
        >>> sequence = model.get_disclosure_sequence(patient_profile)
        >>> # sequence order is randomized
    """

    # QIs to include in random ordering
    DEFAULT_QI_POOL = [
        'age_decade',
        'gender',
        'race',
        'ethnicity',
        'marital_status',
        'primary_condition',
        'secondary_condition',
        'primary_medication',
        'has_procedure',
        'has_allergy',
        'first_encounter_year',
    ]

    def __init__(
        self,
        qi_pool: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the random ordering model.

        Args:
            qi_pool: List of QI names to include. If None, uses default pool.
            seed: Random seed for reproducibility. If None, uses system random.
        """
        super().__init__(
            name="Random Ordering",
            description="Quasi-identifiers disclosed in random order"
        )
        self.qi_pool = qi_pool or self.DEFAULT_QI_POOL
        self.seed = seed
        self._rng = random.Random(seed)

    def get_disclosure_sequence(
        self,
        patient_profile: pd.Series,
        value_frequencies: Optional[Dict[str, Dict[Any, float]]] = None
    ) -> List[Tuple[str, Any]]:
        """
        Generate randomly ordered disclosure sequence for a patient.

        Args:
            patient_profile: Series containing patient's QI values
            value_frequencies: Not used by this model

        Returns:
            List of (qi_name, qi_value) tuples in random order
        """
        # Get available QIs with non-empty values
        available_qis = []
        for qi_name in self.qi_pool:
            if qi_name in patient_profile.index:
                qi_value = patient_profile[qi_name]
                if qi_value is not None and qi_value != "none":
                    if isinstance(qi_value, (set, frozenset)) and len(qi_value) == 0:
                        continue
                    available_qis.append((qi_name, qi_value))

        # Shuffle the list
        self._rng.shuffle(available_qis)

        return available_qis

    def get_multiple_sequences(
        self,
        patient_profile: pd.Series,
        n_permutations: int = 100
    ) -> List[List[Tuple[str, Any]]]:
        """
        Generate multiple random disclosure sequences for distribution analysis.

        Args:
            patient_profile: Series containing patient's QI values
            n_permutations: Number of random permutations to generate

        Returns:
            List of n_permutations disclosure sequences
        """
        sequences = []
        for _ in range(n_permutations):
            seq = self.get_disclosure_sequence(patient_profile)
            sequences.append(seq)
        return sequences

    def reset_seed(self, seed: int) -> None:
        """Reset the random seed for reproducibility."""
        self.seed = seed
        self._rng = random.Random(seed)


class RarityOrderedModel(DisclosureModel):
    """
    Rarity-ordered disclosure model (worst-case scenario).

    This model discloses quasi-identifiers in order of ascending population
    frequency (rarest first). This represents the worst-case scenario for
    privacy, showing how quickly k can collapse when distinctive clinical
    details are revealed early in a conversation.

    This model requires population-level value frequencies to rank QIs
    by rarity.

    Example:
        >>> model = RarityOrderedModel()
        >>> # Patient with rare cancer will have that disclosed first
        >>> sequence = model.get_disclosure_sequence(patient_profile, frequencies)
    """

    DEFAULT_QI_POOL = [
        'age_decade',
        'gender',
        'race',
        'ethnicity',
        'marital_status',
        'primary_condition',
        'secondary_condition',
        'primary_medication',
        'has_procedure',
        'has_allergy',
        'first_encounter_year',
    ]

    def __init__(self, qi_pool: Optional[List[str]] = None):
        """
        Initialize the rarity-ordered model.

        Args:
            qi_pool: List of QI names to include. If None, uses default pool.
        """
        super().__init__(
            name="Rarity-Ordered",
            description="Rarest quasi-identifiers disclosed first (worst case)"
        )
        self.qi_pool = qi_pool or self.DEFAULT_QI_POOL

    def get_disclosure_sequence(
        self,
        patient_profile: pd.Series,
        value_frequencies: Optional[Dict[str, Dict[Any, float]]] = None
    ) -> List[Tuple[str, Any]]:
        """
        Generate rarity-ordered disclosure sequence for a patient.

        QIs are ordered by ascending population frequency so that the
        rarest (most identifying) information is disclosed first.

        Args:
            patient_profile: Series containing patient's QI values
            value_frequencies: Dict mapping QI names to {value: frequency} dicts.
                             Required for rarity ordering.

        Returns:
            List of (qi_name, qi_value) tuples ordered by rarity (rarest first)

        Raises:
            ValueError: If value_frequencies is not provided
        """
        if value_frequencies is None:
            raise ValueError("RarityOrderedModel requires value_frequencies parameter")

        # Collect available QIs with their frequencies
        qi_with_freq = []
        for qi_name in self.qi_pool:
            if qi_name not in patient_profile.index:
                continue

            qi_value = patient_profile[qi_name]

            # Skip empty values
            if qi_value is None or qi_value == "none":
                continue
            if isinstance(qi_value, (set, frozenset)) and len(qi_value) == 0:
                continue

            # Get frequency
            if qi_name in value_frequencies:
                if isinstance(qi_value, (set, frozenset)):
                    # For set values, use minimum frequency among elements
                    if qi_value:
                        freqs = [
                            value_frequencies[qi_name].get(v, 1.0)
                            for v in qi_value
                        ]
                        freq = min(freqs)
                    else:
                        freq = 1.0
                else:
                    freq = value_frequencies[qi_name].get(qi_value, 1.0)
            else:
                freq = 1.0  # Default to common if no frequency data

            qi_with_freq.append((qi_name, qi_value, freq))

        # Sort by frequency (ascending = rarest first)
        qi_with_freq.sort(key=lambda x: x[2])

        # Return without frequency values
        return [(name, value) for name, value, _ in qi_with_freq]

    def get_rarity_breakdown(
        self,
        patient_profile: pd.Series,
        value_frequencies: Dict[str, Dict[Any, float]]
    ) -> List[Tuple[str, Any, float]]:
        """
        Get disclosure sequence with rarity scores for analysis.

        Args:
            patient_profile: Series containing patient's QI values
            value_frequencies: Dict mapping QI names to {value: frequency} dicts

        Returns:
            List of (qi_name, qi_value, frequency) tuples ordered by rarity
        """
        sequence = self.get_disclosure_sequence(patient_profile, value_frequencies)

        result = []
        for qi_name, qi_value in sequence:
            if qi_name in value_frequencies:
                if isinstance(qi_value, (set, frozenset)):
                    if qi_value:
                        freqs = [value_frequencies[qi_name].get(v, 1.0) for v in qi_value]
                        freq = min(freqs)
                    else:
                        freq = 1.0
                else:
                    freq = value_frequencies[qi_name].get(qi_value, 1.0)
            else:
                freq = 1.0
            result.append((qi_name, qi_value, freq))

        return result


def create_disclosure_model(
    model_type: str,
    **kwargs
) -> DisclosureModel:
    """
    Factory function to create disclosure model instances.

    Args:
        model_type: One of 'progressive', 'random', 'rarity'
        **kwargs: Additional arguments passed to model constructor

    Returns:
        DisclosureModel instance

    Example:
        >>> model = create_disclosure_model('progressive')
        >>> model = create_disclosure_model('random', seed=42)
    """
    model_map = {
        'progressive': ProgressiveRefinementModel,
        'random': RandomOrderingModel,
        'rarity': RarityOrderedModel,
    }

    if model_type.lower() not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: {list(model_map.keys())}")

    return model_map[model_type.lower()](**kwargs)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create sample patient profile
    sample_profile = pd.Series({
        'age_decade': '40-49',
        'gender': 'F',
        'race': 'asian',
        'ethnicity': 'nonhispanic',
        'marital_status': 'M',
        'primary_condition': 38341003,  # Hypertension
        'secondary_condition': 44054006,  # Diabetes
        'primary_medication': 316049,  # Lisinopril
        'has_procedure': True,
        'has_allergy': True,
        'first_encounter_year': 2010,
    })

    # Sample value frequencies
    sample_frequencies = {
        'age_decade': {'40-49': 0.15, '50-59': 0.18, '30-39': 0.12},
        'gender': {'M': 0.48, 'F': 0.52},
        'race': {'white': 0.60, 'black': 0.20, 'asian': 0.05, 'other': 0.15},
        'ethnicity': {'nonhispanic': 0.85, 'hispanic': 0.15},
        'marital_status': {'M': 0.45, 'S': 0.35, 'W': 0.10, 'D': 0.10},
        'primary_condition': {38341003: 0.20, 44054006: 0.15, 195662009: 0.05},
        'secondary_condition': {44054006: 0.15, 38341003: 0.20},
        'primary_medication': {316049: 0.10, 834060: 0.25},
        'has_procedure': {True: 0.70, False: 0.30},
        'has_allergy': {True: 0.40, False: 0.60},
        'first_encounter_year': {2010: 0.08, 2011: 0.09, 2012: 0.10},
    }

    print("Sample patient profile:")
    print(sample_profile)
    print()

    # Test Progressive Refinement Model
    print("=" * 60)
    print("Progressive Refinement Model")
    print("=" * 60)
    progressive = ProgressiveRefinementModel()
    seq = progressive.get_disclosure_sequence(sample_profile)
    for i, (qi_name, qi_value) in enumerate(seq, 1):
        print(f"  Turn {i}: {qi_name} = {qi_value}")

    # Test Random Ordering Model
    print("\n" + "=" * 60)
    print("Random Ordering Model (seed=42)")
    print("=" * 60)
    random_model = RandomOrderingModel(seed=42)
    seq = random_model.get_disclosure_sequence(sample_profile)
    for i, (qi_name, qi_value) in enumerate(seq, 1):
        print(f"  Turn {i}: {qi_name} = {qi_value}")

    # Test Rarity-Ordered Model
    print("\n" + "=" * 60)
    print("Rarity-Ordered Model")
    print("=" * 60)
    rarity_model = RarityOrderedModel()
    breakdown = rarity_model.get_rarity_breakdown(sample_profile, sample_frequencies)
    for i, (qi_name, qi_value, freq) in enumerate(breakdown, 1):
        print(f"  Turn {i}: {qi_name} = {qi_value} (freq={freq:.3f})")

    # Test factory function
    print("\n" + "=" * 60)
    print("Factory function test")
    print("=" * 60)
    for model_type in ['progressive', 'random', 'rarity']:
        model = create_disclosure_model(model_type)
        print(f"  Created: {model}")
