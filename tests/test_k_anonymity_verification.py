#!/usr/bin/env python3
"""
Unit tests to verify k-anonymity calculations are correct.

This test verifies that the k values reported in our simulation
match direct DataFrame filtering on the population.

Test case: Patient 8d112137-37e7-4614-ab38-f233925e0c13
- Age decade: 60-69
- Gender: M
- Race: white
- Ethnicity: french
- Marital status: M
- Primary condition: Prediabetes (code: 15777000)
"""

import os
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.anonymity_engine import AnonymityEngine


class TestKAnonymityVerification:
    """Verify k-anonymity calculations against direct DataFrame queries."""

    @pytest.fixture(scope="class")
    def data(self):
        """Load patient profiles once for all tests."""
        data_dir = os.environ.get('SYNTHEA_DATA_DIR', './data')
        loader = DataLoader(data_dir)
        profiles = loader.build_patient_profiles()
        engine = AnonymityEngine(profiles)
        return profiles, engine

    def test_total_population(self, data):
        """Verify total population size."""
        profiles, engine = data
        assert len(profiles) == 133262, f"Expected 133,262 patients, got {len(profiles)}"
        print(f"\n✓ Total population: {len(profiles):,}")

    def test_k_after_age_decade(self, data):
        """Verify k after disclosing age_decade = 60-69."""
        profiles, engine = data

        # Direct DataFrame filter
        mask = profiles['age_decade'] == '60.0-69.0'
        direct_count = mask.sum()

        # Via AnonymityEngine
        engine_k = engine.compute_k({'age_decade': '60.0-69.0'})

        print(f"\n  Age decade 60-69:")
        print(f"    Direct filter count: {direct_count:,}")
        print(f"    AnonymityEngine k:   {engine_k:,}")

        assert direct_count == engine_k, f"Mismatch: direct={direct_count}, engine={engine_k}"
        print(f"  ✓ k after age_decade (60-69) = {engine_k:,}")

    def test_k_after_age_and_gender(self, data):
        """Verify k after disclosing age_decade + gender."""
        profiles, engine = data

        # Direct DataFrame filter
        mask = (profiles['age_decade'] == '60.0-69.0') & (profiles['gender'] == 'M')
        direct_count = mask.sum()

        # Via AnonymityEngine
        engine_k = engine.compute_k({
            'age_decade': '60.0-69.0',
            'gender': 'M'
        })

        print(f"\n  Age decade 60-69 + Male:")
        print(f"    Direct filter count: {direct_count:,}")
        print(f"    AnonymityEngine k:   {engine_k:,}")

        assert direct_count == engine_k, f"Mismatch: direct={direct_count}, engine={engine_k}"
        print(f"  ✓ k after age_decade + gender = {engine_k:,}")

    def test_k_after_demographics(self, data):
        """Verify k after full demographics (age, gender, race, ethnicity, marital)."""
        profiles, engine = data

        # Direct DataFrame filter
        mask = (
            (profiles['age_decade'] == '60.0-69.0') &
            (profiles['gender'] == 'M') &
            (profiles['race'] == 'white') &
            (profiles['ethnicity'] == 'french') &
            (profiles['marital_status'] == 'M')
        )
        direct_count = mask.sum()

        # Via AnonymityEngine
        engine_k = engine.compute_k({
            'age_decade': '60.0-69.0',
            'gender': 'M',
            'race': 'white',
            'ethnicity': 'french',
            'marital_status': 'M'
        })

        print(f"\n  Full demographics (60s, M, white, french, married):")
        print(f"    Direct filter count: {direct_count:,}")
        print(f"    AnonymityEngine k:   {engine_k:,}")

        assert direct_count == engine_k, f"Mismatch: direct={direct_count}, engine={engine_k}"
        print(f"  ✓ k after full demographics = {engine_k:,}")

    def test_k_after_primary_condition(self, data):
        """Verify k after demographics + primary condition (prediabetes)."""
        profiles, engine = data

        # Prediabetes SNOMED code
        prediabetes_code = 15777000

        # Direct DataFrame filter
        mask = (
            (profiles['age_decade'] == '60.0-69.0') &
            (profiles['gender'] == 'M') &
            (profiles['race'] == 'white') &
            (profiles['ethnicity'] == 'french') &
            (profiles['marital_status'] == 'M') &
            (profiles['primary_condition'] == prediabetes_code)
        )
        direct_count = mask.sum()

        # Via AnonymityEngine
        engine_k = engine.compute_k({
            'age_decade': '60.0-69.0',
            'gender': 'M',
            'race': 'white',
            'ethnicity': 'french',
            'marital_status': 'M',
            'primary_condition': prediabetes_code
        })

        print(f"\n  Demographics + Prediabetes:")
        print(f"    Direct filter count: {direct_count:,}")
        print(f"    AnonymityEngine k:   {engine_k:,}")

        assert direct_count == engine_k, f"Mismatch: direct={direct_count}, engine={engine_k}"
        print(f"  ✓ k after demographics + primary_condition = {engine_k:,}")

    def test_progressive_k_sequence(self, data):
        """Verify the full progressive disclosure sequence matches raw results."""
        profiles, engine = data

        # Patient 8d112137-37e7-4614-ab38-f233925e0c13
        patient_id = '8d112137-37e7-4614-ab38-f233925e0c13'
        patient = profiles.loc[patient_id]

        print(f"\n  Progressive disclosure for patient {patient_id[:8]}...:")
        print(f"  " + "-" * 60)

        # Build up constraints progressively
        constraints = {}
        qi_sequence = [
            ('age_decade', patient['age_decade']),
            ('gender', patient['gender']),
            ('race', patient['race']),
            ('ethnicity', patient['ethnicity']),
            ('marital_status', patient['marital_status']),
            ('primary_condition', patient['primary_condition']),
            ('secondary_condition', patient['secondary_condition']),
            ('primary_medication', patient['primary_medication']),
            ('has_procedure', patient['has_procedure']),
            ('has_allergy', patient['has_allergy']),
            ('first_encounter_year', patient['first_encounter_year']),
        ]

        k_values = []
        for qi_name, qi_value in qi_sequence:
            constraints[qi_name] = qi_value
            k = engine.compute_k(constraints)
            k_values.append(k)
            print(f"    After {qi_name:25} = {str(qi_value):20} → k = {k:,}")

        # Load raw results and compare
        raw = pd.read_csv("results/simulation_output/progressive_progressive_refinement_raw_results.csv")
        patient_row = raw[raw['patient_id'] == patient_id].iloc[0]

        expected_k = [
            int(patient_row['k_turn_1']),
            int(patient_row['k_turn_2']),
            int(patient_row['k_turn_3']),
            int(patient_row['k_turn_4']),
            int(patient_row['k_turn_5']),
            int(patient_row['k_turn_6']),
            int(patient_row['k_turn_7']),
            int(patient_row['k_turn_8']),
            int(patient_row['k_turn_9']),
            int(patient_row['k_turn_10']),
            int(patient_row['k_turn_11']),
        ]

        print(f"\n  Comparison with saved results:")
        print(f"    Computed: {k_values}")
        print(f"    Saved:    {expected_k}")

        assert k_values == expected_k, f"K sequence mismatch!\nComputed: {k_values}\nExpected: {expected_k}"
        print(f"  ✓ All k values match!")


class TestPopulationDistributions:
    """Verify population distributions to sanity-check the data."""

    @pytest.fixture(scope="class")
    def profiles(self):
        """Load patient profiles."""
        data_dir = os.environ.get('SYNTHEA_DATA_DIR', './data')
        loader = DataLoader(data_dir)
        return loader.build_patient_profiles()

    def test_age_distribution(self, profiles):
        """Check age decade distribution."""
        print("\n  Age decade distribution:")
        age_counts = profiles['age_decade'].value_counts().sort_index()
        for age, count in age_counts.items():
            pct = count / len(profiles) * 100
            print(f"    {age}: {count:,} ({pct:.1f}%)")

        # Verify 60-69 count
        sixties = profiles[profiles['age_decade'] == '60.0-69.0']
        print(f"\n  ✓ Patients in 60s: {len(sixties):,}")

    def test_gender_distribution(self, profiles):
        """Check gender distribution."""
        print("\n  Gender distribution:")
        gender_counts = profiles['gender'].value_counts()
        for gender, count in gender_counts.items():
            pct = count / len(profiles) * 100
            print(f"    {gender}: {count:,} ({pct:.1f}%)")

    def test_ethnicity_distribution(self, profiles):
        """Check ethnicity distribution - this explains the dramatic k drop."""
        print("\n  Ethnicity distribution (top 10):")
        eth_counts = profiles['ethnicity'].value_counts().head(10)
        for eth, count in eth_counts.items():
            pct = count / len(profiles) * 100
            print(f"    {eth}: {count:,} ({pct:.1f}%)")

        # Check french specifically
        french = profiles[profiles['ethnicity'] == 'french']
        print(f"\n  ✓ French ethnicity: {len(french):,} ({len(french)/len(profiles)*100:.2f}%)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
