# =============================================================================
# display_utils.py
# =============================================================================
# Utilities for mapping codes to human-readable names for figures/examples.
#
# The simulation uses canonicalized codes internally to avoid synonym noise.
# This module provides display mappings for publication figures only.
#
# Author: James Weatherhead
# Institution: University of Texas Medical Branch (UTMB)
# =============================================================================

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, Set, FrozenSet


def build_code_lookups(data_dir: str) -> Dict[str, Dict]:
    """
    Build code-to-description lookup dictionaries from Synthea CSV files.

    Args:
        data_dir: Path to directory containing Synthea CSV files

    Returns:
        Dict with keys: 'conditions', 'medications', 'procedures', 'allergies'
        Each value is a dict mapping CODE -> DESCRIPTION
    """
    data_dir = Path(data_dir)

    lookups = {}

    # Conditions (SNOMED-CT codes)
    conditions = pd.read_csv(data_dir / "conditions.csv", low_memory=False)
    lookups['conditions'] = (
        conditions.dropna(subset=["CODE", "DESCRIPTION"])
        .drop_duplicates("CODE")
        .set_index("CODE")["DESCRIPTION"]
        .to_dict()
    )

    # Medications (RxNorm codes)
    meds = pd.read_csv(data_dir / "medications.csv", low_memory=False)
    lookups['medications'] = (
        meds.dropna(subset=["CODE", "DESCRIPTION"])
        .drop_duplicates("CODE")
        .set_index("CODE")["DESCRIPTION"]
        .to_dict()
    )

    # Procedures (SNOMED-CT codes)
    procs = pd.read_csv(data_dir / "procedures.csv", low_memory=False)
    lookups['procedures'] = (
        procs.dropna(subset=["CODE", "DESCRIPTION"])
        .drop_duplicates("CODE")
        .set_index("CODE")["DESCRIPTION"]
        .to_dict()
    )

    # Allergies (SNOMED-CT codes)
    allergies = pd.read_csv(data_dir / "allergies.csv", low_memory=False)
    lookups['allergies'] = (
        allergies.dropna(subset=["CODE", "DESCRIPTION"])
        .drop_duplicates("CODE")
        .set_index("CODE")["DESCRIPTION"]
        .to_dict()
    )

    return lookups


def code_to_name(
    code: Union[int, float, str],
    lookup: Dict,
    unknown: str = "unknown"
) -> str:
    """Map a single code to its human-readable name."""
    if code is None or code == "none" or (isinstance(code, float) and pd.isna(code)):
        return "none"
    try:
        return lookup.get(int(code), unknown)
    except (ValueError, TypeError):
        return str(code)


def codes_to_names(
    codes: Union[Set, FrozenSet, list, None],
    lookup: Dict,
    max_items: Optional[int] = None,
    unknown: str = "unknown"
) -> list:
    """
    Map a collection of codes to human-readable names.

    Args:
        codes: Collection of codes (frozenset, set, list)
        lookup: Code-to-description dictionary
        max_items: Maximum number of items to return (None = all)
        unknown: String to use for codes not in lookup

    Returns:
        List of human-readable names
    """
    if codes is None or (isinstance(codes, (set, frozenset)) and len(codes) == 0):
        return []

    names = []
    for code in codes:
        try:
            name = lookup.get(int(code), unknown)
            names.append(name)
        except (ValueError, TypeError):
            names.append(str(code))

    if max_items is not None:
        return names[:max_items]
    return names


def add_display_columns(
    profiles: pd.DataFrame,
    lookups: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Add human-readable display columns to patient profiles.

    This creates *_display columns for use in figures while preserving
    the original code-based columns for simulation.

    Args:
        profiles: Patient profiles DataFrame with code-based columns
        lookups: Code lookup dictionaries from build_code_lookups()

    Returns:
        DataFrame with additional *_display columns
    """
    df = profiles.copy()

    # Primary condition display
    df["primary_condition_display"] = df["primary_condition"].apply(
        lambda c: code_to_name(c, lookups['conditions'])
    )

    # Secondary condition display
    df["secondary_condition_display"] = df["secondary_condition"].apply(
        lambda c: code_to_name(c, lookups['conditions'])
    )

    # Primary medication display
    df["primary_medication_display"] = df["primary_medication"].apply(
        lambda c: code_to_name(c, lookups['medications'])
    )

    # Condition codes display (list of names)
    df["conditions_display"] = df["condition_codes"].apply(
        lambda codes: codes_to_names(codes, lookups['conditions'])
    )

    # Medication codes display
    df["medications_display"] = df["medication_codes"].apply(
        lambda codes: codes_to_names(codes, lookups['medications'])
    )

    # Procedure codes display
    df["procedures_display"] = df["procedure_codes"].apply(
        lambda codes: codes_to_names(codes, lookups['procedures'])
    )

    # Allergy codes display
    df["allergies_display"] = df["allergy_codes"].apply(
        lambda codes: codes_to_names(codes, lookups['allergies'])
    )

    return df


def format_patient_example(
    patient_row: pd.Series,
    lookups: Dict[str, Dict],
    max_items: int = 3
) -> Dict[str, str]:
    """
    Format a patient row for display in figures/examples.

    Args:
        patient_row: Single patient row from profiles DataFrame
        lookups: Code lookup dictionaries
        max_items: Max items to show for list fields

    Returns:
        Dict with human-readable field names and values
    """
    example = {}

    # Demographics (already human-readable)
    example["Age"] = patient_row.get("age_decade", "unknown")
    example["Gender"] = patient_row.get("gender", "unknown")
    example["Race"] = patient_row.get("race", "unknown")
    example["Ethnicity"] = patient_row.get("ethnicity", "unknown")
    example["Marital Status"] = patient_row.get("marital_status", "unknown")

    # Conditions
    primary_cond = patient_row.get("primary_condition")
    example["Primary Condition"] = code_to_name(primary_cond, lookups['conditions'])

    secondary_cond = patient_row.get("secondary_condition")
    example["Secondary Condition"] = code_to_name(secondary_cond, lookups['conditions'])

    cond_codes = patient_row.get("condition_codes", frozenset())
    cond_names = codes_to_names(cond_codes, lookups['conditions'], max_items)
    if len(cond_codes) > max_items:
        example["All Conditions"] = f"{cond_names} + {len(cond_codes) - max_items} more"
    else:
        example["All Conditions"] = cond_names

    # Medications
    primary_med = patient_row.get("primary_medication")
    example["Primary Medication"] = code_to_name(primary_med, lookups['medications'])

    med_codes = patient_row.get("medication_codes", frozenset())
    med_names = codes_to_names(med_codes, lookups['medications'], max_items)
    if len(med_codes) > max_items:
        example["All Medications"] = f"{med_names} + {len(med_codes) - max_items} more"
    else:
        example["All Medications"] = med_names

    # Procedures
    proc_codes = patient_row.get("procedure_codes", frozenset())
    proc_names = codes_to_names(proc_codes, lookups['procedures'], max_items)
    example["Procedures"] = proc_names if proc_names else "none"

    # Allergies
    allergy_codes = patient_row.get("allergy_codes", frozenset())
    allergy_names = codes_to_names(allergy_codes, lookups['allergies'], max_items)
    example["Allergies"] = allergy_names if allergy_names else "none"

    # Encounter info
    example["First Encounter Year"] = patient_row.get("first_encounter_year", "unknown")
    example["Encounter Count"] = patient_row.get("encounter_count", 0)

    return example


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_loader import DataLoader

    data_dir = os.environ.get('SYNTHEA_DATA_DIR', './data')

    print("Building code lookups...")
    lookups = build_code_lookups(data_dir)
    print(f"  Conditions: {len(lookups['conditions'])} codes")
    print(f"  Medications: {len(lookups['medications'])} codes")
    print(f"  Procedures: {len(lookups['procedures'])} codes")
    print(f"  Allergies: {len(lookups['allergies'])} codes")

    print("\nLoading patient profiles...")
    loader = DataLoader(data_dir)
    profiles = loader.build_patient_profiles()

    print("\nExample patient (with display names):")
    print("=" * 60)

    # Find a patient with conditions, meds, and allergies
    for patient_id, row in profiles.iterrows():
        if row['condition_count'] > 2 and row['medication_count'] > 0 and row['allergy_count'] > 0:
            example = format_patient_example(row, lookups)
            for field, value in example.items():
                print(f"  {field:25}: {value}")
            break
