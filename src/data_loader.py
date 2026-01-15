# =============================================================================
# data_loader.py
# =============================================================================
# Module for loading and preprocessing Synthea synthetic EHR data.
#
# This module handles:
#   - Loading raw CSV files from the Synthea dataset
#   - Computing derived quasi-identifiers (age decades, etc.)
#   - Aggregating clinical data to patient-level profiles
#   - Building the unified patient profile DataFrame for k-anonymity analysis
#
# Data Source:
#   Synthea synthetic EHR dataset (https://mitre.box.com/shared/static/3bo45m48ocpzp8fc0tp005vax7l93xji.gz)
#   Version: CSV 3.0, 24 May 2017
#   Population: 133,262 synthetic patients (after removing malformed rows)
#
# Author: James Weatherhead
# Institution: University of Texas Medical Branch (UTMB)
# Contact: jacweath@utmb.edu
# =============================================================================

import os
import logging
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

# Suppress FutureWarning about fillna downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Configure logging
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and preprocesses Synthea synthetic EHR data for k-anonymity analysis.

    This class handles the complete data pipeline from raw CSV files to a unified
    patient profile DataFrame containing all quasi-identifiers.

    Attributes:
        data_dir (str): Path to directory containing Synthea CSV files
        reference_date (pd.Timestamp): Reference date for age calculation
        config (dict): Configuration parameters

    Example:
        >>> loader = DataLoader("/path/to/synthea/csv")
        >>> profiles = loader.build_patient_profiles()
        >>> print(f"Loaded {len(profiles)} patients")
    """

    def __init__(
        self,
        data_dir: str,
        reference_date: str = "2017-12-31",
        config: Optional[Dict] = None
    ):
        """
        Initialize the DataLoader.

        Args:
            data_dir: Path to directory containing Synthea CSV files
            reference_date: Reference date for age calculation (YYYY-MM-DD format)
            config: Optional configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.reference_date = pd.Timestamp(reference_date)
        self.config = config or {}

        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Cache for loaded DataFrames
        self._cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"DataLoader initialized with data_dir: {self.data_dir}")

    def load_patients(self) -> pd.DataFrame:
        """
        Load patient demographics from patients.csv.

        Returns:
            DataFrame with columns: ID, BIRTHDATE, DEATHDATE, GENDER, RACE,
            ETHNICITY, MARITAL, BIRTHPLACE, ADDRESS, etc.

        Note:
            Direct identifiers (SSN, DRIVERS, PASSPORT, NAME fields) are loaded
            but will be excluded from quasi-identifier analysis.
        """
        if "patients" not in self._cache:
            filepath = self.data_dir / "patients.csv"
            logger.info(f"Loading patients from {filepath}")

            df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')
            df["BIRTHDATE"] = pd.to_datetime(df["BIRTHDATE"], errors="coerce")
            df["DEATHDATE"] = pd.to_datetime(df["DEATHDATE"], errors="coerce")

            self._cache["patients"] = df
            logger.info(f"Loaded {len(df)} patients")

        return self._cache["patients"]

    def load_conditions(self) -> pd.DataFrame:
        """
        Load patient conditions/diagnoses from conditions.csv.

        Returns:
            DataFrame with columns: START, STOP, PATIENT, ENCOUNTER, CODE, DESCRIPTION

        Note:
            CODE is a SNOMED-CT code. STOP is null for chronic/ongoing conditions.
        """
        if "conditions" not in self._cache:
            filepath = self.data_dir / "conditions.csv"
            logger.info(f"Loading conditions from {filepath}")

            df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')
            df["START"] = pd.to_datetime(df["START"], errors="coerce")
            df["STOP"] = pd.to_datetime(df["STOP"], errors="coerce")

            self._cache["conditions"] = df
            logger.info(f"Loaded {len(df)} condition records")

        return self._cache["conditions"]

    def load_medications(self) -> pd.DataFrame:
        """
        Load patient medications from medications.csv.

        Returns:
            DataFrame with columns: START, STOP, PATIENT, ENCOUNTER, CODE,
            DESCRIPTION, REASONCODE, REASONDESCRIPTION
        """
        if "medications" not in self._cache:
            filepath = self.data_dir / "medications.csv"
            logger.info(f"Loading medications from {filepath}")

            df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')
            df["START"] = pd.to_datetime(df["START"], errors="coerce")
            df["STOP"] = pd.to_datetime(df["STOP"], errors="coerce")

            self._cache["medications"] = df
            logger.info(f"Loaded {len(df)} medication records")

        return self._cache["medications"]

    def load_procedures(self) -> pd.DataFrame:
        """
        Load patient procedures from procedures.csv.

        Returns:
            DataFrame with columns: DATE, PATIENT, ENCOUNTER, CODE, DESCRIPTION,
            REASONCODE, REASONDESCRIPTION
        """
        if "procedures" not in self._cache:
            filepath = self.data_dir / "procedures.csv"
            logger.info(f"Loading procedures from {filepath}")

            df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

            self._cache["procedures"] = df
            logger.info(f"Loaded {len(df)} procedure records")

        return self._cache["procedures"]

    def load_allergies(self) -> pd.DataFrame:
        """
        Load patient allergies from allergies.csv.

        Returns:
            DataFrame with columns: START, STOP, PATIENT, ENCOUNTER, CODE, DESCRIPTION
        """
        if "allergies" not in self._cache:
            filepath = self.data_dir / "allergies.csv"
            logger.info(f"Loading allergies from {filepath}")

            df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')
            df["START"] = pd.to_datetime(df["START"], errors="coerce")
            df["STOP"] = pd.to_datetime(df["STOP"], errors="coerce")

            self._cache["allergies"] = df
            logger.info(f"Loaded {len(df)} allergy records")

        return self._cache["allergies"]

    def load_encounters(self) -> pd.DataFrame:
        """
        Load patient encounters from encounters.csv.

        Returns:
            DataFrame with columns: ID, DATE, PATIENT, CODE, DESCRIPTION,
            REASONCODE, REASONDESCRIPTION
        """
        if "encounters" not in self._cache:
            filepath = self.data_dir / "encounters.csv"
            logger.info(f"Loading encounters from {filepath}")

            df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

            self._cache["encounters"] = df
            logger.info(f"Loaded {len(df)} encounter records")

        return self._cache["encounters"]

    def compute_age_decade(self, birthdate: pd.Series) -> pd.Series:
        """
        Compute age decade (10-year bins) from birthdate.

        This implements HIPAA Safe Harbor's age generalization requirement,
        binning ages into 10-year ranges (0-9, 10-19, ..., 80-89, 90+).

        Args:
            birthdate: Series of birthdates

        Returns:
            Series of age decade strings (e.g., "30-39", "40-49", "90+")
        """
        # Calculate age in years as of reference date
        age_days = (self.reference_date - birthdate).dt.days
        age_years = age_days // 365

        # Bin into decades
        def age_to_decade(age: int) -> str:
            if pd.isna(age) or age < 0:
                return "unknown"
            if age >= 90:
                return "90+"  # HIPAA Safe Harbor: combine 90+
            decade_start = int((age // 10) * 10)
            decade_end = int(decade_start + 9)
            return f"{decade_start}-{decade_end}"

        return age_years.apply(age_to_decade)

    def aggregate_conditions_per_patient(
        self,
        conditions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate condition data to patient level.

        Computes:
            - primary_condition: Most frequent condition SNOMED code
            - secondary_condition: Second most frequent condition
            - condition_codes: Frozenset of all condition codes
            - condition_count: Number of distinct conditions
            - has_chronic_condition: Whether patient has condition without STOP date

        Args:
            conditions_df: Raw conditions DataFrame

        Returns:
            DataFrame indexed by PATIENT with aggregated condition columns
        """
        logger.info("Aggregating conditions per patient...")

        # Count condition occurrences per patient
        condition_counts = (
            conditions_df
            .groupby(["PATIENT", "CODE"])
            .size()
            .reset_index(name="count")
        )

        # Get primary (most frequent) condition per patient
        primary = (
            condition_counts
            .sort_values(["PATIENT", "count"], ascending=[True, False])
            .groupby("PATIENT")
            .first()
            .reset_index()
            [["PATIENT", "CODE"]]
            .rename(columns={"CODE": "primary_condition"})
        )

        # Get secondary condition per patient
        def get_nth_condition(group, n):
            if len(group) > n:
                return group.iloc[n]["CODE"]
            return None

        secondary = (
            condition_counts
            .sort_values(["PATIENT", "count"], ascending=[True, False])
            .groupby("PATIENT")
            .apply(lambda x: get_nth_condition(x, 1), include_groups=False)
            .reset_index()
            .rename(columns={0: "secondary_condition"})
        )

        # Get all condition codes as frozenset
        all_conditions = (
            conditions_df
            .groupby("PATIENT")["CODE"]
            .apply(frozenset)
            .reset_index()
            .rename(columns={"CODE": "condition_codes"})
        )

        # Count distinct conditions
        condition_count = (
            conditions_df
            .groupby("PATIENT")["CODE"]
            .nunique()
            .reset_index()
            .rename(columns={"CODE": "condition_count"})
        )

        # Check for chronic conditions (no STOP date)
        has_chronic = (
            conditions_df[conditions_df["STOP"].isna()]
            .groupby("PATIENT")
            .size()
            .reset_index(name="chronic_count")
        )
        has_chronic["has_chronic_condition"] = has_chronic["chronic_count"] > 0

        # Merge all aggregations
        result = primary.merge(secondary, on="PATIENT", how="left")
        result = result.merge(all_conditions, on="PATIENT", how="left")
        result = result.merge(condition_count, on="PATIENT", how="left")
        result = result.merge(
            has_chronic[["PATIENT", "has_chronic_condition"]],
            on="PATIENT",
            how="left"
        )
        result["has_chronic_condition"] = result["has_chronic_condition"].fillna(False)

        return result

    def aggregate_medications_per_patient(
        self,
        medications_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate medication data to patient level.

        Computes:
            - primary_medication: Most frequent medication code
            - medication_codes: Frozenset of all medication codes
            - medication_count: Number of distinct medications

        Args:
            medications_df: Raw medications DataFrame

        Returns:
            DataFrame indexed by PATIENT with aggregated medication columns
        """
        logger.info("Aggregating medications per patient...")

        # Count medication occurrences per patient
        med_counts = (
            medications_df
            .groupby(["PATIENT", "CODE"])
            .size()
            .reset_index(name="count")
        )

        # Get primary (most frequent) medication per patient
        primary = (
            med_counts
            .sort_values(["PATIENT", "count"], ascending=[True, False])
            .groupby("PATIENT")
            .first()
            .reset_index()
            [["PATIENT", "CODE"]]
            .rename(columns={"CODE": "primary_medication"})
        )

        # Get all medication codes as frozenset
        all_meds = (
            medications_df
            .groupby("PATIENT")["CODE"]
            .apply(frozenset)
            .reset_index()
            .rename(columns={"CODE": "medication_codes"})
        )

        # Count distinct medications
        med_count = (
            medications_df
            .groupby("PATIENT")["CODE"]
            .nunique()
            .reset_index()
            .rename(columns={"CODE": "medication_count"})
        )

        # Merge all aggregations
        result = primary.merge(all_meds, on="PATIENT", how="left")
        result = result.merge(med_count, on="PATIENT", how="left")

        return result

    def aggregate_procedures_per_patient(
        self,
        procedures_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate procedure data to patient level.

        Computes:
            - procedure_codes: Frozenset of all procedure codes
            - procedure_count: Number of distinct procedures
            - has_procedure: Boolean indicator

        Args:
            procedures_df: Raw procedures DataFrame

        Returns:
            DataFrame indexed by PATIENT with aggregated procedure columns
        """
        logger.info("Aggregating procedures per patient...")

        # Get all procedure codes as frozenset
        all_procs = (
            procedures_df
            .groupby("PATIENT")["CODE"]
            .apply(frozenset)
            .reset_index()
            .rename(columns={"CODE": "procedure_codes"})
        )

        # Count distinct procedures
        proc_count = (
            procedures_df
            .groupby("PATIENT")["CODE"]
            .nunique()
            .reset_index()
            .rename(columns={"CODE": "procedure_count"})
        )

        result = all_procs.merge(proc_count, on="PATIENT", how="left")
        result["has_procedure"] = True

        return result

    def aggregate_allergies_per_patient(
        self,
        allergies_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate allergy data to patient level.

        Computes:
            - allergy_codes: Frozenset of all allergy codes
            - allergy_count: Number of distinct allergies
            - has_allergy: Boolean indicator

        Args:
            allergies_df: Raw allergies DataFrame

        Returns:
            DataFrame indexed by PATIENT with aggregated allergy columns
        """
        logger.info("Aggregating allergies per patient...")

        # Get all allergy codes as frozenset
        all_allergies = (
            allergies_df
            .groupby("PATIENT")["CODE"]
            .apply(frozenset)
            .reset_index()
            .rename(columns={"CODE": "allergy_codes"})
        )

        # Count distinct allergies
        allergy_count = (
            allergies_df
            .groupby("PATIENT")["CODE"]
            .nunique()
            .reset_index()
            .rename(columns={"CODE": "allergy_count"})
        )

        result = all_allergies.merge(allergy_count, on="PATIENT", how="left")
        result["has_allergy"] = True

        return result

    def aggregate_encounters_per_patient(
        self,
        encounters_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate encounter data to patient level.

        Computes:
            - first_encounter_year: Year of first healthcare encounter
            - last_encounter_year: Year of most recent encounter
            - encounter_count: Total number of encounters
            - care_span_years: Duration of care in years

        Args:
            encounters_df: Raw encounters DataFrame

        Returns:
            DataFrame indexed by PATIENT with aggregated encounter columns
        """
        logger.info("Aggregating encounters per patient...")

        encounter_agg = (
            encounters_df
            .groupby("PATIENT")
            .agg(
                first_encounter_date=("DATE", "min"),
                last_encounter_date=("DATE", "max"),
                encounter_count=("DATE", "count")
            )
            .reset_index()
        )

        encounter_agg["first_encounter_year"] = (
            encounter_agg["first_encounter_date"].dt.year
        )
        encounter_agg["last_encounter_year"] = (
            encounter_agg["last_encounter_date"].dt.year
        )
        encounter_agg["care_span_years"] = (
            (encounter_agg["last_encounter_date"] -
             encounter_agg["first_encounter_date"]).dt.days / 365.25
        ).round(1)

        return encounter_agg[[
            "PATIENT", "first_encounter_year", "last_encounter_year",
            "encounter_count", "care_span_years"
        ]]

    def build_patient_profiles(self) -> pd.DataFrame:
        """
        Build unified patient profile DataFrame with all quasi-identifiers.

        This is the main entry point for data loading. It:
            1. Loads all raw CSV files
            2. Computes derived demographic quasi-identifiers
            3. Aggregates clinical data to patient level
            4. Merges all data into a single DataFrame

        Returns:
            DataFrame with one row per patient containing all quasi-identifiers:
                - patient_id: Unique patient identifier
                - age_decade: Age in 10-year bins (e.g., "30-39")
                - gender: M or F
                - race: Race category
                - ethnicity: Ethnicity category
                - marital_status: Marital status code
                - primary_condition: Most frequent SNOMED condition code
                - secondary_condition: Second most frequent condition code
                - condition_codes: Frozenset of all condition codes
                - condition_count: Number of distinct conditions
                - has_chronic_condition: Boolean
                - primary_medication: Most frequent medication code
                - medication_codes: Frozenset of all medication codes
                - medication_count: Number of distinct medications
                - procedure_codes: Frozenset of all procedure codes
                - procedure_count: Number of distinct procedures
                - has_procedure: Boolean
                - allergy_codes: Frozenset of all allergy codes
                - allergy_count: Number of distinct allergies
                - has_allergy: Boolean
                - first_encounter_year: Year of first encounter
                - encounter_count: Total number of encounters

        Example:
            >>> loader = DataLoader("/path/to/csv")
            >>> profiles = loader.build_patient_profiles()
            >>> print(profiles.columns.tolist())
        """
        logger.info("Building patient profiles...")

        # Load raw data
        patients = self.load_patients()
        conditions = self.load_conditions()
        medications = self.load_medications()
        procedures = self.load_procedures()
        allergies = self.load_allergies()
        encounters = self.load_encounters()

        # Start with demographics
        logger.info("Processing demographic quasi-identifiers...")
        profiles = patients[["ID", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY", "MARITAL"]].copy()
        profiles = profiles.rename(columns={
            "ID": "patient_id",
            "GENDER": "gender",
            "RACE": "race",
            "ETHNICITY": "ethnicity",
            "MARITAL": "marital_status"
        })

        # Compute age decade
        profiles["age_decade"] = self.compute_age_decade(profiles["BIRTHDATE"])
        profiles = profiles.drop(columns=["BIRTHDATE"])

        # Handle missing values in demographics
        profiles["gender"] = profiles["gender"].fillna("unknown")
        profiles["race"] = profiles["race"].fillna("unknown")
        profiles["ethnicity"] = profiles["ethnicity"].fillna("unknown")
        profiles["marital_status"] = profiles["marital_status"].fillna("unknown")

        # Aggregate clinical data
        condition_agg = self.aggregate_conditions_per_patient(conditions)
        medication_agg = self.aggregate_medications_per_patient(medications)
        procedure_agg = self.aggregate_procedures_per_patient(procedures)
        allergy_agg = self.aggregate_allergies_per_patient(allergies)
        encounter_agg = self.aggregate_encounters_per_patient(encounters)

        # Merge all aggregations
        logger.info("Merging all quasi-identifiers...")
        profiles = profiles.merge(
            condition_agg.rename(columns={"PATIENT": "patient_id"}),
            on="patient_id",
            how="left"
        )
        profiles = profiles.merge(
            medication_agg.rename(columns={"PATIENT": "patient_id"}),
            on="patient_id",
            how="left"
        )
        profiles = profiles.merge(
            procedure_agg.rename(columns={"PATIENT": "patient_id"}),
            on="patient_id",
            how="left"
        )
        profiles = profiles.merge(
            allergy_agg.rename(columns={"PATIENT": "patient_id"}),
            on="patient_id",
            how="left"
        )
        profiles = profiles.merge(
            encounter_agg.rename(columns={"PATIENT": "patient_id"}),
            on="patient_id",
            how="left"
        )

        # Fill missing values for patients without clinical records
        profiles["primary_condition"] = profiles["primary_condition"].fillna("none")
        profiles["secondary_condition"] = profiles["secondary_condition"].fillna("none")
        profiles["condition_codes"] = profiles["condition_codes"].apply(
            lambda x: x if isinstance(x, frozenset) else frozenset()
        )
        profiles["condition_count"] = profiles["condition_count"].fillna(0).astype(int)
        profiles["has_chronic_condition"] = profiles["has_chronic_condition"].fillna(False)

        profiles["primary_medication"] = profiles["primary_medication"].fillna("none")
        profiles["medication_codes"] = profiles["medication_codes"].apply(
            lambda x: x if isinstance(x, frozenset) else frozenset()
        )
        profiles["medication_count"] = profiles["medication_count"].fillna(0).astype(int)

        profiles["procedure_codes"] = profiles["procedure_codes"].apply(
            lambda x: x if isinstance(x, frozenset) else frozenset()
        )
        profiles["procedure_count"] = profiles["procedure_count"].fillna(0).astype(int)
        profiles["has_procedure"] = profiles["has_procedure"].fillna(False)

        profiles["allergy_codes"] = profiles["allergy_codes"].apply(
            lambda x: x if isinstance(x, frozenset) else frozenset()
        )
        profiles["allergy_count"] = profiles["allergy_count"].fillna(0).astype(int)
        profiles["has_allergy"] = profiles["has_allergy"].fillna(False)

        profiles["first_encounter_year"] = profiles["first_encounter_year"].fillna(0).astype(int)
        profiles["encounter_count"] = profiles["encounter_count"].fillna(0).astype(int)

        # Set patient_id as index
        profiles = profiles.set_index("patient_id")

        logger.info(f"Built profiles for {len(profiles)} patients with {len(profiles.columns)} quasi-identifiers")

        return profiles


def build_patient_profiles(data_dir: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function to build patient profiles in one call.

    Args:
        data_dir: Path to directory containing Synthea CSV files
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        DataFrame with patient profiles

    Example:
        >>> profiles = build_patient_profiles("/path/to/csv")
    """
    loader = DataLoader(data_dir, **kwargs)
    return loader.build_patient_profiles()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Synthea data and build patient profiles")
    parser.add_argument("data_dir", help="Path to Synthea CSV directory")
    parser.add_argument("--output", "-o", help="Output file path for profiles (CSV format)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    profiles = build_patient_profiles(args.data_dir)
    print(f"\nLoaded {len(profiles)} patient profiles")
    print(f"Columns: {profiles.columns.tolist()}")
    print(f"\nSample profile:\n{profiles.iloc[0]}")

    if args.output:
        # Convert frozenset columns to string for CSV export
        export_df = profiles.copy()
        for col in ["condition_codes", "medication_codes", "procedure_codes", "allergy_codes"]:
            export_df[col] = export_df[col].apply(lambda x: ",".join(map(str, x)) if x else "")
        export_df.to_csv(args.output)
        print(f"\nSaved profiles to {args.output}")
