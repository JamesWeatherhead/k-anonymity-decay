# =============================================================================
# Quasi-Identifier Accumulation Study
# =============================================================================
#
# This package implements the analysis of anonymity set size (k) collapse
# across multi-turn de-identified clinical LLM conversations.
#
# Modules:
#   - data_loader: Load and preprocess Synthea CSV files
#   - anonymity_engine: Core k-anonymity computation
#   - disclosure_models: Three disclosure model implementations
#   - simulation_runner: Orchestrate multi-patient simulations
#   - visualization: Generate publication-quality figures
#
# Author: James Weatherhead
# Institution: University of Texas Medical Branch (UTMB)
# =============================================================================

__version__ = "1.0.0"
__author__ = "James Weatherhead"
__email__ = "jacweath@utmb.edu"

from .data_loader import DataLoader, build_patient_profiles
from .anonymity_engine import AnonymityEngine, compute_k
from .disclosure_models import (
    ProgressiveRefinementModel,
    RandomOrderingModel,
    RarityOrderedModel
)
from .simulation_runner import SimulationRunner
from .visualization import Visualizer

__all__ = [
    "DataLoader",
    "build_patient_profiles",
    "AnonymityEngine",
    "compute_k",
    "ProgressiveRefinementModel",
    "RandomOrderingModel",
    "RarityOrderedModel",
    "SimulationRunner",
    "Visualizer"
]
