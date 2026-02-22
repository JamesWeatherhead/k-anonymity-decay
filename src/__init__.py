"""K-anonymity decay analysis for multi-turn clinical LLM conversations."""

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
