"""
Centralized path configuration for CT Recon project.

All scripts should import paths from this module rather than using hardcoded paths.
"""

from pathlib import Path

# Project root directory (parent of ct_core/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
SCANS_DIR = DATA_DIR / "scans"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"
TIFF_OUTPUT_DIR = DATA_DIR / "tiff_output"

# Base model comparison directories
BASE_MODEL_DIR = PROJECT_ROOT / "base_models"
SHARED_DIR = BASE_MODEL_DIR / "shared"
SINOGRAM_DATASET_DIR = SHARED_DIR / "sinogram_dataset"
GROUND_TRUTH_DIR = SHARED_DIR / "ground_truth"

# Model-specific directories (convenience functions)
def get_model_dir(model_name: str) -> Path:
    """Get the directory for a specific base model (lama, mat, deepfill, repaint)."""
    return BASE_MODEL_DIR / "models" / model_name

def get_model_data_dir(model_name: str) -> Path:
    """Get the data directory for a specific base model."""
    return get_model_dir(model_name) / "data"

def get_model_metrics_dir(model_name: str) -> Path:
    """Get the metrics directory for a specific base model."""
    return get_model_dir(model_name) / "metrics"

# Metric calculators
METRIC_CALCULATORS_DIR = PROJECT_ROOT / "metric_calculators"
METRIC_RESULTS_DIR = METRIC_CALCULATORS_DIR / "metric_results"

# Helper function to ensure directories exist
def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
