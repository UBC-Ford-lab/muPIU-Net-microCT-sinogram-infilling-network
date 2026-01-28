"""
CT Core Library

Core utilities for CT reconstruction project including:
- VFF file I/O (vff_io)
- HU calibration (calibration)
- Field correction (field_correction)
- TIFF conversion (tiff_converter)
- Path constants (paths)
"""

from . import vff_io
from . import calibration
from . import field_correction
from . import tiff_converter
from . import paths

# Convenience re-exports of commonly used functions
from .vff_io import read_vff, read_vff_header, read_vff_data, VFFDataset
from .calibration import (
    parse_calibration_from_xml,
    load_calibration_fields,
    flat_field_correction,
    log_transform_transmission,
    log_transform_projections,
    convert_to_hounsfield_units,
)
from .field_correction import (
    bright_dark_field_correction,
    write_vff,
    initiate_bright_dark_correction,
)
from .tiff_converter import save_vff_to_tiff
from .paths import PROJECT_ROOT, DATA_DIR, SCANS_DIR, RESULTS_DIR, MODELS_DIR

__all__ = [
    # Modules
    'vff_io',
    'calibration',
    'field_correction',
    'tiff_converter',
    'paths',
    # VFF I/O
    'read_vff',
    'read_vff_header',
    'read_vff_data',
    'VFFDataset',
    # Calibration
    'parse_calibration_from_xml',
    'load_calibration_fields',
    'flat_field_correction',
    'log_transform_transmission',
    'log_transform_projections',
    'convert_to_hounsfield_units',
    # Field correction
    'bright_dark_field_correction',
    'write_vff',
    'initiate_bright_dark_correction',
    # TIFF
    'save_vff_to_tiff',
    # Paths
    'PROJECT_ROOT',
    'DATA_DIR',
    'SCANS_DIR',
    'RESULTS_DIR',
    'MODELS_DIR',
]
