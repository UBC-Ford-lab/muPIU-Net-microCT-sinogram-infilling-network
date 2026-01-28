"""
HU Calibration Module for CT Reconstruction Pipeline

Provides log transformation and Hounsfield Unit calibration for CT projections.

Standard CT reconstruction requires:
1. Log transformation: p = -log(I/I₀) to convert intensity to line integrals
2. FDK reconstruction yields linear attenuation coefficient μ
3. HU conversion: HU = (μ - μ_water) / μ_water × 1000

Author: Falk Wiegmann, University of British Columbia
Date: January 2026
"""

import numpy as np
import xmltodict
import os
import sys
from typing import Dict, Optional, Tuple

# Import VFF reader using relative import
from .vff_io import read_vff


def parse_calibration_from_xml(xml_path: str) -> Dict[str, float]:
    """
    Extract AirValue, WaterValue, and BoneHU from scan XML.

    XML Path: Series/Tasks/Recon/TaskParams/Advanced/

    Args:
        xml_path: Path to scan.xml file

    Returns:
        Dictionary with 'air_value', 'water_value', 'bone_hu'
    """
    with open(xml_path, 'r') as f:
        header = xmltodict.parse(f.read())

    # Navigate to Advanced calibration parameters
    # Path: Series -> Tasks -> Recon -> TaskParams -> Advanced
    try:
        recon = header['Series']['Tasks']['Recon']
        advanced = recon['TaskParams']['Advanced']

        return {
            'air_value': float(advanced.get('AirValue', 1.0)),
            'water_value': float(advanced.get('WaterValue', 1.0)),
            'bone_hu': float(advanced.get('BoneHU', 3100))
        }
    except (KeyError, TypeError) as e:
        print(f"Warning: Could not parse calibration from XML: {e}")
        print("Using default values: air_value=1.0, water_value=1.0, bone_hu=3100")
        return {
            'air_value': 1.0,
            'water_value': 1.0,
            'bone_hu': 3100
        }


def load_calibration_fields(scan_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load bright and dark field calibration images from scan folder.

    These are used for proper flat-field correction:
    - bright.vff: Unattenuated beam reference (I₀) - air scan with no object
    - dark.vff: Electronic noise/offset reference

    Args:
        scan_folder: Path to original scan folder (e.g., scans/Scan_1681/)

    Returns:
        Tuple of (bright_field, dark_field) as 2D numpy arrays [height, width]

    Raises:
        FileNotFoundError: If bright.vff or dark.vff not found
    """
    bright_path = os.path.join(scan_folder, 'bright.vff')
    dark_path = os.path.join(scan_folder, 'dark.vff')

    if not os.path.exists(bright_path):
        raise FileNotFoundError(f"Bright field not found: {bright_path}")
    if not os.path.exists(dark_path):
        raise FileNotFoundError(f"Dark field not found: {dark_path}")

    # Read VFF files
    _, bright_field = read_vff(bright_path, verbose=False)
    _, dark_field = read_vff(dark_path, verbose=False)

    # Average if multiple frames (calibration images are often averaged)
    if bright_field.ndim == 3:
        bright_field = np.mean(bright_field, axis=0)
    if dark_field.ndim == 3:
        dark_field = np.mean(dark_field, axis=0)

    return bright_field.astype(np.float32), dark_field.astype(np.float32)


def flat_field_correction(
    projections: np.ndarray,
    bright_field: np.ndarray,
    dark_field: np.ndarray,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Apply flat-field correction to convert raw intensities to transmission.

    T = (I - I_dark) / (I_bright - I_dark)

    This normalizes detector response variations and converts raw intensities
    to transmission values (fraction of beam transmitted through object).

    Args:
        projections: Raw detector intensities [n_angles, height, width]
        bright_field: Unattenuated beam reference [height, width]
        dark_field: Electronic noise reference [height, width]
        epsilon: Small value for numerical stability

    Returns:
        Transmission values clipped to range [epsilon, 1.0]
    """
    # Compute denominator once (beam intensity minus electronic offset)
    denominator = bright_field - dark_field + epsilon

    # Apply correction to each projection
    # Broadcasting: projections[n, h, w] / denominator[h, w]
    transmission = (projections - dark_field) / denominator

    # Clip to valid transmission range
    # Values > 1 can occur from noise, values < epsilon prevent log(0)
    transmission = np.clip(transmission, epsilon, 1.0)

    return transmission.astype(np.float32)


def log_transform_transmission(transmission: np.ndarray) -> np.ndarray:
    """
    Convert transmission to line integrals of attenuation.

    p = -log(T) where T is transmission (already normalized by flat-field)

    This converts transmission ratios to line integrals of the linear
    attenuation coefficient μ, as required for FDK reconstruction.

    Physical interpretation:
        T = exp(-∫μ dl) → p = -log(T) = ∫μ dl

    Args:
        transmission: Flat-field corrected transmission values [n_angles, h, w]
                     Values should be in range (0, 1]

    Returns:
        Line integrals of linear attenuation coefficient
        Range typically [0, ~5] for medical CT
    """
    return -np.log(transmission).astype(np.float32)


def log_transform_projections(
    projections: np.ndarray,
    air_value: float,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Apply log transformation to convert raw intensities to line integrals.

    p = -log(I / I₀) where I₀ is the unattenuated beam intensity (air_value)

    This converts transmitted X-ray intensity to line integrals of the
    linear attenuation coefficient, which is required for proper FDK
    reconstruction.

    Args:
        projections: Raw detector intensities [n_angles, height, width]
        air_value: Calibration value representing unattenuated beam intensity I₀
        epsilon: Small value to prevent log(0) for numerical stability

    Returns:
        Log-transformed projections (line integrals of attenuation)
    """
    # Normalize by air value to get transmission T = I/I₀
    transmission = projections / (air_value + epsilon)

    # Clamp to prevent log of negative/zero values (noise can cause these)
    transmission = np.clip(transmission, epsilon, None)

    # Apply negative log: p = -log(T) = -log(I/I₀)
    line_integrals = -np.log(transmission)

    return line_integrals.astype(np.float32)


def estimate_mu_water(
    air_value: float,
    water_value: float,
    calibration_path_length: float = 100.0
) -> float:
    """
    Estimate linear attenuation coefficient of water from calibration values.

    If water_value represents I_water/I₀ after passing through a known
    water path length L_cal:

        μ_water = -log(water_value / air_value) / L_cal

    Args:
        air_value: Calibration value for unattenuated beam
        water_value: Calibration value after passing through water
        calibration_path_length: Path length through water in mm (default 100mm)

    Returns:
        Estimated μ_water in mm⁻¹

    Note:
        Literature value for water at 80 keV: μ ≈ 0.184 cm⁻¹ = 0.00184 mm⁻¹
        The calibration_path_length needs to be verified with scanner documentation.
    """
    if air_value <= 0 or water_value <= 0:
        print("Warning: Invalid calibration values. Using literature mu_water.")
        return 0.00184  # Literature value at 80 keV in mm⁻¹

    transmission_ratio = water_value / air_value

    if transmission_ratio >= 1.0:
        print("Warning: water_value >= air_value suggests incorrect calibration.")
        print("Using literature mu_water value.")
        return 0.00184

    mu_water = -np.log(transmission_ratio) / calibration_path_length

    return mu_water


def convert_to_hounsfield_units(
    mu_volume: np.ndarray,
    mu_water: float
) -> np.ndarray:
    """
    Convert linear attenuation coefficient volume to Hounsfield Units.

    HU = (μ - μ_water) / μ_water × 1000

    Standard HU values:
        - Air: -1000 HU
        - Water: 0 HU
        - Soft tissue: 20-80 HU
        - Bone: +1000 to +3000 HU

    Args:
        mu_volume: Reconstructed linear attenuation coefficient volume
        mu_water: Linear attenuation coefficient of water

    Returns:
        Volume in Hounsfield Units
    """
    hu_volume = ((mu_volume - mu_water) / mu_water) * 1000.0
    return hu_volume.astype(np.float32)


def validate_hu_calibration(
    volume: np.ndarray,
    expected_air_hu: float = -1000.0,
    expected_water_hu: float = 0.0,
    tolerance: float = 100.0
) -> Dict[str, float]:
    """
    Validate HU calibration by checking volume statistics.

    Args:
        volume: Reconstructed volume in HU
        expected_air_hu: Expected HU for air regions
        expected_water_hu: Expected HU for water regions
        tolerance: Acceptable deviation from expected values

    Returns:
        Dictionary with volume statistics
    """
    stats = {
        'min': float(np.min(volume)),
        'max': float(np.max(volume)),
        'mean': float(np.mean(volume)),
        'std': float(np.std(volume)),
        'percentile_1': float(np.percentile(volume, 1)),
        'percentile_99': float(np.percentile(volume, 99)),
    }

    # Check if air regions (low values) are close to -1000 HU
    if stats['percentile_1'] < expected_air_hu - tolerance:
        print(f"Warning: Air regions ({stats['percentile_1']:.0f} HU) below expected {expected_air_hu} HU")
    elif stats['percentile_1'] > expected_air_hu + tolerance:
        print(f"Warning: Air regions ({stats['percentile_1']:.0f} HU) above expected {expected_air_hu} HU")
    else:
        print(f"Air regions: {stats['percentile_1']:.0f} HU (expected ~{expected_air_hu:.0f} HU) - OK")

    return stats


if __name__ == '__main__':
    # Test with Scan_1681
    from .paths import RESULTS_DIR
    xml_path = str(RESULTS_DIR / 'Scan_1681_uwarp_gt' / 'scan.xml')

    print("Testing HU calibration module...")
    print("=" * 60)

    # Parse calibration
    calibration = parse_calibration_from_xml(str(xml_path))
    print(f"\nCalibration values from XML:")
    print(f"  Air Value:   {calibration['air_value']}")
    print(f"  Water Value: {calibration['water_value']}")
    print(f"  Bone HU:     {calibration['bone_hu']}")

    # Estimate mu_water
    mu_water = estimate_mu_water(
        calibration['air_value'],
        calibration['water_value'],
        calibration_path_length=100.0  # mm
    )
    print(f"\nEstimated mu_water: {mu_water:.6f} mm⁻¹")
    print(f"Literature mu_water at 80 keV: 0.00184 mm⁻¹")

    print("\n" + "=" * 60)
    print("Module test complete.")
