import torch
import numpy as np
import xmltodict
import sys
import time
import os
from pathlib import Path

from reconstruction.fdk import FDKReconstructor
from ct_core import tiff_converter
from ct_core.vff_io import VFFDataset


def main():
    start = time.time()
    data_folder = 'data/results/Scan_1681_uwarp_no_pred'
    xml_file = os.path.join(data_folder, 'scan.xml')

    # Load all projections and angles
    dataset = VFFDataset(data_folder, xml_file, paths_str="uwarp*", projection_spacing=2*0.878049)
    projections = dataset.projections  # shape (N_angles, N_b, N_a)
    angles = dataset.angles_rad       # shape (N_angles,)

    # Define geometry
    header = xmltodict.parse(open(xml_file).read())
    sp = header['Series']['SeriesParams']
    source_to_isocenter = float(header['Series']['ObjectPosition'])
    detector_to_isocenter = float(header['Series']['DetectorPosition']) - source_to_isocenter
    geometry = {
        'R_d': detector_to_isocenter,
        'R_s': source_to_isocenter,
        'da': float(header['Series']['DetectorSpacing']),
        'db': float(header['Series']['DetectorSpacing']),
        'vol_shape': (1100, 1100, 300),
        #'vol_shape': (300, 300, 200),
        'vol_origin': (0, 0, 0),
        'dx': 0.085,
        #'dx': 0.085*4,
        'dz': 0.4,
        #'central_pixel_a': float(header['Series']['SeriesParams']['ScanWidth']['#text'])
        #                    - float(header['Series']['CentreOfRotation']),
        'central_pixel_a': float(header['Series']['CentreOfRotation']),
        'central_pixel_b': float(header['Series']['CentralSlice'])
    }

    # Initialize reconstructor with full dataset
    reconstructor = FDKReconstructor(
        projections=projections,
        angles=angles,
        geometry=geometry,
        source_locations=None,
        folder_name='data/results/Scan_1681_no_pred_recon'
    )

    # Run full reconstruction
    reconstructor.reconstruct(display_volume=False)

    end = time.time()
    print(f"Reconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    main()
