# Description: This script plots the NEQ of different image data sets on the same plot.
# Written by Falk Wiegmann at the University of British Columbia in June 2024.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(2, os.path.join(sys.path[0], '../..'))
from metric_calculators import neq_calculator as NEQ_calculator
from ct_core import vff_io as vff

if __name__ == '__main__':
    NORMALIZE = True

    target_directory = 'data/results/metric_plots/16ms_halfscan/'
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # these are all for Halfscan 16ms

    # 25um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-25um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[np.concatenate((np.arange(0, 15), np.arange(70, 188))), :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[80:208, :, :]
    crop_indices_MTF = [788, 1988, 1156, 1572]
    label_1 = '25um'
    NEQ_1_freqs, NEQ_1 = NEQ_calculator.get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=0.025,
                                    target_directory=target_directory, plot_results=False, high_to_low_MTF=True)
    print(f'25% done')


    # 50um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-50um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[32:56, :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    ROI_bounds_NPS = np.round(ROI_bounds_NPS/2).astype(int)
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-50um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[38:103, :, :]
    crop_indices_MTF = np.array([788, 1988, 1156, 1572])
    crop_indices_MTF = np.round(crop_indices_MTF/2).astype(int)
    label_2 = '50um'
    NEQ_2_freqs, NEQ_2 = NEQ_calculator.get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=0.050,
                                    target_directory=target_directory, plot_results=False, high_to_low_MTF=True)
    print(f'50% done')


    # 75um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-75um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[26:54, :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1429, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1981, 512, 918], [1040, 1446, 304, 709], [544, 949, 544, 949]])
    ROI_bounds_NPS = np.round(ROI_bounds_NPS/3).astype(int)
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-75um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[24:67, :, :]
    crop_indices_MTF = np.array([788, 1988, 1156, 1572])
    crop_indices_MTF = np.round(crop_indices_MTF/3).astype(int)
    label_3 = '75um'
    NEQ_3_freqs, NEQ_3 = NEQ_calculator.get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=0.075,
                                    target_directory=target_directory, plot_results=False, high_to_low_MTF=True)
    print(f'75% done')


    # 100um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-100um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[19:40, :, :]
    ROI_bounds_NPS = np.array([[308, 716, 1116, 1524],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2300, 1148, 1556], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    ROI_bounds_NPS = np.round(ROI_bounds_NPS/4).astype(int)
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-100um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[17:50, :, :]
    crop_indices_MTF = np.array([788, 1988, 1156, 1572])
    crop_indices_MTF = np.round(crop_indices_MTF/4).astype(int)
    label_4 = '100um'
    NEQ_4_freqs, NEQ_4 = NEQ_calculator.get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=0.1,
                                    target_directory=target_directory, plot_results=False, high_to_low_MTF=True)
    print(f'100% done')

    if NORMALIZE:
        # normalize NEQ values
        NEQ_1 = NEQ_1 / np.max(NEQ_1)
        NEQ_2 = NEQ_2 / np.max(NEQ_2)
        NEQ_3 = NEQ_3 / np.max(NEQ_3)
        NEQ_4 = NEQ_4 / np.max(NEQ_4)

    # Plotting
    plt.figure()
    plt.plot(NEQ_1_freqs, NEQ_1, label=label_1)
    plt.plot(NEQ_2_freqs, NEQ_2, label=label_2)
    plt.plot(NEQ_3_freqs, NEQ_3, label=label_3)
    plt.plot(NEQ_4_freqs, NEQ_4, label=label_4)
    plt.xlabel('Spatial Frequency (mm$^{-1}$)')
    plt.ylabel('NEQ (mm$^{-2}$)')
    plt.legend()
    plt.grid(True)
    plt.title('NEQ of different image resolutions (Halfscan 16ms)')
    if NORMALIZE:
        plt.savefig(target_directory + '/NEQ_results_16ms_Halfscan_combined_normalised.png', dpi=300)
    else:
        plt.savefig(target_directory + '/NEQ_results_16ms_Halfscan_combined.png', dpi=300)
