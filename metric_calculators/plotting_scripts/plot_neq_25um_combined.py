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
    target_directory = 'data/results/metric_plots/25um/'
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Fullscan-16ms
    image_data = vff.read_vff("data/scans/phantom/Fullscan-16ms/Fullscan-16ms-25um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[70:179, :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    image_data = vff.read_vff("data/scans/phantom/Fullscan-16ms/Fullscan-16ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[20:153, :, :]
    crop_indices_MTF = [776, 1984, 1084, 1592]
    label_1 = 'Fullscan-16ms'
    NEQ_1_freqs, NEQ_1 = NEQ_calculator.get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=0.025,
                                   target_directory=target_directory, plot_results=False, high_to_low_MTF=True)
    print(f'25% done')


    # Fullscan-100ms
    image_data = vff.read_vff("data/scans/phantom/Fullscan-100ms/Fullscan-100ms-25um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[np.concatenate((np.arange(0, 18), np.arange(130, 193))), :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    image_data = vff.read_vff("data/scans/phantom/Fullscan-100ms/Fullscan-100ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[10:150, :, :]
    crop_indices_MTF = [760, 1956, 1192, 1532]
    label_2 = 'Fullscan-100ms'
    NEQ_2_freqs, NEQ_2 = NEQ_calculator.get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=0.025,
                                    target_directory=target_directory, plot_results=False, high_to_low_MTF=True)
    print(f'50% done')


    # Halfscan-16ms
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-25um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[np.concatenate((np.arange(0, 15), np.arange(70, 188))), :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[80:208, :, :]
    crop_indices_MTF = [788, 1988, 1156, 1572]
    label_3 = 'Halfscan-16ms'
    NEQ_3_freqs, NEQ_3 = NEQ_calculator.get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=0.025,
                                    target_directory=target_directory, plot_results=False, high_to_low_MTF=True)
    print(f'75% done')


    # Halfscan-100ms
    image_data = vff.read_vff("data/scans/phantom/Halfscan-100ms/Halfscan-100ms-25um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[np.concatenate((np.arange(0, 30), np.arange(140, 182))), :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    image_data = vff.read_vff("data/scans/phantom/Halfscan-100ms/Halfscan-100ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[10:160, :, :]
    crop_indices_MTF = [732, 1940, 1160, 1548]
    label_4 = 'Halfscan-100ms'
    NEQ_4_freqs, NEQ_4 = NEQ_calculator.get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=0.025,
                                    target_directory=target_directory, plot_results=False, high_to_low_MTF=True)
    print(f'100% done')

    if NORMALIZE:
        NEQ_1 = NEQ_1 / NEQ_1.max()
        NEQ_2 = NEQ_2 / NEQ_2.max()
        NEQ_3 = NEQ_3 / NEQ_3.max()
        NEQ_4 = NEQ_4 / NEQ_4.max()


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
    plt.title('NEQ of different image data sets')
    if NORMALIZE:
        plt.savefig(target_directory + '/NEQ_results_25um_combined_normalized.png', dpi=300)
    else:
        plt.savefig(target_directory + '/NEQ_results_25um_combined.png', dpi=300)
