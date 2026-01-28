# Description: This script plots the NPS of different image data sets on the same plot.
# Written by Falk Wiegmann at the University of British Columbia in June 2024.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(2, os.path.join(sys.path[0], '../..'))
from metric_calculators import nps_calculator as NPS_calculator
from ct_core import vff_io as vff

if __name__ == '__main__':
    target_directory = 'data/results/metric_plots/25um/'
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Fullscan-16ms
    image_data = vff.read_vff("data/scans/phantom/Fullscan-16ms/Fullscan-16ms-25um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[70:179, :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    label_1 = 'Fullscan-16ms'
    NPS_1 = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=0.025, target_directory=target_directory, plot_results=False)
    print(f'25% done')


    # Fullscan-100ms
    image_data = vff.read_vff("data/scans/phantom/Fullscan-100ms/Fullscan-100ms-25um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[np.concatenate((np.arange(0, 18), np.arange(130, 193))), :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    label_2 = 'Fullscan-100ms'
    NPS_2 = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=0.025, target_directory=target_directory, plot_results=False)
    print(f'50% done')


    # Halfscan-16ms
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-25um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[np.concatenate((np.arange(0, 15), np.arange(70, 188))), :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    label_3 = 'Halfscan-16ms'
    NPS_3 = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=0.025, target_directory=target_directory, plot_results=False)
    print(f'75% done')


    # Halfscan-100ms
    image_data = vff.read_vff("data/scans/phantom/Halfscan-100ms/Halfscan-100ms-25um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[np.concatenate((np.arange(0, 30), np.arange(140, 182))), :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    label_4 = 'Halfscan-100ms'
    NPS_4 = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=0.025, target_directory=target_directory, plot_results=False)
    print(f'100% done')
    

    # Plotting
    plt.figure()
    plt.plot(NPS_1[0], NPS_1[1], label=label_1)
    plt.plot(NPS_2[0], NPS_2[1], label=label_2)
    plt.plot(NPS_3[0], NPS_3[1], label=label_3)
    plt.plot(NPS_4[0], NPS_4[1], label=label_4)
    plt.xlabel('Radial Average Frequency (mm$^{-1}$)')
    plt.ylabel('NPS (HU$^2$ mm$^2$)')
    plt.legend()
    plt.grid(True)
    plt.title('Radially averaged NPS of different image data sets')
    plt.savefig(target_directory + '/NPS_results_15um_combined.png', dpi=300)
