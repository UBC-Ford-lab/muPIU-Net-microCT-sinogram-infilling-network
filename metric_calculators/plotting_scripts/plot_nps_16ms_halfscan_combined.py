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
    target_directory = 'data/results/metric_plots/16ms_halfscan/'
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # this is all for Halfscan-16ms

    # 25um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-25um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[np.concatenate((np.arange(0, 15), np.arange(70, 188))), :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    label_1 = '25um'
    NPS_1 = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=0.025, target_directory=target_directory, plot_results=False)
    print(f'25% done')


    # 50um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-50um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[32:56, :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    ROI_bounds_NPS = np.round(ROI_bounds_NPS/2).astype(int)
    label_2 = '50um'
    NPS_2 = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=0.050, target_directory=target_directory, plot_results=False)
    print(f'50% done')


    # 75um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-75um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[26:54, :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1429, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1981, 512, 918], [1040, 1446, 304, 709], [544, 949, 544, 949]])
    ROI_bounds_NPS = np.round(ROI_bounds_NPS/3).astype(int)
    label_3 = '75um'
    NPS_3 = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=0.075, target_directory=target_directory, plot_results=False)
    print(f'75% done')


    # 100um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-100um-wireHU.vff", verbose=False)[1]
    image_data_NPS = image_data[19:40, :, :]
    ROI_bounds_NPS = np.array([[308, 716, 1116, 1524],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2300, 1148, 1556], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])
    ROI_bounds_NPS = np.round(ROI_bounds_NPS/4).astype(int)
    label_4 = '100um'
    NPS_4 = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=0.100, target_directory=target_directory, plot_results=False)
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
    plt.title('Radially averaged NPS of different image data resolutions (Halfscan-16ms)')
    plt.savefig(target_directory + '/NPS_results_16ms_Halfscan_combined.png', dpi=300)
