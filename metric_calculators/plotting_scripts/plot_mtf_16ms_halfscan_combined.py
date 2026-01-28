# Description: This script plots the MTF of different image data sets on the same plot.
# Written by Falk Wiegmann at the University of British Columbia in June 2024.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(2, os.path.join(sys.path[0], '../..'))
from metric_calculators import mtf_calculator as MTF_calculator
from ct_core import vff_io as vff

if __name__ == '__main__':
    target_directory = 'data/results/metric_plots/16ms_halfscan/'
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # this is all for Halfscan-16ms

    # 25um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[80:208, :, :]
    crop_indices_MTF = [788, 1988, 1156, 1572]
    MTF_freq_1, MTF_1 = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=0.025,
                               target_directory=target_directory, plot_results=False, edge_angle=5.5, high_to_low=True)
    label_1 = '25um'
    print(f'25% done')


    # 50um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-50um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[38:103, :, :]
    crop_indices_MTF = np.array([788, 1988, 1156, 1572])
    crop_indices_MTF = np.round(crop_indices_MTF/2).astype(int)
    MTF_freq_2, MTF_2 = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=0.050,
                               target_directory=target_directory, plot_results=False, edge_angle=5.5, high_to_low=True)
    label_2 = '50um'
    print(f'50% done')                


    # 75um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-75um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[24:67, :, :]
    crop_indices_MTF = np.array([788, 1988, 1156, 1572])
    crop_indices_MTF = np.round(crop_indices_MTF/3).astype(int)
    MTF_freq_3, MTF_3 = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=0.075,
                               target_directory=target_directory, plot_results=False, edge_angle=5.5, high_to_low=True)
    label_3 = '75um'
    print(f'75% done')


    # 100um
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-100um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[17:50, :, :]
    crop_indices_MTF = np.array([788, 1988, 1156, 1572])
    crop_indices_MTF = np.round(crop_indices_MTF/4).astype(int)
    MTF_freq_4, MTF_4 = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=0.100,
                               target_directory=target_directory, plot_results=False, edge_angle=5.5, high_to_low=True)
    label_4 = '100um'
    print(f'100% done')
    

    # Plotting
    plt.figure()
    plt.plot(MTF_freq_1, MTF_1, label=label_1)
    plt.plot(MTF_freq_2, MTF_2, label=label_2)
    plt.plot(MTF_freq_3, MTF_3, label=label_3)
    plt.plot(MTF_freq_4, MTF_4, label=label_4)
    plt.xlabel('Spatial Frequency (lp per mm)')
    plt.ylabel('Normalised Modulation (i.e. Contrast)')
    plt.legend()
    plt.grid(True)
    plt.title('MTF of different image data resolutions (Halfscan-16ms)')
    plt.xlim([0, np.max(MTF_freq_1)])
    plt.savefig(target_directory + '/MTF_results_16ms_Halfscan_combined.png', dpi=300)
