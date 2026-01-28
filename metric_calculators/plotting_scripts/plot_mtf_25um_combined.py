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
    target_directory = 'data/results/metric_plots/25um/'
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # this is all for all 25um

    # Halfscan-16ms
    image_data = vff.read_vff("data/scans/phantom/Halfscan-16ms/Halfscan-16ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[80:208, :, :]
    crop_indices_MTF = [788, 1988, 1156, 1572]
    MTF_freq_1, MTF_1 = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=0.025,
                               target_directory=target_directory, plot_results=False, edge_angle=5.5, high_to_low=True)
    label_1 = 'Halfscan-16ms'
    print(f'25% done')


    # Halfscan-100ms
    image_data = vff.read_vff("data/scans/phantom/Halfscan-100ms/Halfscan-100ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[10:160, :, :]
    crop_indices_MTF = [732, 1940, 1160, 1548]
    MTF_freq_2, MTF_2 = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=0.025,
                               target_directory=target_directory, plot_results=False, edge_angle=5.5, high_to_low=True)
    label_2 = 'Halfscan-100ms'
    print(f'50% done')                


    # Fullscan-100ms
    image_data = vff.read_vff("data/scans/phantom/Fullscan-100ms/Fullscan-100ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[10:150, :, :]
    crop_indices_MTF = [760, 1956, 1192, 1532]
    MTF_freq_3, MTF_3 = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=0.025,
                               target_directory=target_directory, plot_results=False, edge_angle=5.5, high_to_low=True)
    label_3 = 'Fullscan-100ms'
    print(f'75% done')


    # Fullscan-16ms
    image_data = vff.read_vff("data/scans/phantom/Fullscan-16ms/Fullscan-16ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[20:153, :, :]
    crop_indices_MTF = [776, 1984, 1084, 1592]
    MTF_freq_4, MTF_4 = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=0.025,
                            target_directory=target_directory, plot_results=False, edge_angle=5.5, high_to_low=True)
    label_4 = 'Fullscan-16ms'
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
    plt.title('MTF of different image data sets')
    plt.xlim([0, np.max(MTF_freq_1)])
    plt.savefig(target_directory + '/MTF_results_25um_combined.png', dpi=300)
