# Description: This script compares MTF measurements from multiple VFF files
# and creates professional plots of ERF and MTF curves for comparison.
# Written by Falk Wiegmann at the University of British Columbia in September 2024.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', '..'))
from ct_core import vff_io as vff
from metric_calculators import mtf_calculator as MTF_calculator

def create_mtf_comparison_plot():
    """
    Creates a professional comparison plot of ERF and MTF curves from three VFF files.
    """
    
    # Define the VFF files to compare
    vff_files = [
        "data/results/ground_truth_reconstruction.vff",
        #"data/results/ground_truth_reconstruction.vff",
        "base_models/models/repaint/results/reconstructed_volume.vff", 
        "data/results/unet_reconstruction.vff"
    ]
    
    # Define descriptive names for the datasets
    dataset_names = [
        "Ground Truth",
        "RePaint model", 
        "Task-specific U-Net"
    ]
    
    # Define colors for professional appearance
    colors = ['#2E86C1', '#E74C3C', '#28B463']  # Blue, Red, Green
    
    # MTF calculation parameters (adjust these based on your specific phantom)
    crop_indices_MTF = [270, 664, 522, 640]  # [y1, y2, x1, x2]
    pixel_size = 0.085  # mm
    edge_angle = 5.4    # degrees
    slice_range = [228, 229]  # which slices to analyze
    
    # Storage for results
    erf_results = []
    mtf_freq_results = []
    mtf_results = []
    mtf_max_values = []

    # Process each VFF file
    for i, vff_file in enumerate(vff_files):
        print(f"Processing {dataset_names[i]}...")

        # Load the VFF data
        header, image_data = vff.read_vff(vff_file, verbose=False)

        # Select the slice range containing the edge pattern
        image_data_mtf = image_data[slice_range[0]:slice_range[1], :, :]

        # Calculate MTF and ERF (get unnormalized MTF)
        mtf_freq, mtf, erf = MTF_calculator.get_MTF(
            image_data_mtf,
            crop_indices_MTF,
            find_absolute_MTF=True,
            pixel_size=pixel_size,
            target_directory=os.getcwd(),
            plot_results=False,  # We'll create our own plots
            edge_angle=edge_angle,
            high_to_low=True,
            process_LSF=True,
            return_ERF=True,
            normalize_MTF=False  # Don't normalize yet - we'll do it globally
        )

        # Store results
        erf_results.append(erf)
        mtf_freq_results.append(mtf_freq)
        mtf_results.append(mtf)
        mtf_max_values.append(mtf.max())

    # Find the global maximum MTF value (at zero frequency)
    global_mtf_max = max(mtf_max_values)

    # Normalize all MTFs by the global maximum
    mtf_results = [mtf / global_mtf_max for mtf in mtf_results]
    
    # Create the comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set the overall style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot ERF comparison
    for i, (erf, name, color) in enumerate(zip(erf_results, dataset_names, colors)):
        x_axis_erf = np.linspace(0, len(erf) * pixel_size, len(erf))
        ax1.plot(x_axis_erf, erf, color=color, linewidth=2.5, label=name, alpha=0.8)
    
    ax1.set_xlabel('Distance perpendicular to edge (mm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
    ax1.set_title('Edge Response Function (ERF)', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=10)
    
    # Plot MTF comparison
    for i, (mtf_freq, mtf, name, color) in enumerate(zip(mtf_freq_results, mtf_results, dataset_names, colors)):
        # Only plot positive frequencies
        positive_freq_mask = mtf_freq >= 0
        
        # Find and mark MTF50 and MTF10
        mtf_freq_pos = mtf_freq[positive_freq_mask]
        mtf_pos = mtf[positive_freq_mask]
        
        # Interpolate for more accurate MTF50/MTF10 values
        mtf_freq_interp = np.linspace(0, np.max(mtf_freq_pos), 1000)
        mtf_interp = np.interp(mtf_freq_interp, mtf_freq_pos, mtf_pos)
        
        # Find MTF50
        try:
            mtf50_idx = np.where((mtf_interp < 0.5) & (mtf_freq_interp > 0))[0][0]
            mtf50_freq = mtf_freq_interp[mtf50_idx]
            ax2.axvline(mtf50_freq, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
            #ax2.text(mtf50_freq, 0.52, f'{mtf50_freq:.2f}', rotation=90, 
            #        color=color, fontsize=9, ha='right', va='bottom')
        except Exception as e:
            print('Error finding MTF50:', e)
            pass

        ax2.plot(mtf_freq[positive_freq_mask], mtf[positive_freq_mask], 
                color=color, linewidth=2.5, label=name + f' MTF50={mtf50_freq:.2f}', alpha=0.8)

    # Add reference lines
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax2.axhline(0.1, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax2.text(ax2.get_xlim()[1]*0.02, 0.52, 'MTF50', fontsize=10, color='gray')
    ax2.text(ax2.get_xlim()[1]*0.02, 0.12, 'MTF10', fontsize=10, color='gray')
    
    ax2.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Modulation Transfer Function', fontsize=12, fontweight='bold')
    ax2.set_title('MTF', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1/(2*pixel_size)])
    ax2.set_ylim([0, 1.05])
    ax2.tick_params(labelsize=10)
    
    # Overall figure styling
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save the figure
    output_filename = 'MTF_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_filename[:-3]+'pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Comparison plot saved as: {output_filename}")
    
    # Display the plot
    #plt.show()
    
    return fig

if __name__ == '__main__':
    _ = create_mtf_comparison_plot()



''' For original pred vs gt vs no pred comparison:
# Define the VFF files to compare
    vff_files = [
        "data/results/ground_truth_reconstruction.vff",
        "data/results/undersampled_reconstruction.vff", 
        "data/results/unet_reconstruction.vff"
    ]
    
    # Define descriptive names for the datasets
    dataset_names = [
        "Ground Truth",
        "No Prediction", 
        "With Prediction"
    ]
'''