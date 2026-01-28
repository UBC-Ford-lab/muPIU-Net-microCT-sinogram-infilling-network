# Description: This script compares NPS measurements from multiple VFF files
# and creates professional plots of NPS curves for comparison.
# Written by Falk Wiegmann at the University of British Columbia in September 2024.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', '..'))
from ct_core import vff_io as vff
from metric_calculators import nps_calculator as NPS_calculator

def create_nps_comparison_plot():
    """
    Creates a professional comparison plot of NPS curves from three VFF files.
    Displays both linear and logarithmic NPS plots side by side.
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
    
    # NPS calculation parameters (adjust these based on your specific phantom)
    pixel_size = 0.085  # mm
    slice_range = np.concatenate((np.arange(204, 208), np.arange(216, 226)))  # homogeneous region slices
    
    # Define the ROIs over which to compute the NPS (y1, y2, x1, x2) in pixel coordinates
    # These should be in homogeneous regions of the phantom
    ROI_bounds_NPS = np.array([
        [178, 294, 510, 626], 
        [258, 374, 750, 866], 
        [310, 426, 328, 444], 
        [432, 548, 830, 946], 
        [488, 604, 248, 364], 
        [580, 696, 730, 846], 
        [624, 740, 414, 530], 
        [724, 840, 598, 714]
    ])
    
    # Storage for results
    nps_freq_results = []
    nps_results = []
    
    # Process each VFF file
    for i, vff_file in enumerate(vff_files):
        print(f"Processing {dataset_names[i]}...")
        
        # Load the VFF data
        header, image_data = vff.read_vff(vff_file, verbose=False)
        
        # Select the slice range containing homogeneous regions
        image_data_nps = image_data[slice_range, :, :]
        
        # Calculate NPS
        nps_freq, nps = NPS_calculator.get_NPS(
            image_data_nps, 
            ROI_bounds_NPS, 
            pixel_size=pixel_size,
            target_directory=os.getcwd(), 
            plot_results=False,  # We'll create our own plots
            filter_low_freq=False
        )
        
        # Store results
        nps_freq_results.append(nps_freq)
        nps_results.append(nps)
    
    # Create the comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set the overall style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot linear NPS comparison
    for i, (nps_freq, nps, name, color) in enumerate(zip(nps_freq_results, nps_results, dataset_names, colors)):
        ax1.plot(nps_freq, nps, color=color, linewidth=2.5, label=name, alpha=0.8)
        
        # Calculate and display the noise variance (integral of NPS)
        noise_variance = np.trapz(nps, nps_freq)
        print(f"{name} - Noise variance: {noise_variance:.1f} HU²")
    
    ax1.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('NPS (HU² mm²)', fontsize=12, fontweight='bold')
    ax1.set_title('Noise Power Spectrum (Linear Scale)', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=10)
    ax1.set_xlim([0, np.max(nps_freq_results[0])])
    ax1.set_ylim([0, None])
    
    # Plot logarithmic NPS comparison
    for i, (nps_freq, nps, name, color) in enumerate(zip(nps_freq_results, nps_results, dataset_names, colors)):
        # Avoid log(0) by setting minimum value
        nps_log_safe = np.maximum(nps, np.max(nps) * 1e-6)
        ax2.semilogy(nps_freq, nps_log_safe, color=color, linewidth=2.5, label=name, alpha=0.8)
    
    ax2.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('NPS (HU² mm²)', fontsize=12, fontweight='bold')
    ax2.set_title('Noise Power Spectrum (Log Scale)', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=10)
    ax2.set_xlim([0, np.max(nps_freq_results[0])])
    
    # Add frequency markers for common imaging frequencies
    freq_markers = [1, 2, 3, 4, 5]  # lp/mm
    for freq in freq_markers:
        if freq <= np.max(nps_freq_results[0]):
            ax1.axvline(freq, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax2.axvline(freq, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add text annotations for noise variance on the linear plot
    y_pos = 0.85
    for i, (nps_freq, nps, name, color) in enumerate(zip(nps_freq_results, nps_results, dataset_names, colors)):
        noise_variance = np.trapz(nps, nps_freq)
        ax1.text(0.02, y_pos - i*0.08, f'{name}: σ² = {noise_variance:.1f} HU²', 
                transform=ax1.transAxes, fontsize=10, color=color, fontweight='bold')
    
    # Overall figure styling
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save the figure
    output_filename = 'NPS_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_filename[:-3]+'pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"NPS comparison plot saved as: {output_filename}")
    
    # Calculate and print summary statistics
    print("\n=== NPS Comparison Summary ===")
    for i, (nps_freq, nps, name) in enumerate(zip(nps_freq_results, nps_results, dataset_names)):
        noise_variance = np.trapz(nps, nps_freq)
        peak_freq = nps_freq[np.argmax(nps)]
        peak_nps = np.max(nps)
        
        print(f"\n{name}:")
        print(f"  Noise variance (σ²): {noise_variance:.2f} HU²")
        print(f"  Peak NPS frequency: {peak_freq:.3f} lp/mm")
        print(f"  Peak NPS value: {peak_nps:.2f} HU² mm²")
        
        # Calculate NPS at specific frequencies
        for freq_interest in [1.0, 2.0, 3.0]:
            if freq_interest <= np.max(nps_freq):
                nps_at_freq = np.interp(freq_interest, nps_freq, nps)
                print(f"  NPS at {freq_interest:.1f} lp/mm: {nps_at_freq:.3f} HU² mm²")
    
    # Display the plot
    # plt.show()
    
    return fig


if __name__ == '__main__':
    _ = create_nps_comparison_plot()
