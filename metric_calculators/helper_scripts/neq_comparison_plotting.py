# Description: This script compares NEQ measurements from multiple VFF files
# and creates professional plots of NEQ curves for comparison.
# Written by Falk Wiegmann at the University of British Columbia in September 2024.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', '..'))
from ct_core import vff_io as vff
from metric_calculators import neq_calculator as NEQ_calculator

def create_neq_comparison_plot():
    """
    Creates a professional comparison plot of NEQ curves from three VFF files.
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
    
    # NEQ calculation parameters (adjust these based on your specific phantom)
    pixel_size = 0.085  # mm
    
    # MTF parameters
    mtf_slice_range = [228, 229]  # which slices to analyze for MTF
    crop_indices_MTF = [270, 664, 522, 640]  # [y1, y2, x1, x2]
    
    # NPS parameters
    nps_slice_range = np.concatenate((np.arange(204, 208), np.arange(216, 226)))  # homogeneous region slices
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
    neq_freq_results = []
    neq_results = []
    
    # Process each VFF file
    for i, vff_file in enumerate(vff_files):
        print(f"Processing {dataset_names[i]}...")
        
        # Load the VFF data
        header, image_data = vff.read_vff(vff_file, verbose=False)
        
        # Select the slice ranges for MTF and NPS
        image_data_mtf = image_data[mtf_slice_range[0]:mtf_slice_range[1], :, :]
        image_data_nps = image_data[nps_slice_range, :, :]
        
        # Calculate NEQ
        neq_freq, neq = NEQ_calculator.get_NEQ(
            image_data_mtf,
            image_data_nps,
            crop_indices_MTF,
            ROI_bounds_NPS,
            pixel_size=pixel_size,
            target_directory=os.getcwd(),
            plot_results=False,  # We'll create our own plots
            high_to_low_MTF=True
        )
        
        # Store results
        neq_freq_results.append(neq_freq)
        neq_results.append(neq)    # Create the comparison figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Set the overall style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot NEQ comparison
    for i, (neq_freq, neq, name, color) in enumerate(zip(neq_freq_results, neq_results, dataset_names, colors)):
        # Filter out any NaN or infinite values
        valid_mask = np.isfinite(neq) & (neq > 0)
        neq_freq_clean = neq_freq[valid_mask]
        neq_clean = neq[valid_mask]
        
        ax.plot(neq_freq_clean, neq_clean, color=color, linewidth=3, label=name, alpha=0.8)
        
        # Calculate and display key NEQ metrics
        if len(neq_clean) > 0:
            max_neq = np.max(neq_clean)
            max_neq_freq = neq_freq_clean[np.argmax(neq_clean)]
            
            # Find NEQ at specific frequencies of interest
            freq_of_interest = [1.0, 2.0, 3.0]  # lp/mm
            print(f"\n{name} NEQ Analysis:")
            print(f"  Peak NEQ: {max_neq:.2f} mm^-2 at {max_neq_freq:.3f} lp/mm")
            
            for freq in freq_of_interest:
                if freq <= np.max(neq_freq_clean):
                    neq_at_freq = np.interp(freq, neq_freq_clean, neq_clean)
                    print(f"  NEQ at {freq:.1f} lp/mm: {neq_at_freq:.3f} mm^-2")
    
    # Customize the plot
    ax.set_xlabel('Spatial Frequency (lp/mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('NEQ (mm$^{-2}$)', fontsize=14, fontweight='bold')
    ax.set_title('Noise Equivalent Quanta (NEQ)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Set reasonable axis limits
    ax.set_xlim([0, np.max([np.max(freq) for freq in neq_freq_results])])
    ax.set_ylim(bottom=0, top=np.max(neq_results)*1.2)
    
    # Add frequency markers for common imaging frequencies
    freq_markers = [1, 2, 3, 4, 5]  # lp/mm
    for freq in freq_markers:
        if freq <= np.max([np.max(freq_arr) for freq_arr in neq_freq_results]):
            ax.axvline(freq, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(freq, ax.get_ylim()[1] * 0.95, f'{freq} lp/mm', 
                   rotation=90, ha='right', va='top', fontsize=9, color='gray')
    
    # Add text box with summary statistics
    textstr = "NEQ Summary:\n"
    for i, (neq_freq, neq, name) in enumerate(zip(neq_freq_results, neq_results, dataset_names)):
        valid_mask = np.isfinite(neq) & (neq > 0)
        if np.any(valid_mask):
            neq_clean = neq[valid_mask]
            max_neq = np.max(neq_clean)
            textstr += f"{name}: {max_neq:.2f} mm$^{{-2}}$ (peak)\n"
    
    # Remove the last newline
    textstr = textstr.rstrip('\n')
    
    # Place text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Overall figure styling
    plt.tight_layout()
    
    # Save the figure
    output_filename = 'NEQ_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_filename[:-3]+'pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nNEQ comparison plot saved as: {output_filename}")
    
    # Calculate and print detailed summary statistics
    print("\n=== NEQ Comparison Summary ===")
    for i, (neq_freq, neq, name) in enumerate(zip(neq_freq_results, neq_results, dataset_names)):
        valid_mask = np.isfinite(neq) & (neq > 0)
        if np.any(valid_mask):
            neq_freq_clean = neq_freq[valid_mask]
            neq_clean = neq[valid_mask]
            
            max_neq = np.max(neq_clean)
            max_neq_freq = neq_freq_clean[np.argmax(neq_clean)]
            
            # Calculate area under NEQ curve (integrated NEQ)
            integrated_neq = np.trapz(neq_clean, neq_freq_clean)
            
            print(f"\n{name}:")
            print(f"  Peak NEQ: {max_neq:.3f} mm^-2")
            print(f"  Peak NEQ frequency: {max_neq_freq:.3f} lp/mm")
            print(f"  Integrated NEQ: {integrated_neq:.3f} mm^-1")
            
            # Calculate NEQ at specific frequencies
            for freq_interest in [1.0, 2.0, 3.0]:
                if freq_interest <= np.max(neq_freq_clean):
                    neq_at_freq = np.interp(freq_interest, neq_freq_clean, neq_clean)
                    print(f"  NEQ at {freq_interest:.1f} lp/mm: {neq_at_freq:.3f} mm^-2")
        else:
            print(f"\n{name}: No valid NEQ data")
    
    # Display the plot
    # plt.show()
    
    return fig

def calculate_neq_metrics(neq_freq, neq):
    """
    Calculate various NEQ metrics for analysis.
    
    Parameters:
    -----------
    neq_freq : array_like
        Frequency array
    neq : array_like
        NEQ values
        
    Returns:
    --------
    dict : Dictionary containing various NEQ metrics
    """
    
    # Filter out invalid values
    valid_mask = np.isfinite(neq) & (neq > 0)
    if not np.any(valid_mask):
        return {"valid_data": False}
    
    neq_freq_clean = neq_freq[valid_mask]
    neq_clean = neq[valid_mask]
    
    metrics = {
        "valid_data": True,
        "peak_neq": np.max(neq_clean),
        "peak_neq_freq": neq_freq_clean[np.argmax(neq_clean)],
        "integrated_neq": np.trapz(neq_clean, neq_freq_clean),
        "neq_at_1_lpmm": np.interp(1.0, neq_freq_clean, neq_clean) if 1.0 <= np.max(neq_freq_clean) else None,
        "neq_at_2_lpmm": np.interp(2.0, neq_freq_clean, neq_clean) if 2.0 <= np.max(neq_freq_clean) else None,
        "neq_at_3_lpmm": np.interp(3.0, neq_freq_clean, neq_clean) if 3.0 <= np.max(neq_freq_clean) else None,
    }
    
    return metrics

if __name__ == '__main__':
    _ = create_neq_comparison_plot()
