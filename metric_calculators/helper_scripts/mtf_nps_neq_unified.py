# Description: This script creates a unified comparison of MTF, NPS, and NEQ measurements 
# from multiple VFF files in a single professional figure.
# Written by Falk Wiegmann at the University of British Columbia in September 2024.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', '..'))
from ct_core import vff_io as vff
from metric_calculators import mtf_calculator as MTF_calculator
from metric_calculators import nps_calculator as NPS_calculator
from metric_calculators import neq_calculator as NEQ_calculator

def create_unified_comparison_plot():
    """
    Creates a unified professional comparison plot showing MTF, NPS, and NEQ curves 
    from three VFF files in a single figure with three subplots.
    """
    
    # Define the VFF files to compare
    vff_files = [
        "data/results/ground_truth_reconstruction.vff",
        #"data/results/ground_truth_reconstruction.vff",
        "base_models/models/deepfill/results/reconstructed_volume.vff", 
        "data/results/unet_reconstruction.vff"
    ]
    
    # Define descriptive names for the datasets
    dataset_names = [
        "Ground Truth",
        "DeepFill v2 model", 
        "Task-specific U-Net"
    ]
    
    # Define colors for professional appearance
    colors = ['#2E86C1', '#E74C3C', '#28B463']  # Blue, Red, Green
    
    # Common parameters
    pixel_size = 0.085  # mm
    
    # MTF parameters
    mtf_slice_range = [228, 229]  # which slices to analyze for MTF
    crop_indices_MTF = [270, 664, 522, 640]  # [y1, y2, x1, x2]
    edge_angle = 5.4    # degrees
    
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
    mtf_freq_results = []
    mtf_results = []
    mtf_max_values = []
    nps_freq_results = []
    nps_results = []
    neq_freq_results = []
    neq_results = []

    # Process each VFF file
    for i, vff_file in enumerate(vff_files):
        print(f"Processing {dataset_names[i]}...")

        # Load the VFF data
        header, image_data = vff.read_vff(vff_file, verbose=False)

        # Select the slice ranges for different measurements
        image_data_mtf = image_data[mtf_slice_range[0]:mtf_slice_range[1], :, :]
        image_data_nps = image_data[nps_slice_range, :, :]

        # Calculate MTF (get unnormalized MTF)
        print(f"  Calculating MTF...")
        mtf_freq, mtf = MTF_calculator.get_MTF(
            image_data_mtf,
            crop_indices_MTF,
            find_absolute_MTF=True,
            pixel_size=pixel_size,
            target_directory=os.getcwd(),
            plot_results=False,
            edge_angle=edge_angle,
            high_to_low=True,
            process_LSF=True,
            normalize_MTF=False  # Don't normalize yet - we'll do it globally
        )
        
        # Calculate NPS
        print(f"  Calculating NPS...")
        nps_freq, nps = NPS_calculator.get_NPS(
            image_data_nps, 
            ROI_bounds_NPS, 
            pixel_size=pixel_size,
            target_directory=os.getcwd(), 
            plot_results=False,
            filter_low_freq=False
        )
        
        # Calculate NEQ
        print(f"  Calculating NEQ...")
        neq_freq, neq = NEQ_calculator.get_NEQ(
            image_data_mtf,
            image_data_nps,
            crop_indices_MTF,
            ROI_bounds_NPS,
            pixel_size=pixel_size,
            target_directory=os.getcwd(),
            plot_results=False,
            high_to_low_MTF=True
        )
        
        # Store results
        mtf_freq_results.append(mtf_freq)
        mtf_results.append(mtf)
        mtf_max_values.append(mtf.max())
        nps_freq_results.append(nps_freq)
        nps_results.append(nps)
        neq_freq_results.append(neq_freq)
        neq_results.append(neq)

    # Find the global maximum MTF value (at zero frequency)
    global_mtf_max = max(mtf_max_values)

    # Normalize all MTFs by the global maximum
    mtf_results = [mtf / global_mtf_max for mtf in mtf_results]

    # Create the unified comparison figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Set the overall style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ================================= MTF PLOT =================================
    for i, (mtf_freq, mtf, name, color) in enumerate(zip(mtf_freq_results, mtf_results, dataset_names, colors)):
        # Only plot positive frequencies
        positive_freq_mask = mtf_freq >= 0
        mtf_freq_pos = mtf_freq[positive_freq_mask]
        mtf_pos = mtf[positive_freq_mask]
        
        # Interpolate for more accurate MTF50 values
        mtf_freq_interp = np.linspace(0, np.max(mtf_freq_pos), 1000)
        mtf_interp = np.interp(mtf_freq_interp, mtf_freq_pos, mtf_pos)
        
        # Find MTF50
        mtf50_freq = None
        try:
            mtf50_idx = np.where((mtf_interp < 0.5) & (mtf_freq_interp > 0))[0][0]
            mtf50_freq = mtf_freq_interp[mtf50_idx]
            ax1.axvline(mtf50_freq, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
        except:
            mtf50_freq = 0.0
        
        # Plot MTF curve with MTF50 in legend
        ax1.plot(mtf_freq_pos, mtf_pos, 
                color=color, linewidth=2.5, 
                label=f'{name} (MTF50={mtf50_freq:.2f})', alpha=0.8)
    
    # Add reference lines for MTF
    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax1.axhline(0.1, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax1.text(ax1.get_xlim()[1]*0.02, 0.52, 'MTF50', fontsize=10, color='gray')
    ax1.text(ax1.get_xlim()[1]*0.02, 0.12, 'MTF10', fontsize=10, color='gray')
    
    ax1.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Modulation Transfer Function', fontsize=12, fontweight='bold')
    ax1.set_title('MTF', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1/(2*pixel_size)])
    ax1.set_ylim([0, 1.05])
    ax1.tick_params(labelsize=10)
    
    # ================================= NPS PLOT =================================
    for i, (nps_freq, nps, name, color) in enumerate(zip(nps_freq_results, nps_results, dataset_names, colors)):
        # Avoid log(0) by setting minimum value
        nps_log_safe = np.maximum(nps, np.max(nps) * 1e-6)
        noise_variance = np.trapz(nps, nps_freq)
        
        ax2.semilogy(nps_freq, nps_log_safe, color=color, linewidth=2.5, 
                    label=f'{name} (σ²={noise_variance:.1f})', alpha=0.8)
    
    ax2.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('NPS (HU² mm²)', fontsize=12, fontweight='bold')
    ax2.set_title('NPS (Log Scale)', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=10)
    ax2.set_xlim([0, np.max(nps_freq_results[0])])
    
    # ================================= NEQ PLOT =================================
    for i, (neq_freq, neq, name, color) in enumerate(zip(neq_freq_results, neq_results, dataset_names, colors)):
        # Filter out any NaN or infinite values
        valid_mask = np.isfinite(neq) & (neq > 0)
        neq_freq_clean = neq_freq[valid_mask]
        neq_clean = neq[valid_mask]
        
        if len(neq_clean) > 0:
            max_neq = np.max(neq_clean)
            ax3.plot(neq_freq_clean, neq_clean, color=color, linewidth=2.5, 
                    label=f'{name} (Peak={max_neq:.2f})', alpha=0.8)
    
    ax3.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('NEQ (mm$^{-2}$)', fontsize=12, fontweight='bold')
    ax3.set_title('NEQ', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=10)
    ax3.set_xlim([0, np.max([np.max(freq) for freq in neq_freq_results])])
    ax3.set_ylim(bottom=0)
    
    # Add frequency markers to all plots
    freq_markers = [1, 2, 3, 4, 5]  # lp/mm
    for freq in freq_markers:
        # MTF plot
        if freq <= 1/(2*pixel_size):
            ax1.axvline(freq, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        
        # NPS plot
        if freq <= np.max(nps_freq_results[0]):
            ax2.axvline(freq, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        
        # NEQ plot
        if freq <= np.max([np.max(freq_arr) for freq_arr in neq_freq_results]):
            ax3.axvline(freq, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Overall figure styling
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save the figure
    output_filename = 'MTF_NPS_NEQ_unified_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nUnified comparison plot saved as: {output_filename}")
    
    # Print comprehensive summary statistics
    print("\n" + "="*70)
    print("COMPREHENSIVE IMAGE QUALITY ASSESSMENT SUMMARY")
    print("="*70)
    
    for i, name in enumerate(dataset_names):
        print(f"\n{name.upper()}:")
        print("-" * (len(name) + 1))
        
        # MTF metrics
        mtf_freq = mtf_freq_results[i]
        mtf = mtf_results[i]
        positive_freq_mask = mtf_freq >= 0
        mtf_freq_pos = mtf_freq[positive_freq_mask]
        mtf_pos = mtf[positive_freq_mask]
        
        mtf_freq_interp = np.linspace(0, np.max(mtf_freq_pos), 1000)
        mtf_interp = np.interp(mtf_freq_interp, mtf_freq_pos, mtf_pos)
        
        try:
            mtf50_idx = np.where((mtf_interp < 0.5) & (mtf_freq_interp > 0))[0][0]
            mtf50_freq = mtf_freq_interp[mtf50_idx]
        except:
            mtf50_freq = 0.0
            
        try:
            mtf10_idx = np.where((mtf_interp < 0.1) & (mtf_freq_interp > 0))[0][0]
            mtf10_freq = mtf_freq_interp[mtf10_idx]
        except:
            mtf10_freq = 0.0
        
        print(f"  MTF Metrics:")
        print(f"    MTF50: {mtf50_freq:.3f} lp/mm")
        print(f"    MTF10: {mtf10_freq:.3f} lp/mm")
        
        # NPS metrics
        nps_freq = nps_freq_results[i]
        nps = nps_results[i]
        noise_variance = np.trapz(nps, nps_freq)
        peak_nps = np.max(nps)
        peak_nps_freq = nps_freq[np.argmax(nps)]
        
        print(f"  NPS Metrics:")
        print(f"    Noise variance (σ²): {noise_variance:.2f} HU²")
        print(f"    Peak NPS: {peak_nps:.3f} HU² mm² at {peak_nps_freq:.3f} lp/mm")
        
        # NEQ metrics
        neq_freq = neq_freq_results[i]
        neq = neq_results[i]
        valid_mask = np.isfinite(neq) & (neq > 0)
        
        if np.any(valid_mask):
            neq_freq_clean = neq_freq[valid_mask]
            neq_clean = neq[valid_mask]
            max_neq = np.max(neq_clean)
            max_neq_freq = neq_freq_clean[np.argmax(neq_clean)]
            integrated_neq = np.trapz(neq_clean, neq_freq_clean)
            
            print(f"  NEQ Metrics:")
            print(f"    Peak NEQ: {max_neq:.3f} mm^-2 at {max_neq_freq:.3f} lp/mm")
            print(f"    Integrated NEQ: {integrated_neq:.3f} mm^-1")
            
            # NEQ at specific frequencies
            for freq_interest in [1.0, 2.0, 3.0]:
                if freq_interest <= np.max(neq_freq_clean):
                    neq_at_freq = np.interp(freq_interest, neq_freq_clean, neq_clean)
                    print(f"    NEQ at {freq_interest:.1f} lp/mm: {neq_at_freq:.3f} mm^-2")
        else:
            print(f"  NEQ Metrics: No valid data")
    
    print("\n" + "="*70)
    
    # Display the plot
    # plt.show()
    
    return fig

def calculate_image_quality_score(mtf50, noise_variance, peak_neq):
    """
    Calculate a composite image quality score based on MTF, NPS, and NEQ metrics.
    Higher scores indicate better image quality.
    
    Parameters:
    -----------
    mtf50 : float
        MTF50 value in lp/mm
    noise_variance : float
        Noise variance in HU²
    peak_neq : float
        Peak NEQ value in mm^-2
        
    Returns:
    --------
    float : Composite image quality score
    """
    
    # Normalize metrics (higher MTF50 and NEQ are better, lower noise variance is better)
    # These normalization factors should be adjusted based on your specific application
    mtf_score = mtf50 / 10.0  # Assuming good MTF50 is around 5-10 lp/mm
    noise_score = 1000.0 / max(noise_variance, 1.0)  # Invert noise (lower is better)
    neq_score = peak_neq / 100.0  # Assuming good peak NEQ is around 50-100 mm^-2
    
    # Weighted composite score
    composite_score = 0.4 * mtf_score + 0.3 * noise_score + 0.3 * neq_score
    
    return composite_score

if __name__ == '__main__':
    _ = create_unified_comparison_plot()
