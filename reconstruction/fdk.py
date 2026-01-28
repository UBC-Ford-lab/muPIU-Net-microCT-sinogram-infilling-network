# Script created by Falk Wiegmann in Feb 2025 to simulate a 3D cone beam CT scan reconstruction
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from ct_core.calibration import flat_field_correction, log_transform_transmission

# Device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FDKReconstructor:
    def __init__(self, projections, angles, geometry, source_locations, folder_name,
                 air_value=None, mu_water=None, output_hu=False,
                 bright_field=None, dark_field=None):
        """
        projections: Tensor of shape (N_angles, N_b, N_a) in float32.
        angles: Tensor of shape (N_angles,) in radians.
        geometry: dictionary with keys:
           - D: source-to-detector distance (in mm)
           - R: source-to-isocenter distance (in mm)
           - da: detector pixel size in horizontal direction (mm)
           - db: detector pixel size in vertical direction (mm)
           - vol_shape: tuple (Nx, Ny, Nz) for the reconstruction volume (number of voxels)
           - dx: voxel size (assumed isotropic, in mm)
        source_locations: list of source locations in the form [(x1, y1, z1), (x2, y2, z2), ...]
        air_value: float, calibration value for unattenuated beam (I₀) for log transform (deprecated, use bright_field)
        mu_water: float, linear attenuation coefficient of water in mm⁻¹ for HU conversion
        output_hu: bool, if True, convert output to Hounsfield Units
        bright_field: np.ndarray, unattenuated beam reference (I₀) for flat-field correction [height, width]
        dark_field: np.ndarray, electronic noise reference for flat-field correction [height, width]
        """
        self.projections = projections # (N_angles, N_b, N_a)
        self.angles = angles.to(device)
        self.R_s = geometry["R_s"]
        self.R_d = geometry["R_d"]
        self.SDD = self.R_s + self.R_d
        self.da = geometry["da"]
        self.db = geometry["db"]
        self.vol_shape = geometry["vol_shape"] # (Nx, Ny, Nz)
        self.vol_origin = geometry["vol_origin"] # (x, y, z) in mm
        self.dx = geometry["dx"] # voxel size in mm
        self.dz = geometry["dz"] # voxel size in mm
        self.central_pixel_a = geometry["central_pixel_a"]
        self.central_pixel_b = geometry["central_pixel_b"]
        self.source_locations = source_locations
        self.folder_name = folder_name

        # HU calibration parameters
        self.air_value = air_value
        self.mu_water = mu_water
        self.output_hu = output_hu
        self.bright_field = bright_field
        self.dark_field = dark_field

        # Determine detector dimensions and center indices
        self.N_angles, self.N_b, self.N_a = self.projections.shape
        self.a_center = (self.N_a - 1) / 2.0
        self.a_length = self.da * self.N_a
        self.b_center = (self.N_b - 1) / 2.0
        self.b_length = self.db * self.N_b

    def log_transform(self):
        """
        Apply log transformation to convert raw intensities to line integrals.

        p = -log(I / I₀) where I₀ is the unattenuated beam intensity

        This is required for proper CT reconstruction - the FDK algorithm
        expects line integrals of attenuation, not raw transmitted intensity.
        Without this, the output will NOT be in proper attenuation units.

        If air_value is very small compared to projection data (indicating it's
        not the actual I₀), we estimate I₀ from the projection data itself using
        the 99.9th percentile (representing unattenuated beam through air).
        """
        epsilon = 1e-6  # Prevent log(0)
        chunk_size = 20

        # First pass: estimate I₀ from projection data if air_value seems wrong
        # Sample a few projections to check the scale
        sample_indices = [0, self.N_angles // 2, self.N_angles - 1]
        sample_values = []
        for idx in sample_indices:
            chunk = self.projections[idx].astype(np.float32)
            sample_values.append(chunk.max())

        data_max = max(sample_values)

        # If air_value is much smaller than data max, it's not the actual I₀
        if self.air_value is not None and self.air_value < data_max * 0.01:
            print(f"Warning: air_value ({self.air_value}) is much smaller than projection max ({data_max:.0f})")
            print("Estimating I₀ from projection data instead...")

            # Compute 99.9th percentile across sampled projections as I₀
            all_samples = []
            for idx in range(0, self.N_angles, max(1, self.N_angles // 20)):
                chunk = self.projections[idx].astype(np.float32)
                all_samples.append(chunk.flatten())
            all_samples = np.concatenate(all_samples)
            I0_estimate = np.percentile(all_samples, 99.9)
            print(f"Estimated I₀ from data: {I0_estimate:.1f}")
            effective_air_value = I0_estimate
        else:
            effective_air_value = self.air_value if self.air_value is not None else data_max
            print(f"Using air_value = {effective_air_value:.1f} as I₀")

        print(f"Applying log transformation: p = -log(I / {effective_air_value:.1f})")

        for start in range(0, self.N_angles, chunk_size):
            end = min(start + chunk_size, self.N_angles)

            # Load chunk
            chunk = self.projections[start:end].astype(np.float32)

            # Normalize by I₀ to get transmission T = I/I₀
            transmission = chunk / (effective_air_value + epsilon)

            # Clamp to prevent log of negative/zero values
            # Also clamp max to 1.0 (transmission can't exceed unattenuated)
            transmission = np.clip(transmission, epsilon, 1.0)

            # Apply negative log: p = -log(T)
            # This gives line integrals of attenuation (always positive)
            chunk = -np.log(transmission)

            # Write back
            self.projections[start:end] = chunk
            self.projections.flush()

        print(f"Log transformation complete. Projections now represent line integrals of attenuation.")

    def preprocess(self):
        """
        Apply proper preprocessing for HU reconstruction:
        1. Flat-field correction: T = (I - I_dark) / (I_bright - I_dark)
        2. Log transform: p = -log(T)
        3. Ring artifact correction: sinogram median filtering

        This converts raw detector intensities to line integrals of attenuation,
        which is required for proper FDK reconstruction with physically meaningful output.

        Requires bright_field and dark_field to be set.

        NOTE: Creates a new float32 memmap because the original int16 memmap
        cannot store the small float values from log transformation.
        """
        if self.bright_field is None or self.dark_field is None:
            raise ValueError("bright_field and dark_field must be set for preprocessing")

        print("=" * 60)
        print("Preprocessing for HU reconstruction")
        print("=" * 60)

        # Report input statistics
        sample_proj = self.projections[0].astype(np.float32)
        print(f"Raw projection statistics:")
        print(f"  Range: [{sample_proj.min():.0f}, {sample_proj.max():.0f}]")
        print(f"  Mean: {sample_proj.mean():.0f}")
        print(f"  Original dtype: {self.projections.dtype}")

        print(f"Bright field statistics:")
        print(f"  Range: [{self.bright_field.min():.0f}, {self.bright_field.max():.0f}]")
        print(f"  Mean: {self.bright_field.mean():.0f}")

        print(f"Dark field statistics:")
        print(f"  Range: [{self.dark_field.min():.0f}, {self.dark_field.max():.0f}]")
        print(f"  Mean: {self.dark_field.mean():.0f}")

        # Create a new float32 memmap for the preprocessed data
        # The original int16 memmap cannot store float values properly!
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
        float_projections = np.memmap(
            temp_file.name,
            dtype=np.float32,
            mode='w+',
            shape=self.projections.shape
        )
        print(f"  Created float32 memmap: {temp_file.name}")

        # Process in chunks to manage memory
        chunk_size = 20

        for start in range(0, self.N_angles, chunk_size):
            end = min(start + chunk_size, self.N_angles)

            # Load chunk from original int16 projections
            chunk = self.projections[start:end].astype(np.float32)

            # Step 1: Flat-field correction
            transmission = flat_field_correction(chunk, self.bright_field, self.dark_field)

            # Step 2: Log transform
            line_integrals = log_transform_transmission(transmission)

            # Write to NEW float32 memmap
            float_projections[start:end] = line_integrals
            float_projections.flush()

            if start == 0:
                # Report intermediate statistics for first chunk
                print(f"\nAfter preprocessing (first chunk):")
                print(f"  Transmission range: [{transmission.min():.4f}, {transmission.max():.4f}]")
                print(f"  Line integral range: [{line_integrals.min():.3f}, {line_integrals.max():.3f}]")

        # Replace projection reference with float32 memmap
        self.projections = float_projections

        # Step 3: Ring artifact correction
        # Apply sinogram median filtering to remove detector pixel bias
        self.projections = self._correct_ring_artifacts(self.projections)

        # Final statistics
        final_sample = self.projections[self.N_angles // 2]
        print(f"\nPreprocessing complete. Line integral statistics (middle projection):")
        print(f"  Range: [{final_sample.min():.3f}, {final_sample.max():.3f}]")
        print(f"  Mean: {final_sample.mean():.3f}")
        print(f"  New dtype: {self.projections.dtype}")
        print("=" * 60)

    def _correct_ring_artifacts(self, projections, kernel_size=101, strength=1.0):
        """
        Correct ring artifacts using sorting-based sinogram filtering (Titarenko method).

        This is more effective than simple median filtering because it:
        1. Sorts the sinogram along the angle axis (makes stripes horizontal)
        2. Applies strong smoothing on sorted data (safe since no real features align)
        3. Subtracts the difference between sorted original and sorted filtered
        4. This isolates the per-column bias regardless of object structure

        Ring artifacts occur when detector pixels have systematic errors that are
        constant across all rotation angles. In the sinogram (angles × detector columns),
        these errors appear as vertical stripes, which become rings after backprojection.

        Args:
            projections: Line integrals [n_angles, height, width] as memmap
            kernel_size: Size of filter kernel (odd number, larger = more aggressive)
            strength: Correction strength (0.0-1.0), use <1.0 to preserve some high-frequency detail

        Returns:
            Corrected projections with ring artifacts reduced
        """
        from scipy.ndimage import uniform_filter1d

        print(f"\nApplying ring artifact correction (Titarenko method)...")
        print(f"  kernel_size={kernel_size}, strength={strength}")

        # Process each detector row (height) separately to manage memory
        for row_idx in range(projections.shape[1]):
            if row_idx % 500 == 0:
                print(f"  Processing detector row {row_idx}/{projections.shape[1]}...")

            # Extract sinogram for this row: shape (n_angles, width)
            sinogram = projections[:, row_idx, :].copy()

            # Step 1: Sort along angle axis (axis=0)
            # This makes vertical stripes (ring artifacts) appear as horizontal features
            sort_indices = np.argsort(sinogram, axis=0)
            sorted_sinogram = np.take_along_axis(sinogram, sort_indices, axis=0)

            # Step 2: Apply strong smoothing along the sorted axis
            # Since we sorted, real anatomy features are scattered, only systematic column bias remains aligned
            filtered_sorted = uniform_filter1d(sorted_sinogram, size=kernel_size, axis=0, mode='nearest')

            # Step 3: Compute the stripe component (difference in sorted domain)
            stripe_component_sorted = sorted_sinogram - filtered_sorted

            # Step 4: Unsort to get back the stripe component in original order
            unsort_indices = np.argsort(sort_indices, axis=0)
            stripe_component = np.take_along_axis(stripe_component_sorted, unsort_indices, axis=0)

            # Step 5: Subtract stripe component (with strength factor)
            corrected = sinogram - strength * stripe_component

            # Write back
            projections[:, row_idx, :] = corrected.astype(np.float32)

        projections.flush()
        print(f"  Ring artifact correction complete.")

        return projections

    def pre_weight(self):
        """
        Applies FDK weighting to correct for cone-beam divergence.
        Each projection is multiplied by:
            w(a,b) = R / sqrt(R^2 + a^2 + b^2)
        where a and b are the physical coordinates (in mm) relative to the detector center.
        """
        # Create coordinate grids for a and b (physical units)
        a = (torch.arange(self.N_a, device=device) - self.central_pixel_a) * self.da  # shape (N_a,)
        b = (torch.arange(self.N_b, device=device) - self.central_pixel_b) * self.db  # shape (N_b,)

        B, A = torch.meshgrid(b, a, indexing='ij')

        # Compute weighting factor (avoid division by zero)
        denom = torch.sqrt(self.SDD**2 + A**2 + B**2) + 1e-8
        weight = self.SDD / denom  # shape (N_a, N_b)

        chunk_size = 20
        
        for start in range(0, len(self.angles), chunk_size):
            end = min(start + chunk_size, len(self.angles))

            # slice → torch on GPU
            chunk = torch.from_numpy(self.projections[start:end]).to(device).to(torch.float32)

            chunk = chunk * weight # shape (chunk_size, N_b, N_a)

            # write back
            self.projections[start:end] = chunk.cpu().numpy()

            self.projections.flush()

    def ramp_filter(self):
        """
        Apply a 1D ramp filter (Ram-Lak) to each row (along a) of the parallel projections.
        Filtering is done in the frequency domain.

        Uses cosine-windowed ramp filter for all reconstructions to ensure
        consistent noise characteristics between HU and raw modes.
        """
        freqs = torch.fft.rfftfreq(self.N_a, d=self.da).to(device)

        # Use cosine-windowed filter for all reconstructions (consistent noise characteristics)
        # This ensures matched noise between HU and raw reconstructions
        cosine_window = torch.cos(torch.pi * freqs / (2 * freqs.abs().max()))
        cosine_kernel = (torch.abs(freqs) * cosine_window).unsqueeze(0)
        cosine_kernel = cosine_kernel.clamp(min=0) / cosine_kernel.max()
        filter_kernel = cosine_kernel

        for i in range(0, len(self.angles)):

            proj = torch.from_numpy(self.projections[i]).to(device).to(torch.float32) # proj: shape (N_b, N_a)

            proj = torch.fft.rfft(proj, dim=1, norm='forward')  # shape (N_b, N_a/2+1)

            proj = proj * filter_kernel

            proj = torch.fft.irfft(proj, n=self.N_a, dim=1, norm='forward')  # shape (N_b, N_a)

            self.projections[i] = proj.cpu().numpy()

            self.projections.flush()

    def backprojection(self):
        Nx, Ny, Nz = self.vol_shape

        with torch.no_grad():
            # Create 1D coordinate vectors with lower precision if acceptable.
            self.x = (torch.arange(Nx, device=device, dtype=torch.float32) - (Nx - 1) / 2) * self.dx + self.vol_origin[0]
            self.y = (torch.arange(Ny, device=device, dtype=torch.float32) - (Ny - 1) / 2) * self.dx + self.vol_origin[1]
            self.z = (torch.arange(Nz, device=device, dtype=torch.float32) - (Nz - 1) / 2) * self.dz + self.vol_origin[2]

            # Instead of expanding full grids, process in chunks (e.g., along the z-axis)
            self.reconstructed_volume = torch.zeros((Nx, Ny, Nz), device=device, dtype=torch.float32)
            
            # Choose a chunk size along the z-axis (adjust based on available memory)
            chunk_size = max(1, Nz // 10)
            
            for z_start in range(0, Nz, chunk_size):
                z_end = min(z_start + chunk_size, Nz)
                z_chunk = self.z[z_start:z_end]
                # Create a grid for the current chunk using meshgrid
                X, Y, Z_chunk = torch.meshgrid(self.x, self.y, z_chunk, indexing='ij')
                
                # Loop over each projection angle
                for i, beta in enumerate(self.angles):
                    proj = torch.from_numpy(self.projections[i]).to(device).to(torch.float32)
                    proj = proj.unsqueeze(0).unsqueeze(0)  # shape (1, 1, N_b, N_a)

                    U = self.SDD + X * torch.cos(beta) + Y * torch.sin(beta) + 1e-8
                    a = self.SDD * (-X * torch.sin(beta) + Y * torch.cos(beta)) / U
                    b = Z_chunk * self.SDD / U

                    a = a + (self.N_a / 2 - self.central_pixel_a) * self.da
                    b = b + (self.N_b / 2 - self.central_pixel_b) * self.db

                    a = a / (self.a_length / 2)
                    b = b / (self.b_length / 2)

                    # Prepare grid tensor for grid_sample
                    grid_tensor = torch.stack((a, b), dim=-1).unsqueeze(0)  # shape (1, Nx, Ny, chunk_size, 2)
                    grid_tensor = grid_tensor.view(1, Nx, -1, 2)  # reshape appropriately

                    sampled = F.grid_sample(proj, grid_tensor, mode='bilinear', align_corners=True)
                    self.reconstructed_volume[:, :, z_start:z_end] += sampled[0, 0].view(Nx, Ny, -1) * (self.SDD / U)**2

                    #self.reconstructed_volume[:, :, z_start:z_end] *= (self.SDD / U)**2

                    # Accumulate the results into the correct chunk
                    #self.reconstructed_volume[:, :, z_start:z_end] += output_tensor

                    # Free up memory for the current iteration
                    del proj, grid_tensor, sampled

                torch.cuda.empty_cache()

            #self.reconstructed_volume = reconstructed_volume.to(dtype=torch.float32)

        # Apply angular normalization (Δβ) to ensure proper scaling
        # This converts the discrete sum to a proper integral approximation:
        # f(x,y,z) = Σᵢ p_filtered(βᵢ) * (R/L)² * Δβ
        # Without this, reconstructions with different numbers of projections
        # would have different intensity scales (proportional to N_angles)
        if self.N_angles > 1:
            angle_range = float(self.angles[-1] - self.angles[0])
            # Handle angle wraparound (e.g., when angles are modulo 360 and span ~360°)
            # If computed range is very small (<1°) but we have many projections,
            # this indicates the angles wrapped around - assume full 360° scan
            if abs(angle_range) < np.pi / 180:  # Less than 1 degree
                print(f"Warning: Detected angle wraparound (range={np.rad2deg(angle_range):.4f}°). Assuming full 360° scan.")
                delta_beta = 2 * np.pi / self.N_angles
            else:
                delta_beta = angle_range / (self.N_angles - 1)
        else:
            delta_beta = 2 * np.pi  # Single projection edge case
        self.reconstructed_volume *= delta_beta
        print(f"Applied angular normalization: Δβ = {float(delta_beta):.6f} rad ({float(delta_beta) * 180 / np.pi:.4f}°)")

    def convert_to_hu(self):
        """
        Convert reconstructed volume to Hounsfield Units.

        Uses the scale-independent HU formula that maps:
            - Air → -1000 HU
            - Water/soft tissue → 0 HU

        HU = (μ - μ_water) / (μ_water - μ_air) × 1000

        This formula is independent of the reconstruction's internal scaling
        because the units cancel out. The FDK produces values proportional to μ,
        and we identify air/water references from the volume statistics.

        With proper preprocessing (bright/dark field + log transform):
            - Air regions have μ ≈ 0 (low attenuation)
            - Water/soft tissue has μ > 0 (higher attenuation)

        Standard HU values:
            - Air: -1000 HU
            - Water: 0 HU
            - Soft tissue: 20-80 HU
            - Bone: +1000 to +3000 HU
        """
        vol_np = self.reconstructed_volume.cpu().numpy() if hasattr(self.reconstructed_volume, 'cpu') else self.reconstructed_volume

        # Identify air reference from lowest values (background/air regions)
        # With proper flat-field correction, air should have μ ≈ 0
        ref_air = float(np.percentile(vol_np, 5))

        # Identify water/tissue reference from phantom regions
        # For a typical phantom, the 85th percentile represents soft tissue/water equivalent
        # (Higher than 50th because most of volume is air outside phantom)
        ref_water = float(np.percentile(vol_np, 85))

        # Validate: water should have higher attenuation than air
        if ref_water <= ref_air + 1e-6:
            print("Warning: Could not distinguish air and water regions.")
            print("Using 5th and 95th percentiles as fallback.")
            ref_air = float(np.percentile(vol_np, 5))
            ref_water = float(np.percentile(vol_np, 95))

        print(f"Scale-independent HU calibration:")
        print(f"  Air reference (5th percentile): {ref_air:.6f}")
        print(f"  Water reference (85th percentile): {ref_water:.6f}")
        print(f"  Reference spread: {ref_water - ref_air:.6f}")

        # Apply scale-independent HU formula:
        # HU = (μ - μ_water) / (μ_water - μ_air) × 1000
        # This maps: air → -1000 HU, water → 0 HU
        scale = 1000.0 / (ref_water - ref_air + 1e-10)
        self.reconstructed_volume = (self.reconstructed_volume - ref_water) * scale

        # Clamp to valid HU range
        self.reconstructed_volume = torch.clamp(self.reconstructed_volume, -1024, 4095)

        # Report statistics
        vol_min = float(self.reconstructed_volume.min())
        vol_max = float(self.reconstructed_volume.max())
        vol_mean = float(self.reconstructed_volume.mean())
        vol_std = float(self.reconstructed_volume.std())
        print(f"HU conversion complete:")
        print(f"  Range: [{vol_min:.0f}, {vol_max:.0f}] HU")
        print(f"  Mean: {vol_mean:.0f} HU, Std: {vol_std:.0f} HU")

        # Validate calibration
        if vol_mean > -800 or vol_mean < -1000:
            print(f"  Note: Mean HU is {vol_mean:.0f}. For a phantom in air, expect ~-900 to -950 HU.")

    def save_volume(self, filename=None, data_type=np.dtype('>h'), title="FDK Reconstruction", 
                   subject="CT Reconstruction", water_hu=0.0, air_hu=-1000.0, bone_hu=1000.0):
        """
        Save the reconstructed volume as a VFF file.
        
        Parameters:
        - filename: string, output filename (if None, auto-generated from folder_name)
        - data_type: numpy dtype, data type for the output (default: '>h' for 16-bit big-endian)
        - title: string, title for the VFF file
        - subject: string, subject description for the VFF file
        - water_hu: float, Hounsfield Unit value for water calibration
        - air_hu: float, Hounsfield Unit value for air calibration
        - bone_hu: float, Hounsfield Unit value for bone calibration
        """
        if filename is None:
            filename = self.folder_name+".vff"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert volume to CPU and numpy if it's still on GPU
        if isinstance(self.reconstructed_volume, torch.Tensor):
            volume_np = self.reconstructed_volume.cpu().numpy()
        else:
            volume_np = self.reconstructed_volume
        
        # Get volume dimensions (VFF format expects x, y, z order)
        Nx, Ny, Nz = volume_np.shape
        
        # Convert data type if necessary
        if data_type == np.dtype('>h'):  # 16-bit
            # Scale to 16-bit range if needed
            if volume_np.max() <= 1.0:  # Normalized data
                volume_scaled = (volume_np * 32767).astype(np.int16)
            else:
                volume_scaled = volume_np.astype(np.int16)
            bits = 16
        elif data_type == np.dtype('>b'):  # 8-bit
            # Scale to 8-bit range if needed
            if volume_np.max() <= 1.0:  # Normalized data
                volume_scaled = (volume_np * 127).astype(np.int8)
            else:
                volume_scaled = volume_np.astype(np.int8)
            bits = 8
        else:
            volume_scaled = volume_np.astype(data_type)
            bits = data_type.itemsize * 8
        
        # Convert to big-endian format and reshape to (z, y, x) for VFF
        # Flip y-axis to match VFF coordinate system convention
        volume_vff = volume_scaled.transpose(2, 1, 0)[:, ::-1, :].astype(data_type)
        
        # Calculate statistics for the volume
        vol_min = float(volume_np.min())
        vol_max = float(volume_np.max())
        
        # Calculate angle increment if we have multiple angles
        angle_increment = 0.0
        if len(self.angles) > 1:
            angle_increment = float(torch.diff(self.angles).mean().item()) * 180.0 / np.pi  # Convert to degrees
        
        # Get current timestamp
        from datetime import datetime
        current_time = datetime.now().isoformat()
        
        # Calculate element size (assuming isotropic pixels, convert to appropriate units)
        elementsize = self.dx / 1000.0  # Convert from mm to meters if needed, or keep as mm
        
        # Create VFF header with comprehensive information
        header_lines = [
            "ncaa",
            f"rank=3;",
            f"type=raster;",
            f"modality=CT;",
            f"size={Nx} {Ny} {Nz};",
            f"origin={self.vol_origin[0]:.4f} {self.vol_origin[1]:.4f} {self.vol_origin[2]:.4f};",
            f"y_bin=1;",
            f"z_bin=1;",
            f"bands=1;",
            f"bits={bits};",
            f"format=slice;",
            f"title={title};",
            f"subject={subject};",
            f"date={current_time};",
            f"center_of_rotation={self.central_pixel_a * self.da:.6f};",
            f"central_slice={Nz//2:.6f};",
            f"rfan_y={self.R_s:.6f};",
            f"rfan_z={self.R_s:.6f};",
            f"angle_increment={angle_increment:.6f};",
            f"reverse_order=no;",
            f"min={vol_min:.6f};",
            f"max={vol_max:.6f};",
            f"spacing={self.dx:.2f} {self.dx:.2f} {self.dz:.2f};",
            f"elementsize={elementsize:.6f};",
            f"water={water_hu:.6f};",
            f"air={air_hu:.6f};",
            f"boneHU={bone_hu:.0f};",
            f"recon_sysid={torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'};",
            f"rawsize={volume_vff.nbytes};",
            ""
        ]
        
        # Write VFF file
        with open(filename, 'wb') as f:
            # Write header
            for line in header_lines:
                f.write(line.encode('latin-1') + b'\n')
            
            # Write form feed character to separate header from data
            f.write(b'\f')
            
            # Write binary data
            f.write(volume_vff.tobytes())
        
        print(f"Volume saved as VFF file: {filename}")
        print(f"Volume dimensions: {Nx} x {Ny} x {Nz}")
        print(f"Voxel spacing: {self.dx:.2f} x {self.dx:.2f} x {self.dz:.2f} mm")
        print(f"Volume origin: ({self.vol_origin[0]:.2f}, {self.vol_origin[1]:.2f}, {self.vol_origin[2]:.2f}) mm")
        print(f"Data range: [{vol_min:.3f}, {vol_max:.3f}]")
        print(f"Data type: {data_type}, {bits} bits per voxel")
        print(f"File size: {os.path.getsize(filename) / (1024**2):.2f} MB")
        if len(self.angles) > 1:
            print(f"Angle increment: {angle_increment:.3f} degrees")
            print(f"Total angles: {len(self.angles)}")

    def display_volume(self):

        torch.cuda.empty_cache()
        self.x = self.x.cpu().numpy()
        self.y = self.y.cpu().numpy()
        self.z = self.z.cpu().numpy()

        num_bytes = self.reconstructed_volume.element_size() * self.reconstructed_volume.nelement()
        num_megabytes = num_bytes / (1024 ** 2)
        print(f"Reconstructed volume uses approximately {num_megabytes:.2f} MB")

        # normalise the reconstruction values by gamma factor
        gamma = 1 # gamma correction to brighten low values
        torch.cuda.empty_cache()
        
        self.reconstructed_volume /= self.reconstructed_volume.max()
        self.reconstructed_volume *= 256

        for i in range(self.reconstructed_volume.shape[2]):
            z_slice = self.reconstructed_volume[:, :, i].cpu().numpy()

            if i == len(self.z)-1:
                break
            os.makedirs(self.folder_name, exist_ok=True)

            fig, ax = plt.subplots()

            # Display using standard imshow convention (no transpose)
            # z_slice is (Nx, Ny), displayed as (rows=x, cols=y) to match VFF loading convention
            ax.imshow(z_slice, cmap='gray', origin='lower',
                     extent=[self.y.min(), self.y.max(), self.x.min(), self.x.max()])
            ax.set_xlabel('Y position (mm)')
            ax.set_ylabel('X position (mm)')

            ax.set_aspect('equal', adjustable='box') # set aspect ratio to 1:1 (no squishing of pixels)

            # include source locations
            # Separate x and y arrays for scatter:
            if self.source_locations is not None:
                for source_pos in self.source_locations:
                    if source_pos[2] >= self.z[i] and source_pos[2] < self.z[i+1]:
                        ax.scatter(source_pos[1], source_pos[0], c='red', s=10)

            ax.set_title(f'Reconstruction slice {i} (z: {self.z[i]:.2f}-{self.z[i+1]:.2f} mm)')

            plt.savefig(f'{self.folder_name}/reconstruction_slice_{i:03d}_{self.z[i]:.2f}-{self.z[i+1]:.2f}.png', dpi=500, bbox_inches='tight')
            plt.close(fig)

    def reconstruct(self, display_volume=True):
        """
        Complete reconstruction pipeline.

        If output_hu is True and bright_field/dark_field are provided:
        1. Applies proper preprocessing (flat-field correction + log transform)
        2. Runs FDK reconstruction
        3. Converts to Hounsfield Units using theoretical calibration

        Otherwise, runs standard FDK on raw intensities.
        """
        # Step 1: Optional preprocessing for HU reconstruction
        if self.output_hu and self.bright_field is not None and self.dark_field is not None:
            print("\nApplying preprocessing for HU reconstruction...")
            self.preprocess()

        # Step 2: Standard FDK pipeline
        print("\nPre-weighting projections...")
        self.pre_weight()
        print("Applying ramp filter...")
        self.ramp_filter()
        print("Backprojecting...")
        self.backprojection()

        # Step 3: Optional HU conversion
        if self.output_hu:
            print("\nConverting to Hounsfield Units...")
            self.convert_to_hu()

        # Step 4: Save results
        print("\nSaving reconstruction...")
        self.save_volume()
        print("Reconstruction saved.")

        if display_volume == True:
            self.display_volume()
            print("Reconstruction plots created.")


