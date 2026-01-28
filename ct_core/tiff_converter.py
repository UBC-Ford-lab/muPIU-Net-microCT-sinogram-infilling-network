# Description: This script takes vff file data and saves it as TIFF images
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import os
import numpy as np
import imageio

def save_vff_to_tiff(vff_data, target_directory=None, filename=None, verbose=True, compute_average_img=True):
    '''
    This function takes the data from a VFF file and saves it as TIFF images.
    :param vff_data: The 3D numpy array with the voxel data
    :param target_directory: The directory where the TIFF images should be saved
    :param filename: Optional filename for 2D data
    :param verbose: Boolean to print out the progress
    :param compute_average_img: Boolean to compute the average image and save it as well
    :return: None
    '''
    # Default target directory
    if target_directory is None:
        target_directory = os.path.join(os.getcwd(), 'TIFF_output')

    # Create the target directory if it does not exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Create the TIFF files
    if vff_data.ndim == 3:
        for slice_index in range(len(vff_data)):
            # Save the VFF data as a TIFF image
            imageio.imwrite(os.path.join(target_directory, f'slice_{slice_index}.tiff'), vff_data[slice_index])

            if verbose:
                print(f"Saved slice {slice_index} as TIFF image")

    elif vff_data.ndim == 2:
        # Save the VFF data as a TIFF image
        if filename is not None:
            # Save the VFF data as a TIFF image with the specified filename
            imageio.imwrite(os.path.join(target_directory, f'{filename}.tiff'), vff_data)
        else:
            # Save the VFF data as a TIFF image with a default name
            imageio.imwrite(os.path.join(target_directory, 'image.tiff'), vff_data)

        if verbose:
            print("Saved the image as a TIFF file")

    else:
        raise ValueError("The input data must be a 2D or 3D numpy array")

    if compute_average_img and vff_data.ndim == 3:
        # Save the average image as a TIFF file
        imageio.imwrite(os.path.join(target_directory, 'averaged.tiff'), np.average(vff_data, axis=0))

        if verbose:
            print("Saved the average image as a TIFF file")

if __name__ == '__main__':
    # Load the VFF file
    from .vff_io import read_vff
    header, data = read_vff(filename='Base_model_comparison/lama_reconstruction.vff', verbose=True)

    # Save the VFF data as TIFF images
    save_vff_to_tiff(data, target_directory='Base_model_comparison/lama_reconstruction',
                     verbose=True, compute_average_img=False)
