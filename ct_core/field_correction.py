import torch
import numpy as np
import os
import shutil

from .vff_io import read_vff, VFFDataset

def bright_dark_field_correction(projections, dark_field, bright_field):

    corrected = (projections - dark_field) / (bright_field - dark_field + 1e-8)

    scale = np.max(projections) / np.max(corrected)
    corrected = corrected * scale

    return corrected

def undo_projection_pre_processing(projection, dark_field_2d):
    be        = dark_field_2d.byteswap()
    # 2) view as big‐endian dtype
    be        = be.view(projection.dtype.newbyteorder('>'))
    # 3) restore the singleton z‐dimension
    restored = np.expand_dims(be, axis=0)  # shape: (1, y, x)
    return restored

def write_vff(filename, header, data, verbose=True):
    """
    Write a VFF file from a header dict and 2D/3D NumPy array.

    :param filename: Path to output .vff file
    :param header: Dict with metadata keys (e.g., 'bits'); size is inferred from `data`
    :param data: 3D NumPy array shaped (z, y, x) or 2D array (y, x)
    """
    # Convert input to NumPy array and ensure 3D shape
    arr = np.array(data, copy=False)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Data must be 2D or 3D array, got {arr.ndim}D")
    zdim, ydim, xdim = arr.shape

    # Build header entirely from inferred dimensions and provided bits
    bits = int(header.get('bits', 16))
    dtype = np.dtype('>u1') if bits == 8 else np.dtype('>i2') if bits == 16 else None
    if dtype is None:
        raise ValueError("Unsupported bits per voxel: must be 8 or 16")

    # Assemble canonical header fields
    hdr = header.copy()
    hdr['size'] = [xdim, ydim, zdim]
    hdr['bits'] = bits
    hdr.setdefault('format', 'unsigned-byte' if bits == 8 else 'signed-short')
    hdr.setdefault('endian', 'big')

    # Build header text
    lines = []
    for key in ('size', 'bits', 'format', 'endian'):
        val = hdr[key]
        if key == 'size':
            val = ' '.join(map(str, val))
        lines.append(f"{key} = {val};")
    for key, val in hdr.items():
        if key in ('size', 'bits', 'format', 'endian'):
            continue
        lines.append(f"{key} = {val};")
    header_text = "\n".join(lines) + "\n\f\n"

    # Ensure big-endian C-contiguous data
    if arr.dtype != dtype or arr.dtype.byteorder != '>' or not arr.flags['C_CONTIGUOUS']:
        arr_be = arr.astype(dtype)
    else:
        arr_be = arr

    if verbose:
        print(f"Writing VFF to {filename}: shape={arr_be.shape}, dtype={arr_be.dtype}")

    with open(filename, 'wb') as f:
        f.write(header_text.encode('latin-1'))

        arr_be.tofile(f)

def initiate_bright_dark_correction(projections_folder, output_folder=None):

    if output_folder is None:
        output_folder = os.path.join(projections_folder, 'Corrected')

    # Check if output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the dark and bright field images
    _, dark_field_arr = read_vff(os.path.join(projections_folder, 'dark.vff'), verbose=False)
    dark_field_arr = dark_field_arr.squeeze(0)
    dark_field = dark_field_arr.byteswap().view(dark_field_arr.dtype.newbyteorder())

    _, bright_field_arr = read_vff(os.path.join(projections_folder, 'bright.vff'), verbose=False)
    bright_field_arr = bright_field_arr.squeeze(0)
    bright_field = bright_field_arr.byteswap().view(bright_field_arr.dtype.newbyteorder())

    xml_file = os.path.join(projections_folder, 'scan.xml')

    # Load all projections and angles
    dataset = VFFDataset(projections_folder, xml_file, save_headers=True, tensor_projections=False,
                                                    exclude_pred_paths=True, sub_scan='-00-')
    projections = dataset.projections
    headers = dataset.headers

    for i, projection in enumerate(projections):
        # Correct the projection
        corrected_proj = bright_dark_field_correction(projection, dark_field, bright_field)
        # Save the corrected projection as form: uwarp-00-0000.vff
        corrected_filename = os.path.join(output_folder, f'uwarp-00-{i:04d}.vff')

        write_vff(corrected_filename, headers[i], corrected_proj)

    # write scan.xml file to output_folder as well
    shutil.copyfile(xml_file, os.path.join(output_folder, 'scan.xml'))


if __name__ == "__main__":
    from .paths import RESULTS_DIR
    projections_folder = str(RESULTS_DIR / 'Scan_1539_raw_with_preds')
    output_folder = str(RESULTS_DIR / 'Scan_1539_uwarp_no_pred')
    initiate_bright_dark_correction(projections_folder, output_folder)
