import os
import random
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import argparse

def create_split(args):
    # Normalize desired_scans_in_testing to a proper list of scan names
    ds = args.desired_scans_in_testing
    if isinstance(ds, list) and len(ds) == 1:
        raw = ds[0].strip("[]")
        args.desired_scans_in_testing = [s.strip().strip("'\"") for s in raw.split(",") if s.strip()]
    elif isinstance(ds, str):
        raw = ds.strip("[]")
        args.desired_scans_in_testing = [s.strip().strip("'\"") for s in raw.split(",") if s.strip()]

    # Define the folder structure
    base_dir = os.path.join(os.getcwd(), 'data', 'scans')
    all_scan_folders = []

    # Get a list of subfolders (scan folders)
    for subfolder in sorted(os.listdir(base_dir)):
        subfolder_path = os.path.join(base_dir, subfolder)
        if os.path.isdir(subfolder_path):
            if len(all_scan_folders) < args.number_of_scans_in_total or subfolder in args.desired_scans_in_testing:
                all_scan_folders.append(subfolder_path)

    # Set random seed and split folders randomly including desired scans
    random.seed(args.seed)
    # Identify desired scans in test set
    desired_paths = {os.path.join(base_dir, scan) for scan in args.desired_scans_in_testing}
    desired_test = [f for f in all_scan_folders if f in desired_paths]
    remaining = [f for f in all_scan_folders if f not in desired_test]
    random.shuffle(remaining)
    total = len(all_scan_folders)
    n_test = total - int(args.train_test_split * total)
    additional = max(0, n_test - len(desired_test))
    test_folders = desired_test + remaining[:additional]
    train_folders = remaining[additional:]

    # Function to get image triplets from a folder
    def get_image_triplets_from_folder(folder, train=True):
        vff_files = sorted(glob.glob(os.path.join(folder, 'acq-00-*.vff')))
        #vff_files = sorted(glob.glob(os.path.join(folder, 'uwarp-00-*.vff')))
        inputs = []
        outputs = []
        step_size = 1 if train else 2  # Use step size of 1 for training, 2 for testing
        for i in range(1, len(vff_files) - 1, step_size):
            # Create the input set (two outer images) and the output set (the middle image)
            input_images = [vff_files[i - 1], vff_files[i + 1]]
            output_image = vff_files[i]

            if input_images[0].split('-')[1] != input_images[1].split('-')[1] or input_images[0].split('-')[1] != output_image.split('-')[1]:
                # Skip if the images aren't all from the same scan
                continue
            inputs.append(input_images)
            outputs.append(output_image)
        return inputs, outputs

    # Collect image triplets for training and testing
    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []

    # Process the training folders
    for folder in train_folders:
        inputs, outputs = get_image_triplets_from_folder(folder, train=True)
        train_inputs.extend(inputs)
        train_outputs.extend(outputs)

    # Process the testing folders
    for folder in test_folders:
        inputs, outputs = get_image_triplets_from_folder(folder, train=False)
        test_inputs.extend(inputs)
        test_outputs.extend(outputs)

    # Convert to DataFrame
    train_data = {'train_x': [str(i) for i in train_inputs], 'train_y': [str(i) for i in train_outputs]}
    test_data = {'test_x': [str(i) for i in test_inputs], 'test_y': [str(i) for i in test_outputs]}

    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # Combine train and test DataFrames
    final_df = pd.concat([df_train, df_test], ignore_index=True)

    # Save the DataFrame to a CSV
    final_df.to_csv(args.output_csv, index=False)

    # print only filenames not whole directories
    print(f'Train folders are: {[os.path.basename(f) for f in train_folders]}')
    print(f'Test folders are: {[os.path.basename(f) for f in test_folders]}')

    print(f"Dataset saved as '{args.output_csv}'")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create train/test split for projection infilling dataset.")
    parser.add_argument('--train_test_split', type=float, default=0.8, help='Proportion of data to use for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_csv', type=str, default='data/scans/training_testing_split.csv', help='Output CSV file for train/test split (default: data/scans/training_testing_split.csv)')
    parser.add_argument('--desired_scans_in_testing', nargs='*', default=['Scan_1680','Scan_1681, Scan_1539'], help='List of desired scan folders to include in testing')
    parser.add_argument('--number_of_scans_in_total', type=int, default=8, help='Total number of scans (default: 8)')
    args = parser.parse_args()

    create_split(args)