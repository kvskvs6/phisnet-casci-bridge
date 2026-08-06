"""
Molcas results collection script - Extracts predicted and reference energies and compiles them into energies.txt.

Usage:
    python collect_energies.py

Execute this script within the 20260610_Energy/ directory.
The script iterates over all subdirectories in Output/, reads the predicted .rasscf.h5 files,
retrieves the corresponding reference energies from dataset/open/, and writes the results to energies.txt.
"""

import os
import re
import glob
import csv
import sys

try:
    import h5py
except ImportError:
    print("h5py library is required: pip install h5py", file=sys.stderr)
    sys.exit(1)


def get_rds_root(script_dir):
    """Derives the rds root directory from the script's location (assuming the script is in OpenMolcas_projects/20260610_Energy/)."""
    return os.path.abspath(os.path.join(script_dir, '..', '..'))


def parse_output_dir_name(dirname):
    """
    Parses the output directory name in the format {basis}_{train}_{valid}_{test}_{index}
    Example: VDZP_800_100_100_0001
    Returns (basis, train, valid, test, index) or None.
    """
    # Use non-greedy matching to extract the basis, followed by 4 digits (the index is fixed at 4 digits).
    m = re.match(r'^(.+?)_(\d+)_(\d+)_(\d+)_(\d{4})$', dirname)
    if not m:
        return None
    basis = m.group(1)
    train = int(m.group(2))
    valid = int(m.group(3))
    test  = int(m.group(4))
    index = int(m.group(5))
    return basis, train, valid, test, index


def find_h5_file(directory):
    """Finds the .rasscf.h5 file in the directory (assumes only one exists)."""
    pattern = os.path.join(directory, '*.rasscf.h5')
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return candidates[0]  # Take the first match


def read_root_energies(h5_path):
    """Reads the ROOT_ENERGIES dataset from the h5 file and returns a numpy array."""
    with h5py.File(h5_path, 'r') as f:
        return f['ROOT_ENERGIES'][()]


def main():
    # The script is located in the 20260610_Energy/ directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rds_root = get_rds_root(script_dir)
    dataset_dir = os.path.join(rds_root, 'dataset', 'open')
    output_root = os.path.join(script_dir, 'Output')

    if not os.path.isdir(output_root):
        print(f"Error: Output directory does not exist: {output_root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    # The header assumes 3 roots. This can be dynamically adjusted, but is fixed here for the example format.
    header = ['Basis_set', 'Train_size', 'Val_size', 'Test_size', 'indx',
              'pred_E1', 'pred_E2', 'pred_E3',
              'ref_E1', 'ref_E2', 'ref_E3']

    # Iterate over Output subdirectories.
    for entry in os.listdir(output_root):
        subdir = os.path.join(output_root, entry)
        if not os.path.isdir(subdir):
            continue

        parsed = parse_output_dir_name(entry)
        if parsed is None:
            print(f"Warning: Cannot parse directory name '{entry}', skipping.")
            continue

        basis, train, valid, test, index = parsed

        # Locate the predicted h5 file.
        pred_h5 = find_h5_file(subdir)
        if pred_h5 is None:
            print(f"Warning: No .rasscf.h5 file found in {subdir}, skipping.")
            continue

        # Read predicted energies.
        try:
            pred_energies = read_root_energies(pred_h5)
        except Exception as e:
            print(f"Error: Failed to read predicted file {pred_h5}: {e}, skipping.")
            continue

        # Construct the reference file path.
        ref_subdir = os.path.join(dataset_dir,
                                  f'open_g2m05_{index:04d}',
                                  basis)
        ref_h5_name = f'open_g2m05_{index:04d}_{basis}.rasscf.h5'
        ref_h5 = os.path.join(ref_subdir, ref_h5_name)

        # Read reference energies.
        if not os.path.exists(ref_h5):
            print(f"Warning: Reference file missing {ref_h5}, skipping.")
            continue
        try:
            ref_energies = read_root_energies(ref_h5)
        except Exception as e:
            print(f"Error: Failed to read reference file {ref_h5}: {e}, skipping.")
            continue

        # Ensure the number of roots matches (hardcoded to 3 roots as an example, but can be made adaptive).
        if len(pred_energies) < 3 or len(ref_energies) < 3:
            print(f"Warning: Less than 3 energy roots in {entry}, skipping.")
            continue

        # Assemble row data: extract the first 3 roots.
        row = [basis, train, valid, test, f"{index:04d}"]
        row.extend(f"{e:.10f}" for e in pred_energies[:3])
        row.extend(f"{e:.10f}" for e in ref_energies[:3])
        rows.append(row)

    # Write to energies.txt.
    output_file = os.path.join(script_dir, 'energies.txt')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Successfully wrote {len(rows)} records to {output_file}")


if __name__ == '__main__':
    main()
