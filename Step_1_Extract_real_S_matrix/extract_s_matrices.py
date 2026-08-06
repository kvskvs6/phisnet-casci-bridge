import os
import glob
import h5py
import numpy as np

def extract_all_s_matrices():
    """
    Batch extracts overlap matrices (S matrices) from OpenMolcas .rasscf.h5 files.
    The script should be executed outside the 'open' directory.
    """
    basis_sets = ['MB', 'VDZP']
    
    for basis_set in basis_sets:
        print(f"Processing data for basis set {basis_set}...")
        
        # Prepended 'open' directory to the path
        pattern = os.path.join("open", "open_g2m05_[0-9][0-9][0-9][0-9]", basis_set, f"open_g2m05_[0-9][0-9][0-9][0-9]_{basis_set}.rasscf.h5")
        h5_files = glob.glob(pattern)
        
        if not h5_files:
            print(f"[WARNING] No .rasscf.h5 files found for basis set '{basis_set}', skipping.")
            print("-" * 40)
            continue
            
        s_matrices_dict = {}
        extracted_count = 0
        
        for file_path in sorted(h5_files):
            # Extract the filename, e.g., open_g2m05_0003_MB.rasscf.h5
            filename = os.path.basename(file_path)
            # Extract the 4-digit identifier (e.g., 0003)
            parts = filename.split('_')
            if len(parts) >= 3:
                st_number_str = parts[2]
            else:
                st_number_str = os.path.basename(os.path.dirname(os.path.dirname(file_path))).split('_')[-1]
                
            try:
                with h5py.File(file_path, 'r') as h5file:
                    if 'NBAS' in h5file.attrs:
                        nbasis = h5file.attrs['NBAS'][0]
                    else:
                        print(f"Warning: {filename} is missing 'NBAS', skipping.")
                        continue
                    
                    if 'AO_OVERLAP_MATRIX' in h5file:
                        S_flat = h5file.get('AO_OVERLAP_MATRIX')[:]
                        S = S_flat.reshape(nbasis, nbasis)
                        
                        s_matrices_dict[st_number_str] = S
                        extracted_count += 1
                    else:
                        print(f"Warning: {filename} is missing 'AO_OVERLAP_MATRIX'")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        if s_matrices_dict:
            # Output file is saved in the same directory as the script
            npz_filename = f"all_S_matrices_{basis_set}.npz"
            np.savez_compressed(npz_filename, **s_matrices_dict)
            print(f"[SUCCESS] Extracted {extracted_count} S matrices for {basis_set} basis set!")
            print(f"File saved as: {npz_filename}")
        print("-" * 40)

if __name__ == "__main__":
    extract_all_s_matrices()
