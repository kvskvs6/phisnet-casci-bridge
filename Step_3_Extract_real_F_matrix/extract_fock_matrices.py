import os
import glob
import numpy as np

def extract_all_fock_matrices():
    """
    Batch extracts fock_real.npy files from open/open_g2m05_XXXX/<basis_set>/ directories.
    Packages all Fock matrices into a compressed .npz file, ensuring dictionary keys strictly correspond to their 4-digit indices.
    It is recommended to run this script in the parent directory containing the 'open' folder.
    """
    basis_sets = ['MB', 'VDZP']
    
    for basis_set in basis_sets:
        print(f"Processing Fock matrix data for basis set {basis_set}...")
        
        # Construct path pattern, e.g., open/open_g2m05_[0-9][0-9][0-9][0-9]/MB/fock_real.npy
        pattern = os.path.join("open", "open_g2m05_[0-9][0-9][0-9][0-9]", basis_set, "fock_real.npy")
        npy_files = glob.glob(pattern)
        
        if not npy_files:
            print(f"[Warning] No fock_real.npy files found for basis set '{basis_set}', skipping.")
            print("-" * 40)
            continue
            
        fock_matrices_dict = {}
        extracted_count = 0
        
        for file_path in sorted(npy_files):
            # Example file_path: open/open_g2m05_0003/MB/fock_real.npy
            
            # Extract 4-digit index from directory structure to ensure index matching
            # os.path.dirname(file_path) -> open/open_g2m05_0003/MB
            # os.path.dirname(os.path.dirname(file_path)) -> open/open_g2m05_0003
            parent_dir_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            
            # parent_dir_name is "open_g2m05_0003", split by '_' and take the last part
            parts = parent_dir_name.split('_')
            if len(parts) >= 3:
                st_number_str = parts[-1]  # Yields "0003"
            else:
                print(f"[Warning] Failed to parse standard index from path {file_path}, skipping.")
                continue
                
            try:
                # Read numpy .npy file directly
                fock_matrix = np.load(file_path)
                
                # Store in dictionary with extracted index as key
                fock_matrices_dict[st_number_str] = fock_matrix
                extracted_count += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        if fock_matrices_dict:
            # Save results in script directory, maintaining naming convention consistent with S matrices
            npz_filename = f"all_true_F_matrices_{basis_set}.npz"
            np.savez_compressed(npz_filename, **fock_matrices_dict)
            print(f"[Success] Extracted {extracted_count} Fock matrices for basis set {basis_set}!")
            print(f"File saved as: {npz_filename}")
            
        print("-" * 40)

if __name__ == "__main__":
    extract_all_fock_matrices()
