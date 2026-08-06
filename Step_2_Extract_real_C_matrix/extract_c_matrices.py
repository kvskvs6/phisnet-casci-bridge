import os
import glob
import numpy as np

def read_orbcoef(file):
    '''Read MO coefficients from a Molcas (Ras)Orb file.
    (This function is copied directly from the teacher's molcas_utils.py)
    '''
    orbitals=[]
    with open(file,"r") as stream:
        got_orbitals=False
        anchor="pre-start"
        while not got_orbitals and anchor!="":
            anchor=stream.readline()
            if "#INFO" in anchor and "orbitals" in stream.readline():
                stream.readline()
                norbitals=int(stream.readline())
                if (norbitals%5)!=0:
                    block=norbitals//5+1
                else:
                    block=norbitals//5
            elif "#ORB" in anchor:
                for orb in range(1,norbitals+1):
                    stream.readline()
                    coef_st=""
                    for row in range(block):
                        coef_st=coef_st+stream.readline()
                    orbitals.append(np.array(coef_st.split(),dtype="float"))
                got_orbitals=True
                
    return np.array(orbitals).T # C matrix should have MO coefficients as columns

def extract_all_true_c_matrices():
    """
    Batch extract true C matrices from OpenMolcas .RasOrb files.
    Automatically extracts data for MB and VDZP basis sets per run and packs them into .npz files respectively.
    The script should be run in the same directory level as the 'open' folder.
    """
    basis_sets = ['MB', 'VDZP']
    
    for basis_set in basis_sets:
        print(f"Processing data for basis set {basis_set}...")
        
        # Match pattern: open/open_g2m05_XXXX/MB/open_g2m05_XXXX_MB.RasOrb
        pattern = os.path.join("open", "open_g2m05_[0-9][0-9][0-9][0-9]", basis_set, f"open_g2m05_[0-9][0-9][0-9][0-9]_{basis_set}.RasOrb")
        rasorb_files = glob.glob(pattern)
        
        if not rasorb_files:
            print(f"[Warning] No .RasOrb files found for basis set '{basis_set}', skipping.")
            print("-" * 40)
            continue
            
        c_matrices_dict = {}
        extracted_count = 0
        
        for file_path in sorted(rasorb_files):
            filename = os.path.basename(file_path)
            # Extract 4-digit index from filename, e.g., open_g2m05_0000_MB.RasOrb -> 0000
            parts = filename.split('_')
            if len(parts) >= 3:
                st_number_str = parts[2]
            else:
                st_number_str = os.path.basename(os.path.dirname(os.path.dirname(file_path))).split('_')[-1]
                
            try:
                # Extract true C matrix using the provided utility function
                C_true = read_orbcoef(file_path)
                
                # Store in dictionary
                c_matrices_dict[st_number_str] = C_true
                extracted_count += 1
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                
        # Pack all true C matrices for the current basis set into a .npz file
        if c_matrices_dict:
            # Name it TrueC to differentiate from future PredictedC
            npz_filename = f"all_true_C_matrices_{basis_set}.npz"
            np.savez_compressed(npz_filename, **c_matrices_dict)
            print(f"[Success] Extracted {extracted_count} true C matrices for the {basis_set} basis set!")
            print(f"File saved as: {npz_filename}")
        print("-" * 40)

if __name__ == "__main__":
    extract_all_true_c_matrices()
