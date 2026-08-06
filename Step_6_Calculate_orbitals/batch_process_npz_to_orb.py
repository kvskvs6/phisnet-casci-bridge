import os
import glob
import zipfile
import numpy as np
from molcas_utils import write_orbfile

def batch_process_npz_to_orb():
    # --- Directory Configuration ---
    input_npz_dir = "Input: all_predicted_C_matrices"
    input_txt_dir = "Input: used_data_record"
    output_all_dir = "Output: all_predicted_Orb_files"
    output_test_dir = "Output: own_testset_predicted_Orb_files"
    
    # Check if core input directories exist
    if not os.path.exists(input_npz_dir):
        print(f"❌ Error: Matrix input directory '{input_npz_dir}' not found.")
        return
    if not os.path.exists(input_txt_dir):
        print(f"⚠️ Warning: Record input directory '{input_txt_dir}' not found. Test set separation will fail, but general extraction will proceed.")

    # Retrieve all .npz file paths
    npz_files = glob.glob(os.path.join(input_npz_dir, "*.npz"))
    if not npz_files:
        print(f"⚠️ No .npz files found in '{input_npz_dir}'.")
        return

    print(f"🔍 Found {len(npz_files)} .npz files. Starting batch processing...\n")

    for npz_path in npz_files:
        filename = os.path.basename(npz_path)
        
        # 1. Parse NPZ filenames to extract parameters
        prefix = "all_predicted_C_matrices_"
        if not filename.startswith(prefix) or not filename.endswith(".npz"):
            print(f"⏭️ Skipping non-standard filename: {filename}")
            continue
            
        params_str = filename.replace(prefix, "").replace(".npz", "")
        parts = params_str.split("_")
        
        if len(parts) < 4:
            print(f"❌ Parsing failed. Skipping file (incorrect format): {filename}")
            continue
            
        train_size = parts[0]
        val_size = parts[1]
        test_size = parts[2]
        basis_set = "_".join(parts[3:]) 

        # 2. Dynamically generate orbital types based on basis set name
        if basis_set == "MB":
            orb_types = np.array(16*["i"] + 5*["2"] + 11*["s"])
        elif basis_set == "VDZP":
            orb_types = np.array(16*["i"] + 5*["2"] + 84*["s"])
        else:
            print(f"⚠️ Warning: Unknown basis set '{basis_set}' in file '{filename}'. Skipped.")
            continue

        # 3. Create output directories (Full set and designated test set)
        folder_name = f"{train_size}_{val_size}_{test_size}_{basis_set}"
        
        all_out_path = os.path.join(output_all_dir, folder_name)
        test_out_path = os.path.join(output_test_dir, folder_name)
        
        os.makedirs(all_out_path, exist_ok=True)
        os.makedirs(test_out_path, exist_ok=True)
        
        print(f"📂 Processing: {filename}")
        print(f"   ├─ Basis set: {basis_set}")

        # 4. Parse corresponding .txt records to extract test set indices
        test_indices = set()
        txt_filename = f"used_data_record_{folder_name}.txt"
        txt_filepath = os.path.join(input_txt_dir, txt_filename)
        
        if os.path.exists(txt_filepath):
            with open(txt_filepath, 'r', encoding='utf-8') as f:
                in_test_set = False
                for line in f:
                    line = line.strip()
                    if line.startswith("--- TEST SET"):
                        in_test_set = True
                        continue
                    elif line.startswith("---") and in_test_set:
                        in_test_set = False
                        continue
                        
                    if in_test_set and line:
                        idx = line.split("_")[-1] 
                        test_indices.add(idx)
            print(f"   ├─ Successfully read txt record. Identified {len(test_indices)} samples for the designated test set.")
        else:
            print(f"   ├─ ⚠️ Corresponding txt record ({txt_filename}) not found. Test set extraction skipped.")

        # 5. Load .npz and extract matrices
        try:
            npz_data = np.load(npz_path)
            keys = npz_data.files
            
            all_success_count = 0
            test_success_count = 0
            
            # Collect paths of generated .Orb files for subsequent ZIP archiving
            all_generated_orbs = []
            test_generated_orbs = []
            
            for key in keys:
                c_matrix = npz_data[key]
                
                # Validate matrix dimensions
                if c_matrix.shape[0] != len(orb_types):
                     print(f"   ❌ Error: Matrix dimension for index {key} ({c_matrix.shape[0]}) does not match orbital rule length ({len(orb_types)})! Skipping this matrix.")
                     continue
                
                # Construct Orb filename (including training set size)
                orb_filename = f"all_predicted_Orb_files_{train_size}_{val_size}_{test_size}_{basis_set}_{key}.Orb"
                
                # [Task A]: Write full set Orb file and record physical path
                all_orb_filepath = os.path.join(all_out_path, orb_filename)
                write_orbfile(all_orb_filepath, c_matrix, orb_types)
                all_generated_orbs.append(all_orb_filepath)
                all_success_count += 1
                
                # [Task B]: If belonging to the test set, write designated test set Orb file and record path
                if key in test_indices:
                    test_orb_filepath = os.path.join(test_out_path, orb_filename)
                    write_orbfile(test_orb_filepath, c_matrix, orb_types)
                    test_generated_orbs.append(test_orb_filepath)
                    test_success_count += 1
            
            # 6. Compress physical .Orb files into standard ZIP archives (saved directly to output root)
            if all_generated_orbs:
                all_zip_filename = f"all_predicted_Orb_files_{train_size}_{val_size}_{test_size}_{basis_set}.zip"
                # [Key modification]: Replace all_out_path with the outer output_all_dir
                all_zip_filepath = os.path.join(output_all_dir, all_zip_filename)
                
                with zipfile.ZipFile(all_zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for orb_path in all_generated_orbs:
                        zipf.write(orb_path, arcname=os.path.basename(orb_path))
                print(f"   ├─ 📦 Successfully packed {len(all_generated_orbs)} .Orb files to outer directory: {all_zip_filepath}")

            if test_generated_orbs:
                test_zip_filename = f"own_testset_predicted_Orb_files_{train_size}_{val_size}_{test_size}_{basis_set}.zip"
                # [Key modification]: Replace test_out_path with the outer output_test_dir
                test_zip_filepath = os.path.join(output_test_dir, test_zip_filename)
                
                with zipfile.ZipFile(test_zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for orb_path in test_generated_orbs:
                        zipf.write(orb_path, arcname=os.path.basename(orb_path))
                print(f"   ├─ 📦 Successfully packed {len(test_generated_orbs)} test set .Orb files to outer directory: {test_zip_filepath}")
            
        except Exception as e:
            print(f"   ❌ Unexpected error processing file {filename}: {e}\n")
        finally:
            if 'npz_data' in locals():
                npz_data.close()

    print("🎉 All tasks completed!")

if __name__ == "__main__":
    batch_process_npz_to_orb()
