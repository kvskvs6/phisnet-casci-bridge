import os
import glob
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt  # Added: for plotting matrix heatmaps

def parse_test_indices(record_filepath):
    '''
    Parse test set molecule indices from record files like used_data_record_40_5_5_MB.txt.
    '''
    test_indices = set()
    if not os.path.exists(record_filepath):
        print(f"  [Warning] Data split record file not found: {record_filepath}. Test set specific errors cannot be computed.")
        return test_indices
        
    in_test_section = False
    with open(record_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Check if entering the TEST SET section
            if line.startswith("--- TEST SET"):
                in_test_section = True
                continue
            # Check if entering other sections (if any exist after test)
            elif line.startswith("--- ") and in_test_section:
                in_test_section = False
                continue
                
            # Extract molecule index, e.g., "open_g2m05_0770" -> "0770"
            if in_test_section and line.startswith("open_g2m05_"):
                idx = line.split("_")[-1]
                test_indices.add(idx)
                
    return test_indices

def order_orbitals(ref,overlap,target):
    '''Reorder target molecular orbitals according to maximum overlap with ref.
    Overlap matrix is needed because atomic orbital basis is not orthogonal.
    MO coefficients are the columns of both ref and target.'''
    
    Moverlap = ref.T @ overlap @ target
  
    _ , orb_order = linear_sum_assignment(abs(Moverlap),maximize=True)
    
    return target[:,orb_order]


def phase_order_orbitals(ref,overlap,target):
    '''Reorder and match phase of target molecular orbitals according to
    maximum overlap with ref.
    Overlap matrix is needed because atomic orbital basis is not orthogonal.
    MO coefficients are the columns of both ref and target.'''
    
    Moverlap = ref.T @ overlap @ target
  
    _ , orb_order = linear_sum_assignment(abs(Moverlap),maximize=True)
    
    phases_match = np.sign(np.diag(Moverlap[:,orb_order]))
    
    new_target = target[:,orb_order] * phases_match
    
    return new_target

def solve_and_align_C(F_pred, S_true, C_ref):
    '''
    Core mathematical logic: Solve the Roothaan equations and align the C matrix 
    using built-in sorting and phase matching functions.
    '''
    # 1. Solve the generalized eigenvalue problem to obtain the unaligned predicted C_raw
    eigvals, C_raw = eigh(F_pred, S_true)
    
    # 2. Calculate the original absolute overlap matrix before alignment (for output)
    O_raw = C_raw.T @ S_true @ C_ref
    O_abs = np.abs(O_raw)
    
    # 3. Call the built-in function above for orbital sorting and phase correction
    C_final = phase_order_orbitals(C_ref, S_true, C_raw)
    
    # 4. Calculate the final absolute overlap matrix after alignment
    O_final_abs = np.abs(C_final.T @ S_true @ C_ref)
    
    return C_final, O_abs, O_final_abs

def compute_errors(C_pred, C_ref):
    '''
    Compute MSE and MAE metrics.
    '''
    diff = C_pred - C_ref
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    return mse, mae

def plot_c_matrix_comparison(C_ref, C_pred, mol_idx, mse, split_info, basis_set, case_type, out_filename):
    '''
    Plot and save the C matrix comparison (Reference, Predicted, Error).
    '''
    # 1. Error matrix: Predicted (middle) minus True (left)
    error_mat = C_pred - C_ref
    max_err = np.max(np.abs(error_mat))
    
    # === Custom font size settings ===
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    main_title_size = 24   # Overall title font size
    sub_title_size = 20    # Subplot title font size
    tick_label_size = 16   # Axis and colorbar tick font size
    # ========================
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Format main title (add basis_set, convert "400_50_50" to "400/50/50")
    formatted_split = split_info.replace('_', '/')
    case_title = "Best" if case_type == "best" else "Worst"
    main_title = f"{basis_set} {formatted_split} – {case_title} (No. {mol_idx})"
    
    # Dynamically determine the color scale limits for C matrices to be symmetric around 0
    vmax_c = max(np.max(np.abs(C_ref)), np.max(np.abs(C_pred)))
    
    # 1. Plot True C matrix (Reference) - bold font
    im0 = axes[0].imshow(C_ref, cmap='bwr', vmin=-vmax_c, vmax=vmax_c)
    axes[0].set_title('True C Matrix', fontsize=sub_title_size, fontweight='bold')
    cb0 = fig.colorbar(im0, ax=axes[0])
    cb0.ax.tick_params(labelsize=tick_label_size)
    
    # 2. Plot Predicted C matrix - bold font
    im1 = axes[1].imshow(C_pred, cmap='bwr', vmin=-vmax_c, vmax=vmax_c)
    axes[1].set_title(f'Predicted C Matrix\nMSE: {mse:.2e}', fontsize=sub_title_size, fontweight='bold')
    cb1 = fig.colorbar(im1, ax=axes[1])
    cb1.ax.tick_params(labelsize=tick_label_size)
    
    # 3. Plot Error matrix - bold font
    im2 = axes[2].imshow(error_mat, cmap='PiYG', vmin=-max_err, vmax=max_err)
    axes[2].set_title(f'Error (Max: {max_err:.3e})', fontsize=sub_title_size, fontweight='bold')
    cb2 = fig.colorbar(im2, ax=axes[2])
    cb2.ax.tick_params(labelsize=tick_label_size)
    
    # Set tick font size for all subplots
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
        
    # Removed the top main title
    # fig.suptitle(main_title, fontsize=main_title_size, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_average_o_matrices(O_mean_raw, O_mean_aligned, test_count, split_info, basis_set):
    '''
    Plot the comparison of the average O matrix before and after alignment for the test set.
    '''
    # === Custom font size settings ===
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    main_title_size = 24
    sub_title_size = 20
    tick_label_size = 16
    # ========================

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    formatted_split = split_info.replace('_', '/')
    main_title = f"{basis_set} {formatted_split} – Mean MO Overlap (N={test_count})"
    
    # 1. Plot the average O matrix before alignment
    im0 = axes[0].imshow(O_mean_raw, cmap='Blues', vmin=0, vmax=1.0)
    axes[0].set_title('Mean |O| (Before Alignment)', fontsize=sub_title_size, fontweight='bold')
    axes[0].set_xlabel('Reference MO index', fontsize=tick_label_size, fontweight='bold')
    axes[0].set_ylabel('Predicted MO index', fontsize=tick_label_size, fontweight='bold')
    cb0 = fig.colorbar(im0, ax=axes[0])
    cb0.ax.tick_params(labelsize=tick_label_size)
    
    # 2. Plot the average O matrix after alignment
    im1 = axes[1].imshow(O_mean_aligned, cmap='Blues', vmin=0, vmax=1.0)
    axes[1].set_title('Mean |O| (After Alignment)', fontsize=sub_title_size, fontweight='bold')
    axes[1].set_xlabel('Reference MO index', fontsize=tick_label_size, fontweight='bold')
    axes[1].set_ylabel('Predicted MO index', fontsize=tick_label_size, fontweight='bold')
    cb1 = fig.colorbar(im1, ax=axes[1])
    cb1.ax.tick_params(labelsize=tick_label_size)
    
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
        
    # fig.suptitle(main_title, fontsize=main_title_size, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    out_filename = os.path.join("output_O_matrix_comparison", f"O_matrix_comparison_{split_info}_{basis_set}.png")
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def process_single_evaluation(f_file, s_file, c_true_file, f_true_file, record_file, basis_set, split_info, fixed_test_indices):
    '''
    Process the calculation workflow for a single basis set and data split.
    Returns a dictionary containing various evaluation results.
    '''
    # Load data files
    try:
        f_data = np.load(f_file)
        s_data = np.load(s_file)
        c_data = np.load(c_true_file)
        f_true_data = np.load(f_true_file) # Added: load true F matrix data
    except Exception as e:
        print(f"  [Error] Failed to load npz file: {e}")
        return None
        
    # Get common molecule indices
    f_keys = set(f_data.files)
    s_keys = set(s_data.files)
    c_keys = set(c_data.files)
    f_true_keys = set(f_true_data.files) # Added
    common_keys = sorted(list(f_keys & s_keys & c_keys & f_true_keys)) # Ensure all 4 files share the molecule
    
    if not common_keys:
        print(f"  [Skip] No common molecule indices found between predicted data and true S/C files.")
        return None
        
    # Parse test set indices
    test_indices = parse_test_indices(record_file)
    
    c_pred_dict = {}
    all_mse, all_mae = [], []
    test_mse, test_mae = [], []
    fixed_test_mse, fixed_test_mae = [], []  # Added: error lists for the fixed test set
    individual_mse_list = []                  # Added: for sequentially recording (index, MSE) for each molecule
    individual_f_mse_list = []                # Added: for recording F matrix (index, MSE)
    individual_c_mse_separate_list = []       # Added: for recording spatially partitioned C matrix MSE
    
    # --- Variables to track best and worst molecules in the test set ---
    best_test_key = None
    worst_test_key = None
    min_test_mse = float('inf')
    max_test_mse = -1.0
    best_C_ref, best_C_pred = None, None
    worst_C_ref, worst_C_pred = None, None
    
    # --- Added: list to collect test set O matrices for averaging ---
    test_O_raw_list = []
    test_O_aligned_list = []
    # ------------------------------------------------
    
    print(f"  -> Total {len(common_keys)} molecules. Identified {len(test_indices)} test set molecules. Starting computation...")
    
    for key in common_keys:
        F_pred = f_data[key]
        S_true = s_data[key]
        C_ref = c_data[key]
        F_true = f_true_data[key]             # Added: get true F matrix
        
        try:
            # Perform alignment (unpacking 3 return values)
            C_final, O_abs, O_final_abs = solve_and_align_C(F_pred, S_true, C_ref)
            
            # Compute C matrix errors
            mse, mae = compute_errors(C_final, C_ref)
            
            # === Added: compute MSE for specific subspaces ===
            # Inactive space (columns 0-15)
            mse_inactive = np.mean((C_final[:, :16] - C_ref[:, :16]) ** 2)
            # Active space (columns 16-20)
            mse_active = np.mean((C_final[:, 16:21] - C_ref[:, 16:21]) ** 2)
            # Active + Inactive space (columns 0-20)
            mse_in_act = np.mean((C_final[:, :21] - C_ref[:, :21]) ** 2)
            
            # Virtual orbitals (column 21 to end, handle extreme cases with < 21 cols)
            diff_vir = C_final[:, 21:] - C_ref[:, 21:]
            if diff_vir.size > 0:
                mse_virtual = np.mean(diff_vir ** 2)
            else:
                mse_virtual = np.nan
            # ===============================
            
            # Compute F matrix errors (reusing compute_errors function)
            f_mse, f_mae = compute_errors(F_pred, F_true)
            
            # Record data
            c_pred_dict[key] = C_final
            all_mse.append(mse)
            all_mae.append(mae)
            individual_mse_list.append((key, mse))  
            individual_f_mse_list.append((key, f_mse)) # Added: record current molecule F matrix MSE
            individual_c_mse_separate_list.append((key, mse_inactive, mse_active, mse_in_act, mse_virtual)) # Added: record subspace MSE
            
            # If the index is in the test set, record additionally and evaluate extremes
            if key in test_indices:
                test_mse.append(mse)
                test_mae.append(mae)
                
                # Collect O matrices
                test_O_raw_list.append(O_abs)
                test_O_aligned_list.append(O_final_abs)
                
                # Update best and worst cases
                if mse < min_test_mse:
                    min_test_mse = mse
                    best_test_key = key
                    best_C_ref = C_ref.copy()
                    best_C_pred = C_final.copy()
                
                if mse > max_test_mse:
                    max_test_mse = mse
                    worst_test_key = key
                    worst_C_ref = C_ref.copy()
                    worst_C_pred = C_final.copy()
            if key in fixed_test_indices:
                fixed_test_mse.append(mse)
                fixed_test_mae.append(mae)    
            
        except Exception as e:
            print(f"  [Error] Calculation failed for molecule {key}: {e}")
            
    if not all_mse:
        return None
        
    # Save aligned predicted C matrices
    c_out_filename = os.path.join("output_all_predicted_C_matrices", f"all_predicted_C_matrices_{split_info}_{basis_set}.npz")
    np.savez_compressed(c_out_filename, **c_pred_dict)
    
    # === Save C matrix MSE for each molecule into a separate text file ===
    mse_txt_filename = os.path.join("output_all_C_MSE", f"all_C_MSE_{split_info}_{basis_set}.txt")
    with open(mse_txt_filename, "w", encoding="utf-8") as f_mse:
        for m_idx, m_mse in individual_mse_list:
            f_mse.write(f"{m_idx},{m_mse:.8e}\n")
    print(f"  -> Individual C matrix MSE for each molecule saved to: {mse_txt_filename}")
    
    # === Added: Save subregion C matrix MSE for each molecule into a separate text file ===
    c_mse_sep_txt_filename = os.path.join("output_all_C_MSE_separate", f"all_C_MSE_separate_{split_info}_{basis_set}.txt")
    with open(c_mse_sep_txt_filename, "w", encoding="utf-8") as f_mse_sep:
        f_mse_sep.write("MoleculeIndex,Inactive_MSE(0-15),Active_MSE(16-20),Inactive+Active_MSE(0-20),Virtual_MSE(21-end)\n")
        for m_idx, mse_in, mse_act, mse_in_act, mse_vir in individual_c_mse_separate_list:
            f_mse_sep.write(f"{m_idx},{mse_in:.8e},{mse_act:.8e},{mse_in_act:.8e},{mse_vir:.8e}\n")
    print(f"  -> Individual spatial region C matrix MSE saved to: {c_mse_sep_txt_filename}")
    
    # === Added: Save F matrix MSE for each molecule into a separate text file ===
    f_mse_txt_filename = os.path.join("output_all_F_MSE", f"all_F_MSE_{split_info}_{basis_set}.txt")
    with open(f_mse_txt_filename, "w", encoding="utf-8") as file_f_mse:
        for m_idx, m_f_mse in individual_f_mse_list:
            file_f_mse.write(f"{m_idx},{m_f_mse:.8e}\n")
    print(f"  -> Individual F matrix MSE for each molecule saved to: {f_mse_txt_filename}")
    # ===================================================
    
    # --- Call plotting function to output best and worst C matrix images ---
    if best_test_key is not None:
        best_filename = os.path.join("output_C_comparison", f"C_comparison_{split_info}_{basis_set}_best_No_{best_test_key}.png")
        plot_c_matrix_comparison(
            best_C_ref, best_C_pred, best_test_key, min_test_mse, 
            split_info, basis_set, "best", best_filename
        )
        print(f"  -> Saved test set best error plot: {best_filename} (MSE={min_test_mse:.2e})")
        
    if worst_test_key is not None:
        worst_filename = os.path.join("output_C_comparison", f"C_comparison_{split_info}_{basis_set}_worst_No_{worst_test_key}.png")
        plot_c_matrix_comparison(
            worst_C_ref, worst_C_pred, worst_test_key, max_test_mse, 
            split_info, basis_set, "worst", worst_filename
        )
        print(f"  -> Saved test set worst error plot: {worst_filename} (MSE={max_test_mse:.2e})")        
    # --- Added: Compute test set average O matrix and plot ---
    if test_O_raw_list:
        O_mean_raw = np.mean(test_O_raw_list, axis=0)
        O_mean_aligned = np.mean(test_O_aligned_list, axis=0)
        plot_average_o_matrices(O_mean_raw, O_mean_aligned, len(test_O_raw_list), split_info, basis_set)
        print(f"  -> Saved test set average O matrix comparison plot: O_matrix_comparison_{split_info}_{basis_set}.png")
    # ---------------------------------------------------
    
    # Aggregate error results
    result = {
        "Split": split_info,
        "Basis": basis_set,
        "Total_Count": len(all_mse),
        "Total_MSE": np.mean(all_mse),
        "Total_MAE": np.mean(all_mae),
        "Test_Count": len(test_mse) if test_mse else 0,
        "Test_MSE": np.mean(test_mse) if test_mse else np.nan,
        "Test_MAE": np.mean(test_mae) if test_mae else np.nan,
        "Fixed_Test_Count": len(fixed_test_mse) if fixed_test_mse else 0,
        "Fixed_Test_MSE": np.mean(fixed_test_mse) if fixed_test_mse else np.nan,
        "Fixed_Test_MAE": np.mean(fixed_test_mae) if fixed_test_mae else np.nan,
        "Output_File": c_out_filename
    }
    
    return result

def main():
    print("====== Starting C Matrix Prediction Error Evaluation (Enhanced Version) ======")
    
    # 1. Automatically create output directories (if not exist), added all_C_MSE_separate
    output_dirs = [
        "output_all_predicted_C_matrices", 
        "output_C_comparison", 
        "output_O_matrix_comparison", 
        "output_all_C_MSE", 
        "output_all_F_MSE", 
        "output_all_C_MSE_separate"
    ]
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
    
    # === Added: Parse fixed test set (based on 800_100_100 split record) ===
    fixed_record_files = glob.glob(os.path.join("input_used_data_record", "used_data_record_800_100_100_*.txt"))
    fixed_test_indices = set()
    if fixed_record_files:
        # Extract test set from the first matching file
        fixed_record_file = fixed_record_files[0]
        fixed_test_indices = parse_test_indices(fixed_record_file)
        print(f"  [Info] Found fixed test set reference file: {os.path.basename(fixed_record_file)}. Extracted {len(fixed_test_indices)} molecules as Fixed_Test.")
    else:
        print("  [Warning] Record file starting with used_data_record_800_100_100_ not found. Fixed_Test metrics will be empty.")
    # ============================================================

    # 2. Grab predicted F matrix files from the two specified input folders
    # Exact match for files with molcas prefix
    molcas_files = glob.glob(os.path.join("input_all_molcas_predicted_F_matrices", "all_molcas_predicted_F_matrices_*.npz"))
    
    # Exact match for files with normal prefix
    normal_files = glob.glob(os.path.join("input_all_predicted_F_matrices", "all_predicted_F_matrices_*.npz"))
    
    raw_files = molcas_files + normal_files
    f_files = []
    
    # Filtering logic: For VDZP basis set, only keep files with 'molcas' in the name
    for f in raw_files:
        fname = os.path.basename(f)
        
        # If the filename contains VDZP but does not contain molcas, skip it (discard normal VDZP files)
        if "VDZP" in fname and "molcas" not in fname:
            continue
            
        f_files.append(f)

    if not f_files:
        print("No matching predicted F matrix files found. Please ensure the working directory is correct.")
        return
        
    results_list = []
    
    for f_file in sorted(f_files):
        filename = os.path.basename(f_file)
        # Use "_F_matrices_" as absolute anchor to extract accurately whether prefixed with all or all_molcas
        suffix = filename.replace(".npz", "").split("_F_matrices_")[-1]
        parts = suffix.split("_")
        basis_set = parts[-1]
        split_info = "_".join(parts[:-1])
        
        print(f"\nProcessing: {filename}")
        
        s_file = os.path.join("input_all_S_matrices", f"all_S_matrices_{basis_set}.npz")
        c_true_file = os.path.join("input_all_true_C_matrices", f"all_true_C_matrices_{basis_set}.npz")
        f_true_file = os.path.join("input_all_true_F_matrices", f"all_true_F_matrices_{basis_set}.npz") # Added: true F matrix path
        record_file = os.path.join("input_used_data_record", f"used_data_record_{split_info}_{basis_set}.txt")
        
        if not os.path.exists(s_file) or not os.path.exists(c_true_file) or not os.path.exists(f_true_file):
            print(f"  [Skip] Missing true S, C, or F matrix files.")
            continue
            
        # Note that f_true_file parameter is added here
        result = process_single_evaluation(f_file, s_file, c_true_file, f_true_file, record_file, basis_set, split_info, fixed_test_indices)
        
        if result:
            results_list.append(result)
            print(f"  -> Done! Overall MSE: {result['Total_MSE']:.8e} | Test Set MSE: {result['Test_MSE']:.8e}")
            print(f"  -> Saved to: {result['Output_File']}")
            
    # Write results to list/CSV format for easy loading by plotting scripts
    report_filename = "C_matrix_comparison_results.txt"
    if results_list:
        with open(report_filename, "w", encoding="utf-8") as f_out:
            # Write header (comma-separated)
            headers = ["Split", "Basis", "Total_Count", "Total_MSE", "Total_MAE", "Test_Count", "Test_MSE", "Test_MAE", "Fixed_Test_Count", "Fixed_Test_MSE", "Fixed_Test_MAE"]
            f_out.write(",".join(headers) + "\n")
            
            # Write data rows
            for res in results_list:
                row = [
                    res["Split"],
                    res["Basis"],
                    str(res["Total_Count"]),
                    f"{res['Total_MSE']:.8e}",
                    f"{res['Total_MAE']:.8e}",
                    str(res["Test_Count"]),
                    f"{res['Test_MSE']:.8e}" if not np.isnan(res['Test_MSE']) else "NaN",
                    f"{res['Test_MAE']:.8e}" if not np.isnan(res['Test_MAE']) else "NaN",
                    str(res["Fixed_Test_Count"]),
                    f"{res['Fixed_Test_MSE']:.8e}" if not np.isnan(res['Fixed_Test_MSE']) else "NaN",
                    f"{res['Fixed_Test_MAE']:.8e}" if not np.isnan(res['Fixed_Test_MAE']) else "NaN"
                ]
                f_out.write(",".join(row) + "\n")
                
        print(f"\nAll tasks complete! Error list saved to: {report_filename} (CSV format, ready for pandas/matplotlib)")
    else:
        print("\nNo valid results were generated.")

if __name__ == '__main__':
    main()