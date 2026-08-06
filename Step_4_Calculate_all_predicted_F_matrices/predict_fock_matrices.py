import os
import glob
import torch
import numpy as np
from ase.io import read
from ase.data import chemical_symbols

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Import required modules from the phisnet_fork library
from phisnet_fork.nn.neural_network import NeuralNetwork as CorePhiSNet
from phisnet_fork.utils.transform_hamiltonians import transform_hamiltonians_from_lm_to_ao

# ==========================================
# Global configurations
# ==========================================
# Top-level directory containing the .xyz files
DATA_DIR = "open"

# Locate all xyz files
# Match pattern: open/open_g2m05_XXXX/open_g2m05_XXXX.xyz
XYZ_PATTERN = os.path.join(DATA_DIR, "open_g2m05_*", "open_g2m05_*.xyz")

def get_basis_config(basis_mode):
    """Return the corresponding configuration parameters based on the basis set."""
    if basis_mode == "MB":
        orbital_map = {1: [(1, 0)], 6: [(6, 0), (6, 0), (6, 1)], 7: [(7, 0), (7, 0), (7, 1)], 8: [(8, 0), (8, 0), (8, 1)]}
        L_max = 1
        convention = "molcas_ANO-MB"
    elif basis_mode == "VDZP":
        orbital_map = {1: [(1, 0), (1, 0), (1, 1)], 
                       6: [(6, 0), (6, 0), (6, 0), (6, 1), (6, 1), (6, 2)], 
                       7: [(7, 0), (7, 0), (7, 0), (7, 1), (7, 1), (7, 2)],
                       8: [(8, 0), (8, 0), (8, 0), (8, 1), (8, 1), (8, 2)]} # Include mapping for oxygen atom
        L_max = 2
        convention = "molcas_ANO-VDZP"
    else:
        raise ValueError(f"Unknown basis set mode: {basis_mode}")
    
    return orbital_map, L_max, convention

def build_model_and_load_weights(basis_mode, ckpt_path, sample_Z, device):
    """Construct an empty model architecture and inject checkpoint weights."""
    orbital_map, L_max, _ = get_basis_config(basis_mode)
    model_order = 2 * L_max
    
    # Construct orbitals based on the atomic numbers (Z) of the input atoms
    molecule_orbitals = [orbital_map[z] for z in sample_Z]
    
    # 1. Instantiate the empty model
    core_model = CorePhiSNet(
        orbitals=molecule_orbitals, 
        num_features=128,
        order=model_order,         
        num_modules=5 
    ).double()
    
    # Disable unnecessary computations to accelerate inference
    core_model.calculate_full_hamiltonian = True   
    core_model.calculate_core_hamiltonian = False
    core_model.calculate_overlap_matrix   = False
    core_model.calculate_energy           = False
    core_model.calculate_forces           = False
    core_model.create_graph               = False

    # 2. Extract bare model weights from the PyTorch Lightning checkpoint
    print(f"[{basis_mode}] Loading weights: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # PyTorch Lightning adds a "model." prefix to parameters; remove it to match core_model
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
    
    # 3. Inject weights and switch to evaluation mode
    core_model.load_state_dict(new_state_dict)
    core_model.to(device)
    core_model.eval()
    
    return core_model

# ==========================================
# NEW: Reverse reordering function to restore PhisNet ordering to the original Molcas output format
# ==========================================
def postprocess_to_molcas_ano_vdzp(F_phisnet, atoms):
    """
    Reverse the preprocess_molcas_ano_vdzp operation from ao_preprocessors.py.
    Restores the predicted matrix to the Molcas format using the mathematical inverse of the original index mapping.
    """
    inv_reorder_idx = []
    current_offset = 0
    for atom in atoms:
        if atom in ['C', 'N', 'O']:
            # Original forward mapping: [0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13]
            # The corresponding inverse mapping is derived as follows:
            idx = [0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13]
            inv_reorder_idx.extend([i + current_offset for i in idx])
            current_offset += 14
        elif atom == 'H':
            # H is originally [0, 1, 2, 3, 4]; inverse mapping remains unchanged
            idx = [0, 1, 2, 3, 4]
            inv_reorder_idx.extend([i + current_offset for i in idx])
            current_offset += 5
        else:
            raise ValueError(f"Reordering rules for atom {atom} are not defined yet!")
            
    # Apply reverse reordering to both rows and columns simultaneously using NumPy fancy indexing
    F_molcas = F_phisnet[inv_reorder_idx, :]
    F_molcas = F_molcas[:, inv_reorder_idx]
    return F_molcas

# ==========================================
# Modified main logic for run_prediction
# ==========================================
def run_prediction(basis_mode, ckpt_path, output_npz_name):
    print(f"\n{'='*50}\nStarting prediction pipeline for {basis_mode} basis set\n{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Collect and sort all XYZ files
    xyz_files = glob.glob(XYZ_PATTERN)
    if not xyz_files:
        print(f"❌ No .xyz files found at {XYZ_PATTERN}! Please check the directory structure.")
        return
    
    # Sort by the XXXX suffix in filenames to ensure consistent processing order
    xyz_files.sort()
    
    # 2. Read the first file to determine atomic composition (PhiSNet requires a fixed structure)
    first_atoms = read(xyz_files[0])
    sample_Z = first_atoms.numbers
    atomic_symbols = [chemical_symbols[z] for z in sample_Z]
    
    _, _, convention = get_basis_config(basis_mode)
    
    # 3. Construct the model and load weights for the corresponding basis set
    model = build_model_and_load_weights(basis_mode, ckpt_path, sample_Z, device)
    
    # Dictionary to store matrices in PhisNet ordering
    predictions_dict = {}
    # [NEW] Dictionary to store matrices in original Molcas ordering
    molcas_predictions_dict = {}
    
    print(f"[{basis_mode}] Found {len(xyz_files)} XYZ files. Starting prediction...")
    
    # 4. Iterate over all files for prediction
    with torch.no_grad():
        for i, file_path in enumerate(xyz_files):
            # Extract the 4-digit index XXXX (from filename open_g2m05_XXXX.xyz)
            filename = os.path.basename(file_path)
            idx_str = filename.split('_')[-1].split('.')[0]
            
            # Print progress (every 100 iterations, or the first/last)
            if i % 100 == 0 or i == len(xyz_files) - 1:
                print(f"[{basis_mode}] Processing: {i+1}/{len(xyz_files)} -> Index: {idx_str}")
            
            # Read molecule
            atoms = read(file_path)
            
            # Verify that atomic composition matches the first file (required for PhiSNet's static computation graph)
            if not np.array_equal(atoms.numbers, sample_Z):
                print(f"⚠️ Warning: Atomic composition of {filename} differs from the initial molecule. Skipping!")
                continue
            
            R_tensor = torch.tensor(atoms.positions, dtype=torch.float64, device=device).unsqueeze(0)
            
            # Predict the Fock matrix in LM format
            pred = model(R_tensor)
            pred_F_lm = pred['full_hamiltonian'].cpu().numpy()
            
            # Transform from LM to AO basis format
            pred_F_ao = transform_hamiltonians_from_lm_to_ao(pred_F_lm, atomic_symbols, convention)
            
            # Extract the first matrix (M, M) from the batch and symmetrize
            pred_F_ao_single = pred_F_ao[0]
            pred_F_ao_single = 0.5 * (pred_F_ao_single + pred_F_ao_single.T)
            
            # Store in standard dictionary
            predictions_dict[idx_str] = pred_F_ao_single

            # [NEW LOGIC]: If in VDZP mode, perform reverse reordering and store in the Molcas-specific dictionary
            if basis_mode == "VDZP":
                pred_F_molcas = postprocess_to_molcas_ano_vdzp(pred_F_ao_single, atomic_symbols)
                molcas_predictions_dict[idx_str] = pred_F_molcas

    # 5. Save the standard .npz file
    print(f"[{basis_mode}] Compressing and writing to standard format {output_npz_name} ...")
    np.savez_compressed(output_npz_name, **predictions_dict)
    print(f"✅ {basis_mode} standard predictions completed! File saved to: {output_npz_name}")

    # [NEW LOGIC]: Separately compress and save an .npz file with Molcas orbital ordering
    if basis_mode == "VDZP":
        molcas_output_npz_name = output_npz_name.replace("all_predicted_F_matrices_", "all_molcas_predicted_F_matrices_")
        print(f"[{basis_mode}] Compressing and writing to Molcas-specific format {molcas_output_npz_name} ...")
        np.savez_compressed(molcas_output_npz_name, **molcas_predictions_dict)
        print(f"✅ {basis_mode} Molcas format predictions completed! File saved to: {molcas_output_npz_name}")


if __name__ == "__main__":
    # --- Task 1: Iterate and process all models for the MB basis set ---
    mb_matches = glob.glob("phisnet_best_model_*_MB.ckpt")
    if mb_matches:
        print(f"🔍 Found {len(mb_matches)} weight files for MB basis set. Preparing for batch prediction...")
        for ckpt_MB in mb_matches:
            # Extract filename
            filename = os.path.basename(ckpt_MB)
            
            # Extract size identifier like 8_1_1 (Train_Val_Test split)
            size_str = "_".join(filename.split('_')[3:6])  
            
            # Dynamically construct output filename, e.g., all_predicted_F_matrices_8_1_1_MB.npz
            out_MB = f"all_predicted_F_matrices_{size_str}_MB.npz"
            
            print(f"\n▶ Starting prediction using model {filename}...")
            run_prediction(basis_mode="MB", ckpt_path=ckpt_MB, output_npz_name=out_MB)
    else:
        print("⚠️ No files matching phisnet_best_model_XXXX_MB.ckpt found in the current directory. Skipping MB prediction.")

    # --- Task 2: Iterate and process all models for the VDZP basis set ---
    vdzp_matches = glob.glob("phisnet_best_model_*_VDZP.ckpt")
    if vdzp_matches:
        print(f"\n🔍 Found {len(vdzp_matches)} weight files for VDZP basis set. Preparing for batch prediction...")
        for ckpt_VDZP in vdzp_matches:
            # Extract filename
            filename = os.path.basename(ckpt_VDZP)
            
            # Extract size identifier like 8_1_1 (Train_Val_Test split)
            size_str = "_".join(filename.split('_')[3:6])
            
            # Dynamically construct output filename, e.g., all_predicted_F_matrices_8_1_1_VDZP.npz
            out_VDZP = f"all_predicted_F_matrices_{size_str}_VDZP.npz"
            
            print(f"\n▶ Starting prediction using model {filename}...")
            run_prediction(basis_mode="VDZP", ckpt_path=ckpt_VDZP, output_npz_name=out_VDZP)
    else:
        print("\n⚠️ No files matching phisnet_best_model_XXXX_VDZP.ckpt found in the current directory. Skipping VDZP prediction.")
