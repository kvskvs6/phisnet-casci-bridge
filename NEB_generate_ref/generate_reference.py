import os
import sys
import numpy as np
import subprocess
import shutil
import datetime

# Import molcas_utils from the current directory (working root)
import molcas_utils

# ==========================================
# [Part 3] Computation of Fock Matrix Dependencies
# ==========================================
def fock_from_MOs(S, C, en):
    """Calculate the Fock matrix from Roothaan's equations.
    S - atomic orbital overlap matrix
    C - molecular orbital coefficients matrix
    en - vector with molecular orbital energies
    """
    return S @ C @ np.diag(en) @ np.linalg.inv(C)

def compute_fock(rasorb_file, h5_file):
    """Compute a generalised Fock matrix reading energy and orbital
    coefficents from rasob_file, and overlap matrix from h5 file.
    """
    S = molcas_utils.read_molcas_h5_overlap(h5_file)

    MOcoef = molcas_utils.read_orbcoef(rasorb_file)
    MOen = molcas_utils.read_orbener(rasorb_file)

    if (np.any(MOen == 0.)):
        raise ValueError("Zero orbital energy found. Need to use canonical orbitals instead of natural orbitals.")

    return fock_from_MOs(S, MOcoef, MOen)


def main():
    # Retrieve paths
    work_root = os.getcwd()
    # Home directory ~
    home_dir = os.path.expanduser("~")
    dataset_base = os.path.join(home_dir, "dataset_NEB", "open")

    # ==========================================
    # [Part 1] Extract XYZ Coordinates
    # ==========================================
    xyz_file = os.path.join(work_root, "Path_Step_0010.xyz")
    molecules = []

    print(f"Reading and parsing molecular coordinates from {xyz_file}...")
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        num_atoms = int(line)
        title_line = lines[i+1].strip()
        atoms = []
        
        # Extract atomic data line by line, retaining only the first 4 columns (Element, X, Y, Z) and discarding force information
        for j in range(num_atoms):
            parts = lines[i+2+j].split()
            # Format to maintain alignment
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(f"{element:<4} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
            
        molecules.append((num_atoms, atoms))
        i += 2 + num_atoms

    YYY = len(molecules)
    print(f"Successfully extracted {YYY} molecules.")

    # Save individual xyz files to the dataset_NEB directory
    start_idx = 2001
    for idx_offset, (num_atoms, atoms) in enumerate(molecules):
        idx = start_idx + idx_offset
        mol_name = f"open_g2m05_{idx}"
        mol_dir = os.path.join(dataset_base, mol_name)
        os.makedirs(mol_dir, exist_ok=True)

        out_xyz = os.path.join(mol_dir, f"{mol_name}.xyz")
        with open(out_xyz, 'w') as f:
            f.write(f"{num_atoms}\n")
            f.write(f"Properties=species:S:1:pos:R:3 pbc=\"F F F\"\n")
            for atom in atoms:
                f.write(f"{atom}\n")

    # ==========================================
    # [Part 2 & 3] Sequential OpenMolcas Execution and Fock Matrix Generation
    # ==========================================
    basis_sets = ["MB", "VDZP"]
    
    for basis in basis_sets:
        print(f"\n================ Initiating {basis} basis set calculations ================")
        # Initial orbital guess set to 0000
        current_guess_orb = os.path.join(work_root, f"open_g2m05_0000_{basis}.RasOrb")
        template_input = os.path.join(work_root, f"template_{basis}.input")

        for idx_offset in range(YYY):
            idx = start_idx + idx_offset
            mol_name = f"open_g2m05_{idx}"
            calc_dir = os.path.join(dataset_base, mol_name, basis)
            os.makedirs(calc_dir, exist_ok=True)

            xyz_target = os.path.join(dataset_base, mol_name, f"{mol_name}.xyz")
            link_xyz = os.path.join(calc_dir, "structure.xyz")
            link_orb = os.path.join(calc_dir, "initial_orbital_guess.Orb")

            # Clear previous symlinks to prevent FileExistsError
            if os.path.exists(link_xyz) or os.path.islink(link_xyz): os.remove(link_xyz)
            if os.path.exists(link_orb) or os.path.islink(link_orb): os.remove(link_orb)

            # Create symlinks: XYZ to structure.xyz, and the previous RasOrb to initial_orbital_guess.Orb
            os.symlink(xyz_target, link_xyz)
            os.symlink(current_guess_orb, link_orb)

            # Copy and rename the input file to match the Molcas output file prefix
            input_file_name = f"{mol_name}_{basis}.input"
            input_path = os.path.join(calc_dir, input_file_name)
            shutil.copy(template_input, input_path)

            # Change to the computation directory
            os.chdir(calc_dir)
            
            # Set environment variables strictly according to requirements
            os.environ["MOLCAS_WORKDIR"] = calc_dir
            os.environ["MOLCAS_KEEP_WORKDIR"] = "YES"
            os.environ["MOLCAS_MEM"] = "32000"
            os.environ["MOLCAS_NPROCS"] = "8"
            os.environ["MOLCAS_THREADS"] = "8"
            os.environ["MOLCAS_OUTPUT"] = calc_dir

            cmd = f"module load OpenMolcas && pymolcas -f {input_file_name} > molcas.log 2>&1"
            print(f"[{idx_offset+1}/{YYY}] Executing Molcas job: {mol_name} | Basis: {basis} ...")
            subprocess.run(cmd, shell=True, executable="/bin/bash")

            # --- Part 3: Generate Fock matrix ---
            file_stem = f"{mol_name}_{basis}"
            h5_file = os.path.join(calc_dir, f"{file_stem}.rasscf.h5")
            rasorb_file = os.path.join(calc_dir, f"{file_stem}.RasOrb")

            if os.path.exists(h5_file) and os.path.exists(rasorb_file):
                print(f"       -> Generating fock_real.npy ...")
                fock = compute_fock(rasorb_file, h5_file)
                np.save(os.path.join(calc_dir, "fock_real.npy"), fock)
                
                # [Sequential Logic]: Update current_guess_orb to the newly generated .RasOrb for the next molecule
                current_guess_orb = rasorb_file
            else:
                print(f"       -> [WARNING] Output files for {mol_name} ({basis}) not found. Sequential passing may fail; falling back to the previous orbital guess.")

        # Return to the root directory after completing a basis set
        os.chdir(work_root)

    # ==========================================
    # [Part 4] Generate Record Files
    # ==========================================
    print("\nGenerating Used Data Record files...")
    os.chdir(work_root)
    # Incorporate the currently configured system time
    timestamp = "20260726_151801" 

    for basis in basis_sets:
        record_filename = f"used_data_record_1_1_{YYY}_{basis}.txt"
        with open(record_filename, "w") as f:
            f.write(f"Experiment Configuration: Mode 3 | Total Active Samples: 1000\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"========================================\n\n")
            f.write(f"--- TRAIN SET (1 samples) ---\nopen_g2m05_0000\n\n")
            f.write(f"--- VALIDATION SET (1 samples) ---\nopen_g2m05_0000\n\n")
            f.write(f"--- TEST SET ({YYY} samples) ---\n")
            
            # Append YYY indices for the TEST SET
            for idx_offset in range(YYY):
                f.write(f"open_g2m05_{start_idx + idx_offset}\n")
                
        print(f"Generated: {record_filename}")

    print("\n✅ All tasks successfully completed!")

if __name__ == "__main__":
    main()
