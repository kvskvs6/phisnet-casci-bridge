#!/usr/bin/env python3

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=04:00:00
#PBS -J 1-400

"""
Molcas Batch Calculation v0.2 - Automated orbital/structure discovery and single-point energy calculation
=========================================================================================================
Directory structure assumption:
rds_root/
  dataset/open/open_g2m05_XXXX/          # Structure directory
              open_g2m05_XXXX.xyz
  OpenMolcas_projects/
    _own_testset_predicted_Orb_files/    # Location of zipped orbital files
      own_testset_predicted_Orb_files_train_valid_test_basis.zip
    (or already extracted to ./extracted/)
    20260610_Energy/                     # Directory of this script
      template_VDZP.input
      template_MB.input
      molcas_batch_calc_v0.1.py
      Output/                            # Computation output directory
        basis_train_valid_test_XXXX/
          molcas.input
          structure.xyz -> symlink
          orbitals.Orb  -> symlink
"""

import os
import sys
import subprocess
import glob
import zipfile
import re
import shutil


# ---------- Automatic Path and Parameter Derivation ----------
def get_rds_root(pbs_workdir):
    """Derive the rds root directory from the script location"""
    # Script is located in rds_root/OpenMolcas_projects/20260610_Energy/
    return os.path.abspath(os.path.join(pbs_workdir, '..', '..'))

def collect_structure_indices(dataset_dir):
    """Return a set of existing structure indices (integers) in dataset/open/"""
    indices = set()
    pattern = os.path.join(dataset_dir, "open_g2m05_*")
    for d in glob.glob(pattern):
        if os.path.isdir(d):
            try:
                idx_str = os.path.basename(d).split('_')[-1]
                idx = int(idx_str)
                indices.add(idx)
            except ValueError:
                continue
    return indices

def extract_zips(orb_zip_dir, extract_root):
    """
    Extract all matching zip files in orb_zip_dir to subdirectories under extract_root.
    Returns the top-level directory where the extracted .Orb files are located.
    Assumes all extracted .Orb files are placed under extract_root/<zipname_without_ext>/.
    """
    zip_pattern = os.path.join(orb_zip_dir, "own_testset_predicted_Orb_files_*_*_*_*.zip")
    for zip_path in glob.glob(zip_pattern):
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        dest = os.path.join(extract_root, zip_name)
        if not os.path.exists(dest):
            os.makedirs(dest, exist_ok=True)
            print(f"Extracting {zip_path} -> {dest}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    if member.endswith('.Orb'):
                        zf.extract(member, dest)
    return extract_root

def find_orb_files(search_dir):
    """
    Recursively search for all .Orb files in search_dir and return a list of absolute paths.
    """
    orb_files = []
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f.endswith('.Orb'):
                orb_files.append(os.path.join(root, f))
    return orb_files

def parse_orb_filename(orb_path):
    """
    Parse (basis, train, valid, test, index) from the filename.
    Filename format: all_predicted_Orb_files_{train}_{valid}_{test}_{basis}_{index}.Orb
    Returns a tuple (basis, train, valid, test, index), or None if parsing fails.
    """
    basename = os.path.basename(orb_path)
    # Use regex to allow the basis to be any string without underscores
    m = re.match(r"all_predicted_Orb_files_(\d+)_(\d+)_(\d+)_(\w+)_(\d+)\.Orb$", basename)
    if m:
        train = int(m.group(1))
        valid = int(m.group(2))
        test  = int(m.group(3))
        basis = m.group(4)
        index = int(m.group(5))
        return (basis, train, valid, test, index)
    return None

# ---------- Computational Environment Setup ----------
def setup_calc_dir(output_dir, structure_index, dataset_dir, orb_file_path, template_input):
    """
    Prepare computational files in output_dir:
      - Copy template to molcas.input
      - Create symlink for structure.xyz
      - Create symlink for orbitals.Orb
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy input template
    input_file = os.path.join(output_dir, "molcas.input")
    shutil.copyfile(template_input, input_file)
    
    # Create symlink for the structure file
    xyz_source = os.path.join(dataset_dir, f"open_g2m05_{structure_index:04d}",
                              f"open_g2m05_{structure_index:04d}.xyz")
    if not os.path.exists(xyz_source):
        raise FileNotFoundError(f"Missing structure file: {xyz_source}")
    link_xyz = os.path.join(output_dir, "structure.xyz")
    if os.path.exists(link_xyz):
        os.remove(link_xyz)
    os.symlink(xyz_source, link_xyz)
    
    # Create symlink for the orbital file
    if not os.path.exists(orb_file_path):
        raise FileNotFoundError(f"Missing orbital file: {orb_file_path}")
    link_orb = os.path.join(output_dir, "orbitals.Orb")
    if os.path.exists(link_orb):
        os.remove(link_orb)
    os.symlink(orb_file_path, link_orb)

def run_calculation(calc_dir, input_file_name='molcas.input'):
    """Run Molcas calculation in calc_dir"""
    pwd = os.getcwd()
    os.chdir(calc_dir)

    # 👇 Add these two lines to keep all files in the current directory
    os.environ['MOLCAS_WORKDIR'] = os.getcwd()      # The working directory is the output folder
    os.environ['MOLCAS_KEEP_WORKDIR'] = 'YES'       # Keep files after calculation
    
    # Molcas environment variables
    os.environ['MOLCAS_MEM'] = '16000'
    os.environ['MOLCAS_NPROCS'] = '4'
    os.environ['MOLCAS_THREADS'] = '4'
    os.environ['MOLCAS_OUTPUT'] = os.getcwd()

    # Load the module and run pymolcas (executed via shell)
    # Note: On a cluster, the module command might require environment initialization
    cmd = f'module load OpenMolcas && pymolcas -f {input_file_name}'
    print(f"Executing command: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error: Molcas returned a non-zero exit code {ret}", file=sys.stderr)
        sys.exit(1)

    os.chdir(pwd)

# ---------- Main Program ----------
if __name__ == '__main__':
    # 1. Get the job submission directory
    pbs_workdir = os.environ.get('PBS_O_WORKDIR', os.getcwd())
    os.chdir(pbs_workdir)
    print(f"Working directory: {pbs_workdir}")
    
    # 2. Derive the rds root directory
    rds_root = get_rds_root(pbs_workdir)
    print(f"RDS root directory: {rds_root}")
    
    # 3. Collect the set of structure indices
    dataset_dir = os.path.join(rds_root, 'dataset', 'open')
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    structure_set = collect_structure_indices(dataset_dir)
    print(f"Available structures: {len(structure_set)}")
    
    # 4. Process orbital files: extract (if needed) and collect all .Orb paths
    orb_zip_dir = os.path.join(rds_root, 'OpenMolcas_projects',
                               '_own_testset_predicted_Orb_files')
    extract_root = os.path.join(orb_zip_dir, 'extracted')
    
    # If the 'extracted' directory does not exist or is empty, extract all zip files
    if not os.path.exists(extract_root) or not os.listdir(extract_root):
        print("Extracting orbital files...")
        extract_zips(orb_zip_dir, extract_root)
    else:
        print("Orbital files are already extracted.")
    
    orb_files = find_orb_files(extract_root)
    print(f"Found .Orb files: {len(orb_files)}")
    
    # 5. Parse and construct the valid task list
    tasks = []  # Elements: (basis, train, valid, test, index, orb_path)
    for orb_path in orb_files:
        parsed = parse_orb_filename(orb_path)
        if not parsed:
            continue
        basis, train, valid, test, index = parsed
        if index in structure_set:
            tasks.append((basis, train, valid, test, index, orb_path))
    
    if not tasks:
        print("Error: No valid tasks found (orbital files matching structures)", file=sys.stderr)
        sys.exit(1)
    
    print(f"Total valid tasks: {len(tasks)}")
    
    # 6. Select the current task based on the PBS array index
    pbs_index_str = os.environ.get('PBS_ARRAY_INDEX')
    if pbs_index_str is None:
        print("Error: PBS_ARRAY_INDEX is not set. Please run as an array job.", file=sys.stderr)
        sys.exit(1)
    
    job_number = int(pbs_index_str) - 1  # Convert to 0-based index
    if job_number < 0 or job_number >= len(tasks):
        # Out of bounds, exit cleanly (common when the array limit is set too high)
        print(f"Array index {pbs_index_str} exceeds the valid task range (1-{len(tasks)}), exiting.")
        sys.exit(0)
    
    basis, train, valid, test, index, orb_path = tasks[job_number]
    print(f"\nCurrent task: Basis={basis}, Train={train}, Valid={valid}, Test={test}, Index={index:04d}")
    
    # 7. The template file must exist
    template_name = f"template_{basis}.input"
    template_path = os.path.join(pbs_workdir, template_name)
    if not os.path.exists(template_path):
        print(f"Error: Missing template file {template_path}", file=sys.stderr)
        sys.exit(1)
    
    # 8. Output directory
    output_root = os.path.join(pbs_workdir, 'Output')
    output_subdir = f"{basis}_{train}_{valid}_{test}_{index:04d}"
    output_dir = os.path.join(output_root, output_subdir)
    print(f"Output directory: {output_dir}")
    
    # 9. Prepare the computation directory and run
    setup_calc_dir(output_dir, index, dataset_dir, orb_path, template_path)
    run_calculation(output_dir)
    
    print("Calculation completed.")
