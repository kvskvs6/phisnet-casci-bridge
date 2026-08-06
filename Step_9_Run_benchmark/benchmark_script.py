#!/usr/bin/env python3

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=12:00:00   
#PBS -J 1-40

"""
Molcas Benchmark Distributed Computing v1.0 - CASCI vs CASSCF Performance Comparison
==============================================================
Implementation requirements:
1. `_own_testset_predicted_Orb_files` must be in the same directory as this script.
2. Predicted orbitals are utilized for CASCI calculations.
3. `.RasOrb` files from the dataset serve as initial orbitals for CASSCF.
4. [New] Using PBS array jobs, each node executes exactly one specific CASCI + CASSCF pair.
5. Index XXXX is matched automatically.
6. [New] A dedicated lightweight `.txt` log file is generated upon completion of each task pair.
7. [Patch] Added extraction logic for `.zip` files.
"""

import os
import sys
import glob
import re
import shutil
import time
import zipfile
from datetime import datetime

# ================= Helper Functions =================
def format_time(seconds):
    """Convert seconds into HH:MM:SS format."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

def parse_orb_filename(basename):
    """Parse predicted orbital filename to return (basis, index)."""
    m = re.match(r"all_predicted_Orb_files_\d+_\d+_\d+_(\w+)_(\d+)\.Orb$", basename)
    if m:
        return m.group(1), int(m.group(2))
    return None, None

def run_molcas(calc_dir, template_path, target_xyz, target_orb):
    """Prepare directory and execute Molcas, returning (start_timestamp, end_timestamp, duration_seconds)."""
    os.makedirs(calc_dir, exist_ok=True)
    input_file = os.path.join(calc_dir, "molcas.input")
    shutil.copyfile(template_path, input_file)
    
    link_xyz = os.path.join(calc_dir, "structure.xyz")
    if os.path.exists(link_xyz): os.remove(link_xyz)
    os.symlink(target_xyz, link_xyz)
    
    link_orb = os.path.join(calc_dir, "orbitals.Orb")
    if os.path.exists(link_orb): os.remove(link_orb)
    os.symlink(target_orb, link_orb)
    
    pwd = os.getcwd()
    os.chdir(calc_dir)
    os.environ['MOLCAS_WORKDIR'] = os.getcwd()
    os.environ['MOLCAS_KEEP_WORKDIR'] = 'YES'
    os.environ['MOLCAS_MEM'] = '16000'
    os.environ['MOLCAS_NPROCS'] = '4'
    os.environ['MOLCAS_THREADS'] = '4'
    os.environ['MOLCAS_OUTPUT'] = os.getcwd()
    
    start_time = time.time()
    cmd = 'module load OpenMolcas && pymolcas -f molcas.input > molcas.log 2>&1'
    os.system(cmd)
    end_time = time.time()
    
    os.chdir(pwd)
    return start_time, end_time, (end_time - start_time)

# ================= Main Program =================
if __name__ == '__main__':
    pbs_workdir = os.environ.get('PBS_O_WORKDIR')
    if pbs_workdir:
        os.chdir(pbs_workdir)
        SCRIPT_DIR = pbs_workdir
    else:
        SCRIPT_DIR = os.getcwd()

    RDS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    DATASET_DIR = os.path.join(RDS_ROOT, 'dataset', 'open')
    PREDICTED_ORB_DIR = os.path.join(SCRIPT_DIR, '_own_testset_predicted_Orb_files')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'Benchmark_Output')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== Distributed Node Benchmark Initiated ===")
    
    # 1. Extraction logic (Patch for missing extraction step)
    extract_dir = os.path.join(PREDICTED_ORB_DIR, 'extracted')
    
    # [Safety Mechanism] To prevent file corruption from race conditions across hundreds of concurrent nodes,
    # node 1 (master) handles extraction while other nodes wait.
    pbs_index_str = os.environ.get('PBS_ARRAY_INDEX')
    if pbs_index_str is None:
        print("❌ Error: PBS_ARRAY_INDEX is not set. Please run as an array job (using #PBS -J).")
        sys.exit(1)
        
    job_number = int(pbs_index_str) - 1  # Convert to 0-based indexing

    if job_number == 0:
        # If this is node 1, execute extraction
        if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            zip_files = glob.glob(os.path.join(PREDICTED_ORB_DIR, "*.zip"))
            for zip_path in zip_files:
                print(f"📦 [Node 1] Extracting: {os.path.basename(zip_path)}")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for member in zf.namelist():
                        if member.endswith('.Orb'):
                            zf.extract(member, extract_dir)
            print("✅ Extraction complete. Generating flag file.")
            with open(os.path.join(extract_dir, "EXTRACT_DONE"), 'w') as f:
                f.write("DONE")
    else:
        # Other nodes wait for extraction to finish
        print(f"⏳ [Node {job_number+1}] Waiting for Node 1 to finish extraction...")
        wait_time = 0
        while not os.path.exists(os.path.join(extract_dir, "EXTRACT_DONE")):
            time.sleep(5)
            wait_time += 5
            if wait_time > 300: # Throw error if wait exceeds 5 minutes
                 print("❌ Error: Timeout waiting for extraction!")
                 sys.exit(1)

    # 2. Scan files to construct a global pool of valid tasks
    all_predicted_orbs = sorted(glob.glob(os.path.join(extract_dir, "*.Orb")))
    valid_tasks = []  
    
    # [New] Define target run counts and initialize counters
    NUM_PAIRS_PER_BASIS = 20
    task_counts = {'MB': 0, 'VDZP': 0}
    
    for orb_path in all_predicted_orbs:
        basename = os.path.basename(orb_path)
        basis, idx = parse_orb_filename(basename)
        
        # [New] Only proceed if the quota for this basis set has not been met
        if basis in task_counts and task_counts[basis] < NUM_PAIRS_PER_BASIS:
            dataset_sub = f"open_g2m05_{idx:04d}"
            casscf_orb_path = os.path.join(DATASET_DIR, dataset_sub, basis, f"{dataset_sub}_{basis}.RasOrb")
            xyz_path = os.path.join(DATASET_DIR, dataset_sub, f"{dataset_sub}.xyz")
            
            if os.path.exists(casscf_orb_path) and os.path.exists(xyz_path):
                valid_tasks.append({
                    'basis': basis, 'index': idx,
                    'casci_orb': orb_path, 'casscf_orb': casscf_orb_path, 'xyz': xyz_path
                })
                # Increment counter after successfully registering a valid task
                task_counts[basis] += 1
    
    # Sort by basis and index to guarantee deterministic task lists across nodes
    valid_tasks = sorted(valid_tasks, key=lambda x: (x['basis'], x['index']))
    
    if not valid_tasks:
        print("❌ Error: No matching task pairs found.")
        sys.exit(1)

    print(f"📊 Found a total of {len(valid_tasks)} valid task pairs.")

    # 3. Core allocation logic: Fetch the unique task assigned to the current node
    if job_number >= len(valid_tasks):
        print(f"✅ Node {job_number+1} assigned task index exceeds total tasks {len(valid_tasks)}. Node exiting (idle).")
        sys.exit(0)

    my_task = valid_tasks[job_number]
    basis = my_task['basis']
    idx = my_task['index']
    
    print(f"\n🎯 [Task Allocation] I am Node {job_number+1}, assigned to: Basis {basis}, Molecule Index {idx:04d}")

    # Verify template existence
    casci_tpl = os.path.join(SCRIPT_DIR, f"template_{basis}_CASCI.input")
    casscf_tpl = os.path.join(SCRIPT_DIR, f"template_{basis}_CASSCF.input")
    if not (os.path.exists(casci_tpl) and os.path.exists(casscf_tpl)):
        print(f"❌ Error: Template file for {basis} not found! Exiting.")
        sys.exit(1)

    # 4. Execute computations (CASCI first, then CASSCF)
    # --- Run CASCI ---
    casci_dir = os.path.join(OUTPUT_DIR, f"CASCI_{basis}_{idx:04d}")
    print("  ├─ [CASCI] Running...")
    start_ci, end_ci, time_ci = run_molcas(casci_dir, casci_tpl, my_task['xyz'], my_task['casci_orb'])
    str_start_ci = datetime.fromtimestamp(start_ci).strftime('%H:%M:%S')
    str_end_ci = datetime.fromtimestamp(end_ci).strftime('%H:%M:%S')
    print(f"  ├─ [CASCI] Completed! Duration: {time_ci:.2f} s")
    
    # --- Run CASSCF ---
    casscf_dir = os.path.join(OUTPUT_DIR, f"CASSCF_{basis}_{idx:04d}")
    print("  ├─ [CASSCF] Running...")
    start_scf, end_scf, time_scf = run_molcas(casscf_dir, casscf_tpl, my_task['xyz'], my_task['casscf_orb'])
    str_start_scf = datetime.fromtimestamp(start_scf).strftime('%H:%M:%S')
    str_end_scf = datetime.fromtimestamp(end_scf).strftime('%H:%M:%S')
    print(f"  └─ [CASSCF] Completed! Duration: {time_scf:.2f} s")

    # 5. [Implementation requirement] Generate a fragmented .txt record containing only this row of data
    record_filename = os.path.join(OUTPUT_DIR, f"record_{basis}_{idx:04d}.txt")
    with open(record_filename, 'w', encoding='utf-8') as f:
        # Comma-separated for easy parsing by future aggregation scripts (Format: Basis, Index, CASCI_Time, CASSCF_Time)
        line = f"{basis},{idx},{str_start_ci}-{str_end_ci},{time_ci:.2f},{str_start_scf}-{str_end_scf},{time_scf:.2f}\n"
        f.write(line)

    print(f"\n📁 Node task successfully completed! Record saved to: {record_filename}")
