# Molcas Benchmark Distributed Computing Script

## Overview
This script is designed for High-Performance Computing (HPC) environments to evaluate and compare the computational performance (specifically, wall-time) between Complete Active Space Configuration Interaction (CASCI) and Complete Active Space Self-Consistent Field (CASSCF) calculations. Utilizing OpenMolcas, the script efficiently distributes benchmark tasks across multiple cluster nodes via PBS array jobs, preventing overlapping workloads and maximizing resource utilization.

## Details of the Computation
- **Thread Safety & Extraction:** To mitigate I/O bottlenecks and race conditions during simultaneous node initialization, Node 1 acts as the master node responsible for extracting `.zip` archives containing the predicted `.Orb` files. All other nodes hold their execution until a synchronization flag (`EXTRACT_DONE`) is generated.
- **Deterministic Task Allocation:** The script scans available predicted orbitals and matches them with structural data (`.xyz`) and reference orbitals (`.RasOrb`) from the dataset directory. The assembled list of valid tasks is stringently sorted by basis set and index, establishing absolute determinism across the cluster. Each PBS node then processes a unique workload dictated by its `PBS_ARRAY_INDEX`.
- **Environment & Execution:** `pymolcas` is invoked in a dedicated sandboxed working directory (`MOLCAS_WORKDIR`) allocated with 16 GB of memory and 4 CPU threads.
- **Decentralized Data Logging:** Upon completing a CASCI/CASSCF pair, the script writes its timing data into a unique, fragmented `.txt` log file to bypass concurrent write access errors.

## Usage
### Directory Structure and Requirements
The script expects the following structural hierarchy:
*   **Input Predicted Orbitals:** `_own_testset_predicted_Orb_files/` — Must reside in the same directory as the script. Contains `.zip` files of predicted `.Orb` archives.
*   **Input Dataset Mapping:** `../../dataset/open/` — Used to fetch `.xyz` geometry structures and `.RasOrb` initial guess files.
*   **Input Templates:** `template_{basis}_CASCI.input` and `template_{basis}_CASSCF.input` — Must reside in the script's root directory.
*   **Outputs:** Calculations are isolated within newly created `Benchmark_Output/` subdirectories.

### Output Files
*   **`Benchmark_Output/CASCI_{basis}_{index}/`**: Directory containing CASCI inputs, symlinks, and `molcas.log`.
*   **`Benchmark_Output/CASSCF_{basis}_{index}/`**: Directory containing CASSCF inputs, symlinks, and `molcas.log`.
*   **`Benchmark_Output/record_{basis}_{index}.txt`**: A comma-separated log containing `Basis, Index, CASCI Time Range, CASCI Total Time, CASSCF Time Range, CASSCF Total Time`.

## Example
To launch the benchmark on an HPC cluster managed by PBS/Torque, submit the script as an array job:
```bash
qsub -J 1-40 script_name.py
```
For local testing or execution on a single target node without a queue scheduler, manually pass the array index:
```bash
export PBS_ARRAY_INDEX=1
python script_name.py
```

## Conclusions
The deployment of this distributed benchmark framework facilitates a large-scale, automated analysis of computational bottlenecks in multireference methods. By directly logging wall-time statistics, the outputs can be easily aggregated to conclusively validate whether utilizing machine learning-predicted orbitals (CASCI) significantly reduces computational overhead compared to fully self-consistent orbital optimization (CASSCF), scaling across varying molecular complexities and basis sets.
