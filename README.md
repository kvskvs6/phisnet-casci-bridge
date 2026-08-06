# phisnet-casci-bridge

This repository contains the downstream analysis pipeline for the ML-CASCI framework, covering everything after PhiSNet's Fock matrix prediction. It extracts reference S/C/F matrices from CASSCF calculations, computes predicted F matrices, solves the Roothaan equation to obtain C matrices (with Hungarian-algorithm alignment and phase correction), reconstructs molecular orbitals, and calculates CASCI energies. It also benchmarks CASCI against CASSCF computation time, and evaluates model predictions along the 32-image NEB reaction path connecting DVA and PYR, including subspace-overlap error analysis in regions of orbital character change.

## Directory Structure

### Step 1–9: Main Pipeline

| Folder | Description |
|---|---|
| `Step_1_Extract_real_S_matrix` | Extract the reference overlap matrix (S matrix) from CASSCF calculations |
| `Step_2_Extract_real_C_matrix` | Extract the reference coefficient matrix (C matrix) |
| `Step_3_Extract_real_F_matrix` | Extract the reference Fock matrix (F matrix) |
| `Step_4_Calculate_all_predicted_F_matrices` | Compute all predicted F matrices based on PhiSNet's predictions |
| `Step_5_Predict_C_matrix` | Solve the Roothaan equation to obtain the predicted C matrix (Hungarian-algorithm alignment + phase correction) |
| `Step_6_Calculate_orbitals` | Reconstruct molecular orbitals |
| `Step_7_Calculate_energy` | Calculate CASCI energy |
| `Step_8_Get_energy_summary` | Summarize energy results |
| `Step_9_Run_benchmark` | Benchmark CASCI vs. CASSCF computation time |

### NEB Path Analysis

| Folder | Description |
|---|---|
| `NEB_Part_1` / `NEB_Part_2` / `NEB_Part_3` | Evaluate model predictions along the 32-image NEB reaction path (connecting DVA and PYR), including subspace-overlap error analysis in regions of orbital character change, split into three parts |
| `NEB_generate_ref` | Generate NEB reference data |

## Data Notes

Steps 1–9 are all written using **DVA** as the example system, corresponding to the data folder **`open`**.

To study other molecules, simply switch to the corresponding data folder, for example:

- **`closed`** corresponds to **Pyrroline (PYR)**

The pipeline logic stays the same — only the input data folder needs to be changed.
