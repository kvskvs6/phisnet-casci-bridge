# Molecular Orbital Coefficient (C Matrix) Evaluator

## Overview
This script evaluates and extracts prediction errors for molecular orbital coefficient (C) matrices. It systematically solves the generalized eigenvalue problem (Roothaan equations) using predicted Fock (F) matrices, aligns the derived orbitals against reference true C matrices using overlap maximization, and performs spatial error analysis to assess electronic structure model performance.

## Details of the Computation
- **Eigenvalue Resolution**: Computes unaligned predicted MO coefficients ($C_{raw}$) by solving $F_{pred} C = S_{true} C \epsilon$ utilizing `scipy.linalg.eigh`.
- **Orbital Alignment (Maximum Overlap Method)**: Resolves the arbitrary phase and gauge inconsistencies inherent in eigenvalue solutions. It calculates the absolute overlap matrix $|O| = |C_{raw}^T S_{true} C_{ref}|$ and applies the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) to perfectly map predicted orbitals to references. Phase matching is subsequently applied based on the sign of the overlap diagonal.
- **Spatial Error Partitioning**: Evaluates Mean Squared Error (MSE) independently across inactive (indices 0-15), active (indices 16-20), and virtual (indices 21-end) molecular orbital spaces to track subsystem-specific physical accuracy.

## Usage
**Input Dependencies:**
- `input_all_molcas_predicted_F_matrices/` or `input_all_predicted_F_matrices/` (Predicted F arrays `.npz`)
- `input_all_true_C_matrices/` (Reference C arrays `.npz`)
- `input_all_true_F_matrices/` (Reference F arrays `.npz`)
- `input_all_S_matrices/` (Overlap matrices `.npz`)
- `input_used_data_record/` (Data split records for parsing fixed test sets `.txt`)

**Output Generation:**
- `output_all_predicted_C_matrices/`: Stores aligned predicted C matrices (`.npz`).
- `output_C_comparison/`: Heatmap visual comparisons (Reference vs. Predicted C matrices + Error residuals) for extreme performance cases.
- `output_O_matrix_comparison/`: Side-by-side mean absolute overlap matrices (Before vs. After alignment).
- `output_all_C_MSE/`, `output_all_C_MSE_separate/`, and `output_all_F_MSE/`: Dedicated CSV/txt reports of spatial full and partitioned MSE distributions.

## Example
Execute the standalone evaluation script using Python:
```bash
python evaluate_c_matrix.py
```

## Conclusions
Running this script thoroughly maps the physical robustness of ML-predicted Hamiltonian matrices. By projecting F-matrix predictions into orbital space and performing localized MSE audits, researchers can pinpoint whether prediction bottlenecks stem from core (inactive), valence (active), or high-energy virtual orbital representations, thereby steering further model generalization enhancements across heterogeneous datasets.
