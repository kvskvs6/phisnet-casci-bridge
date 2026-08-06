# Wavefunction Preparation for NEB (01_prep_wavefunctions.py)

## Overview
This script handles the interpolation of intermediate structures along a reaction path using bidirectional normal mode coordinates and performs sequential Complete Active Space Self-Consistent Field (CASSCF) wavefunction propagation. It yields stable initial geometries and accurate starting wavefunctions necessary for subsequent Nudged Elastic Band (NEB) transition state searches.

## Details of the Computation
1. **Geometry Interpolation:** The script applies a rigorous bidirectional normal mode interpolation scheme. The structural differences between the open (reactant) and closed (product) geometries are mapped to forward and backward normal mode basis coordinates, respectively (parsed from `.h5` files). These are linearly scaled by a path fraction $\lambda$ and transformed back into Cartesian space.
2. **Wavefunction Propagation:** The script iteratively optimizes the CASSCF(6,5) wavefunction for each of the 32 images. Each image directly inherits the `.chk` file from the preceding step (e.g., `neb_image_01.chk` inherits the electron density from `neb_image_00.chk`). 
3. **Custom Gaussian Parsing:** It extends `ase.calculators.gaussian.Gaussian` to correctly handle multi-root CASSCF outputs, extracting accurate EIGENVALUE state energies and ignoring standard structural charge misreads typical in ASE for complex output types.

## Usage
### Input Files Required
Place the following files in your working directory before execution:
* `open_g2m05_0000.xyz`: Starting geometry (used as-is; skips prior optimization step).
* `open_CASsa_0000.chk`: The reference CASSCF checkpoint file for the first state.
* `closed_g2m05_0000.xyz`: Target geometry (product).
* `DVA_VDZP_0000_freq.slapaf.h5`: Forward normal mode data.
* `PYR_VDZP_0000_freq.slapaf.h5`: Backward normal mode data.
* `molcas_utils.py`: Utility module for handling `.h5` file coordinate transformations.

### Output Files Generated
* `initial_images.traj`: An ASE trajectory containing all 32 coordinates, to be read by the next script.
* `neb_image_*.chk`: Converged CASSCF checkpoint files representing the initial electronic structure for each NEB image.
* `Prep_Logs/`: A directory backing up all Gaussian `.log` outputs from the sequential wavefunction generation steps.

### Example
Ensure your Gaussian 16 environment is active, then execute:
```bash
python 01_prep_wavefunctions.py
```

## Conclusions
By rigorously pre-converging the multi-reference wavefunctions frame-by-frame along the interpolated reaction pathway, this script drastically reduces typical Self-Consistent Field (SCF) convergence failures during subsequent transition state searches. This systematic initialization ensures a smooth, continuous potential energy surface, allowing the NEB algorithm to robustly and efficiently locate the transition state without breaking symmetry or crashing due to poorly guessed electronic states.
