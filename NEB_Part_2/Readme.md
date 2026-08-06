# 02_run_neb.py

## Overview
This script executes the synchronous optimization step of a Nudged Elastic Band (NEB) calculation using the Atomic Simulation Environment (ASE) and Gaussian 16. It is designed to optimize the reaction path and locate the transition state on a CASSCF potential energy surface.

## Details of the Computation
- **Custom Calculator:** Utilizes a customized `GaussianCASSCF` ASE calculator to resolve formatting issues in Gaussian 16 CASSCF outputs. It explicitly parses multi-root configurations (via `nroot`) and properly reads energy eigenvalues, bypassing default SCF parsing errors.
- **Image Setup:** Reads an existing interpolated trajectory (`initial_images.traj`) and constructs the band. 
- **Calculator Attachment:** Binds individual Gaussian CASSCF calculators to each image using pre-converged wavefunctions from corresponding `.chk` files.
- **Optimization:** Employs the `improvedtangent` method for the NEB forces and optimizes the entire band synchronously using the BFGS algorithm. A custom hook (`record_step`) is attached to log the progress of the optimizer locally.

## Usage
**Input Files Required:**
- `initial_images.traj`: Interpolated configurations from the initial guess.
- `neb_image_00.chk` to `neb_image_31.chk`: Checkpoint files containing pre-converged CASSCF wavefunctions.

**Output Files Generated:**
- `S1_NEB_Full_Path.traj`: Trajectory file recording the full evolution of the NEB optimization.
- `neb_progress.txt`: A text file appending the completion status of each NEB iteration step.
- `neb_image_*.com` & `neb_image_*.log`: Gaussian input and output logs for each image at every step.

## Example
Ensure all checkpoint files and the initial trajectory are present in the working directory, then run:
```bash
python 02_run_neb.py
```

## Conclusions
Successfully running this script yields a converged Minimum Energy Path (MEP) and the exact transition state configuration on the target electronic state (e.g., S1). The resulting trajectory file allows for detailed structural and energetic profiling of the photochemical or thermal reaction barrier.
