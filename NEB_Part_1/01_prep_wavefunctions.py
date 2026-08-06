# -------------------------------------------------------------------------
# 01_prep_wavefunctions.py
#
# [Overview]
# Includes normal mode coordinate interpolation and the computationally expensive wavefunction propagation (up to Image 31).
#
# [Required files before standalone execution]
# 1. open_g2m05_0000.xyz (Replaces the original .gjf file, used directly as the interpolation starting point, skipping optimization)
# 2. open_CASsa_0000.chk
# 3. closed_g2m05_0000.xyz
# 4. DVA_VDZP_0000_freq.slapaf.h5
# 5. PYR_VDZP_0000_freq.slapaf.h5
# 6. molcas_utils.py
# -------------------------------------------------------------------------

import ase
import ase.io
import ase.visualize
import ase.mep
import ase.optimize
import re
from ase.calculators.gaussian import Gaussian
from ase.units import Hartree

import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

import molcas_utils

class GaussianCASSCF(Gaussian):
    """
    1) Fix the "Blank line reading weights" error caused by an extra empty line before the addsec weights section;
    2) Fix the issue where the default ASE parser cannot read CASSCF energies (it only recognizes 'SCF Done:' / 'Energy=', whereas CASSCF uses the 'EIGENVALUE' format);
    3) Automatically parse the CI root number of the target electronic state from nroot=N in the method (when used with IOp(5/97=100,10/97=100), S1 is shifted to the last
       root position, i.e., root=nroot), no need to pass root manually;
    4) Discard the "Mulliken charges with hydrogens summed into heavy atoms" misread by ASE
       (the length does not match the number of atoms, which causes broadcast errors when
       writing xyz/trajectory).
    """

    def write_input(self, atoms, properties=None, system_changes=None):
        super().write_input(atoms, properties, system_changes)
        fname = self.label + ".com"
        with open(fname) as f:
            content = f.read()
        content = content.replace(
            "\n\n\n" + self.parameters["addsec"],
            "\n\n" + self.parameters["addsec"]
        )
        with open(fname, "w") as f:
            f.write(content)

    def _get_target_root(self):
        method = self.parameters.get("method", "")
        m = re.search(r"nroot\s*=\s*(\d+)", method, re.IGNORECASE)
        if not m:
            raise ValueError(
                "Failed to parse nroot from the method parameter, please check the CASSCF keyword format, "
                "e.g., 'CASSCF(6,5,nroot=3,stateaverage)'"
            )
        return int(m.group(1))

    def read_results(self):
        super().read_results()

        self.results.pop("charges", None)

        root = self._get_target_root()
        pattern = re.compile(r"\(\s*%d\)\s+EIGENVALUE\s+(-?\d+\.\d+)" % root)
        energy_hartree = None
        with open(self.label + ".log") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    energy_hartree = float(m.group(1))
        if energy_hartree is None:
            raise RuntimeError(
                f"Failed to find CASSCF EIGENVALUE for root={root} in {self.label}.log, "
                "please check if the log contains the 'EIGENVALUES AND EIGENVECTORS OF CI MATRIX' section."
            )
        self.results["energy"] = energy_hartree * Hartree

# === Step 2: Read initial geometries and skip the optimization phase ===
# Strictly retain the original .chk filename to prepare for subsequent wavefunction propagation
shutil.copy("open_CASsa_0000.chk", "DVA_open_S1_opt.chk")

# Read the replaced .xyz file as the starting point
dva_open = ase.io.read("open_g2m05_0000.xyz")

pyr_closed = ase.io.read("closed_g2m05_0000.xyz")

# === Step 3: Normal mode coordinates ===
n_images = 32

xyz_open = dva_open.get_positions().flatten()
xyz_closed = pyr_closed.get_positions().flatten()

disp_cart_fwd = xyz_closed - xyz_open
disp_cart_bwd = xyz_open - xyz_closed

h5_open = "DVA_VDZP_0000_freq.slapaf.h5"
xyz2nm_open = molcas_utils.read_molcas_h5_xyz2nm_trans(h5_open)
nm2xyz_open = molcas_utils.read_molcas_h5_nm2xyz_trans(h5_open)

h5_closed = "PYR_VDZP_0000_freq.slapaf.h5"
xyz2nm_closed = molcas_utils.read_molcas_h5_xyz2nm_trans(h5_closed)
nm2xyz_closed = molcas_utils.read_molcas_h5_nm2xyz_trans(h5_closed)

disp_nm_open = np.dot(xyz2nm_open, disp_cart_fwd)
disp_nm_closed = np.dot(xyz2nm_closed, disp_cart_bwd)

# === Step 4: Interpolation ===
images = [dva_open.copy()]

for i in range(1, n_images - 1):
    lambda_val = i / (n_images - 1)
    
    interp_cart_open = np.dot(nm2xyz_open, lambda_val * disp_nm_open)
    xyz_fwd = xyz_open + interp_cart_open
    
    interp_cart_closed = np.dot(nm2xyz_closed, (1.0 - lambda_val) * disp_nm_closed)
    xyz_bwd = xyz_closed + interp_cart_closed
    
    weight_bwd = lambda_val
    weight_fwd = 1.0 - lambda_val
    image_xyz = weight_fwd * xyz_fwd + weight_bwd * xyz_bwd
    
    image = dva_open.copy()
    image.set_positions(image_xyz.reshape(-1, 3))
    images.append(image)

images.append(pyr_closed.copy())
print(f"Successfully generated {len(images)} interpolated geometries (based on bidirectional normal modes).")

# === Step 5 (Part A): Preparation for wavefunction propagation ===
for i, image in enumerate(images):
    chk_filename = f"neb_image_{i:02d}.chk"
    
    if i == 0:
        source_chk = "DVA_open_S1_opt.chk" 
    else:
        source_chk = f"neb_image_{i-1:02d}.chk"
        
    try:
        shutil.copy(source_chk, chk_filename)
    except FileNotFoundError:
        print(f"Warning: {source_chk} not found. Check if the previous calculation was successful.")
        
    image_calc = GaussianCASSCF(
        label=f"neb_image_{i:02d}",
        method="CASSCF(6,5,nroot=3,stateaverage)",
        basis="6-31g(d,p)",
        extra=" nosym guess=read IOp(5/97=100,10/97=100)",
        charge=0,
        mult=1,
        output_type="P",
        chk=chk_filename,
        addsec="0.3333333 0.3333333 0.3333333",
        command="bash -l -c 'module load Gaussian/16.C.02-AVX2 && g16 < PREFIX.com > PREFIX.log'"
    )
    
    image.calc = image_calc

    if i > 0:
        print(f"Preparing wavefunction for Image {i:02d} (guess inherited from Image {i-1:02d})...")
        try:
            image.get_potential_energy()
        except Exception as e:
            print(f"Calculation interrupted at Image {i:02d}. Please check neb_image_{i:02d}.log")
            raise e

os.makedirs("Prep_Logs", exist_ok=True)
for i in range(len(images)):
    log_name = f"neb_image_{i:02d}.log"
    if os.path.exists(log_name):
        shutil.copy(log_name, os.path.join("Prep_Logs", log_name))

print("Sequential wavefunction propagation for all geometries is complete. Logs backed up to Prep_Logs directory. Ready for NEB!")

# [Added structure transfer] Save the images containing coordinates for direct reading by the independent second NEB script
ase.io.write("initial_images.traj", images)
print("Initial geometries saved to initial_images.traj. Please copy it along with the generated .chk files to the NEB working directory.")
