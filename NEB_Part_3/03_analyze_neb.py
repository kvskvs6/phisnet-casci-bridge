# -------------------------------------------------------------------------
# 03_analyze_neb.py
#
# [Features]
# 1. Extract the optimized configuration path from NEB calculations and output as an XYZ file for animation.
# 2. Render each configuration step as an image with 3D perspective rotation and save in sequential order.
# 3. Extract all macroscopic optimization steps (Macro steps) and plot individual line charts to monitor convergence.
# 4. Plot and save the final relative energy distribution profile.
# 5. [New] Extract the fmax for all macroscopic optimization steps and plot its convergence trend.
#
# [Required Input Files]
# 1. S1_NEB_Full_Path.traj
# -------------------------------------------------------------------------

import ase.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from ase.visualize.plot import plot_atoms

# =========================================================================
# === User Configuration for Fonts ===
# =========================================================================

# 1. Set global font to Arial bold
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'

# 2. Customize font sizes
FONT_SIZE_TITLE = 18       # Font size for plot titles
FONT_SIZE_LABEL = 16       # Font size for axis labels
FONT_SIZE_TICK = 14        # Font size for tick labels
FONT_SIZE_ANNOTATION = 14  # Font size for annotations (e.g., 'Current' label)
# =========================================================================

n_images = 32

# === Step 1: Read the full trajectory ===
# Read the entire trajectory file to extract all macroscopic optimization steps.
print("Loading trajectory file...")
all_images = ase.io.read("S1_NEB_Full_Path.traj", index=":")

# Calculate the total number of complete macroscopic steps in the trajectory.
n_macro_steps = len(all_images) // n_images
print(f"Detected {n_macro_steps} macroscopic optimization steps.")

# Extract the final converged optimized path.
optimized_images = all_images[-n_images:]

# === Step 2: Export final optimized trajectory ===
ase.io.write("S1_NEB_Final_Optimized_Path.xyz", optimized_images)
print("Saved multi-frame trajectory: S1_NEB_Final_Optimized_Path.xyz")

# === Step 3: Setup directories ===
output_dir = "NEB_Individual_Frames"
image_dir = "NEB_Step_Images"
macro_dir = "NEB_Macro_Step_Profiles"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(macro_dir, exist_ok=True)
print(f"Created/Accessed directories: ./{output_dir}/, ./{image_dir}/, ./{macro_dir}/")

# === Step 4: Prepare energy data for the final optimized path ===
energies = np.array([img.get_potential_energy() for img in optimized_images])
relative_energies = energies - energies[0]

# === Step 5: Draw individual interpolation frames (Structure + Energy Profile) ===
print("Generating individual frame images...")
for i, img in enumerate(optimized_images):
    # 1. Save XYZ file
    xyz_filename = f"Image_{i:02d}.xyz"
    xyz_filepath = os.path.join(output_dir, xyz_filename)
    ase.io.write(xyz_filepath, img)
    
    # 2. Draw and save picture for each step
    img_filename = f"Step_{i:02d}.png"
    img_filepath = os.path.join(image_dir, img_filename)
    
    fig = plt.figure(figsize=(10, 5))
    
    # Left: Atomic structure (includes rotation for better visualization).
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('off')
    plot_atoms(img, ax=ax1, rotation='-45x,45y,10z') 
    ax1.set_title(f"Molecular Structure (Image {i:02d})", fontsize=FONT_SIZE_TITLE)
    
    # Right: Energy profile highlighting the current progress.
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(n_images), relative_energies, marker='o', linestyle='-', color="#1f77b4", linewidth=2, alpha=0.4)
    
    # Highlight the current step.
    ax2.scatter(i, relative_energies[i], color='darkorange', s=120, zorder=5, edgecolor='black')
    ax2.annotate(f'Current: {relative_energies[i]:.2f} eV', 
                 xy=(i, relative_energies[i]), 
                 xytext=(i, relative_energies[i] + max(relative_energies)*0.1),
                 ha='center', 
                 fontsize=FONT_SIZE_ANNOTATION,
                 fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkorange", lw=1))
                 
    ax2.set_xlabel("Reaction Coordinate (Image Index)", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("Relative Energy (eV)", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("Minimum Energy Path Profile", fontsize=FONT_SIZE_TITLE)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(range(0, n_images, 2))
    ax2.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    plt.savefig(img_filepath, dpi=200)
    plt.close(fig)

print(f"Successfully saved {n_images} step images into '{image_dir}'.")

# === Step 6: Draw Macro Step Evolution Profiles ===
print("Generating macro step evolution profiles...")

# Extract all energies and calculate relative energies for consistent axis scaling.
all_energies_flat = np.array([img.get_potential_energy() for img in all_images])
# Truncate incomplete trailing frames (failsafe mechanism).
valid_length = n_macro_steps * n_images
all_energies_reshaped = all_energies_flat[:valid_length].reshape((n_macro_steps, n_images))

# Normalize energies so the first image of each macro step is set to 0.
rel_all_energies = all_energies_reshaped - all_energies_reshaped[:, 0:1]

# Determine the global Y-axis minimum and maximum for consistent scaling across plots.
global_y_min = np.min(rel_all_energies)
global_y_max = np.max(rel_all_energies)
y_padding = (global_y_max - global_y_min) * 0.1

# Array to record fmax for each step.
fmax_history = [] 

for step in range(n_macro_steps):
    plt.figure(figsize=(8, 5))
    
    # Plot all previous historical trajectories in light gray to visualize convergence dynamics.
    for p_step in range(step):
        plt.plot(range(n_images), rel_all_energies[p_step], marker='', linestyle='-', color='gray', alpha=0.15)
        
    # Plot the current macroscopic step trajectory in distinct blue.
    plt.plot(range(n_images), rel_all_energies[step], marker='o', linestyle='-', color="#1f77b4", linewidth=2)
    
    # Extract current step images and calculate fmax (excluding fixed endpoints).
    current_images = all_images[step * n_images : (step + 1) * n_images]
    fmax = 0.0
    # Slicing [1:-1] skips the fixed mirror images at both ends.
    for img in current_images[1:-1]:
        forces = img.get_forces()
        current_fmax = np.sqrt((forces**2).sum(axis=1)).max()
        if current_fmax > fmax:
            fmax = current_fmax
            
    # Record the current step's fmax for subsequent plotting.
    fmax_history.append(fmax)
    
    plt.xlabel("Reaction Coordinate (Image Index)", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("Relative Energy (eV)", fontsize=FONT_SIZE_LABEL)
    
    # Format title to include (Step X/Y) and denote current fmax with units.
    plt.title(f"Energy Profile Evolution (Step {step+1}/{n_macro_steps})  fmax = {fmax:.3f} eV/Å", fontsize=FONT_SIZE_TITLE)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, n_images, 2))
    plt.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
    
    # Lock Y-axis to facilitate flipbook-style observation of profile changes.
    plt.ylim(global_y_min - y_padding, global_y_max + y_padding)
    
    plt.tight_layout()
    plt.savefig(os.path.join(macro_dir, f"Macro_Step_{step:04d}.png"), dpi=200)
    plt.close()

print(f"Successfully saved {n_macro_steps} macro step profiles into '{macro_dir}'.")

# === Step 7: Final Complete 2D Potential Energy Surface Curve ===
plt.figure(figsize=(8, 5))
plt.plot(range(n_images), relative_energies, marker='o', linestyle='-', color="#1f77b4", linewidth=2)

plt.xlabel("Reaction Coordinate (Image Index)", fontsize=FONT_SIZE_LABEL)
plt.ylabel("Relative Energy (eV)", fontsize=FONT_SIZE_LABEL)
plt.title("S1 State Minimum Energy Path", fontsize=FONT_SIZE_TITLE)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(0, n_images, 2))
plt.tick_params(axis='both', labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig("S1_NEB_Energy_Profile.png", dpi=300)
# plt.show() is removed; script only generates and closes plots.
plt.close() 

# === Step 8: Draw fmax Evolution Profile (New Feature) ===
print("Generating fmax evolution profile...")
plt.figure(figsize=(8, 5))
# Plot the fmax evolution line chart in red to distinguish from energy curves.
plt.plot(range(1, n_macro_steps + 1), fmax_history, marker='s', linestyle='-', color="#d62728", linewidth=2)

plt.xlabel("Optimization Step (Macro Step)", fontsize=FONT_SIZE_LABEL)
plt.ylabel("Maximum Force, fmax (eV/Å)", fontsize=FONT_SIZE_LABEL)
plt.title("Convergence of fmax during NEB Optimization", fontsize=FONT_SIZE_TITLE)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig("S1_NEB_Fmax_Evolution.png", dpi=300)
plt.close()

print("Successfully saved 'S1_NEB_Fmax_Evolution.png'.")
print("All tasks completed successfully!")
