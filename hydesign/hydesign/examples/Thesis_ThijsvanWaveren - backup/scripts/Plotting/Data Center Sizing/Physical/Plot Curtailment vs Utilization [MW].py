# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Visualizes the physical sizing trade-off between renewable energy curtailment and unutilized IT hardware.

Plots the average curtailed power against the unutilized data center capacity 
across varying installed IT capacities. This highlights the inherent structural dilemma 
in off-grid operations: undersizing the data center results in significant renewable 
energy waste, while oversizing results in stranded capital assets due to the variable 
generation profile of the hybrid power plant.
"""


import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import os

# =============================================================================
# 1. PARAMETERS & DATA
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"

# Proportional X-axis data
IT_CAPACITIES_MW = np.array([16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0])

# Data extracted from previous optimization scripts
curtailment_mw = np.array([46, 41, 32, 25, 19, 8, 3])
unutilized_mw = np.array([1, 1, 3, 6, 11, 26, 47])

# Academic-Consulting Color Palette
C_CURTAIL = '#e74c3c'      # Strong Red 
C_UNUTIL = '#8c564b'       # Structural Brown/Rust
C_FINANCE_OPT = '#113b5e'  # Deep Navy for the Financial Optimum
C_GRID = '#e0e0e0'

halo = [path_effects.withStroke(linewidth=3, foreground="white", alpha=0.9)]

# =============================================================================
# 2. VISUALIZATION
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6.5), facecolor='white')

# 1. Plot the Main Curves (Thick, confident lines with distinct markers)
ax.plot(IT_CAPACITIES_MW, curtailment_mw, color=C_CURTAIL, marker='o', markersize=8, 
        linewidth=3, label='Curtailed Renewable Power', zorder=4)

ax.plot(IT_CAPACITIES_MW, unutilized_mw, color=C_UNUTIL, marker='s', markersize=8, 
        linewidth=3, label='Unutilized IT Capacity', zorder=4)


# =============================================================================
# 3. FORMATTING & LEGEND
# =============================================================================
ax.set_title("Data Center Sizing Trade-Off", 
             fontsize=16, fontweight='bold', pad=20, color='#333333')
ax.set_ylabel("Average Power (MW)", fontsize=12, fontweight='bold', color='#444444')
ax.set_xlabel("Installed IT Capacity (MW)", fontsize=12, fontweight='bold', color='#444444')

# Proportional X-Axis Ticks
ax.set_xticks(IT_CAPACITIES_MW)
ax.set_xticklabels([f"{int(mw)}" for mw in IT_CAPACITIES_MW], fontsize=11)
ax.set_xlim(12, 105)
ax.set_ylim(-2, 50)

# Clean, professional grid and spines
ax.grid(axis='y', linestyle='-', alpha=0.4, color=C_GRID, zorder=0)
ax.grid(axis='x', linestyle='--', alpha=0.3, color=C_GRID, zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('#444444')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('#444444')

# Legend placed outside to the bottom
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, 
          frameon=False, edgecolor='#e0e0e0', facecolor='white', framealpha=1, fontsize=11, borderpad=1)

plt.tight_layout()

# Save
save_path = os.path.join(BASE_FOLDER, 'Thesis_Sizing_Tradeoff_Academic.svg')
plt.savefig(save_path, dpi=300, bbox_inches='tight')


plt.show()