# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:28:32 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.6 - The Macroeconomic Trade-off
Visualizes Curtailment vs. Unutilized IT Capacity to prove the 
"Conservation of Inefficiency" and the location of the financial optimum.
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

# 2. Shading for Visual Weight (Optional but highly effective)
# Fills the area under the curves with a very faint tint to imply "volume of waste"
# ax.fill_between(IT_CAPACITIES_MW, 0, curtailment_mw, color=C_CURTAIL, alpha=0.05, zorder=1)
# ax.fill_between(IT_CAPACITIES_MW, 0, unutilized_mw, color=C_UNUTIL, alpha=0.08, zorder=1)

# 3. Add the "Kill-Shot" Insight: The Financial Optimum Line
# optimum_mw = 50.0
# ax.axvline(x=optimum_mw, color=C_FINANCE_OPT, linestyle='--', linewidth=2, zorder=2)

# Annotate the Financial Optimum
# ax.annotate('FINANCIAL\nOPTIMUM\n(50 MW)', 
#             xy=(optimum_mw, 40), xytext=(optimum_mw - 12, 40),
#             ha='center', va='center', fontsize=10, fontweight='bold', color=C_FINANCE_OPT,
#             arrowprops=dict(arrowstyle='->', color=C_FINANCE_OPT, lw=1.5), path_effects=halo, zorder=5)

# Annotate the Physical Intersection
# ax.annotate('Physical Intersection\n(~57 MW)', 
#             xy=(57, 15), xytext=(75, 18),
#             ha='center', va='center', fontsize=10, fontweight='bold', color='#555555',
#             arrowprops=dict(arrowstyle='-', color='#888888', lw=1.5), path_effects=halo, zorder=5)

# 4. Selective Data Labels (Cleanliness over clutter)
# for i, cap in enumerate(IT_CAPACITIES_MW):
#     if cap in [16, 50, 100]:
#         # Curtailment Labels
#         ax.text(cap, curtailment_mw[i] + 2, f"{int(curtailment_mw[i])} MW", 
#                 color=C_CURTAIL, fontweight='bold', ha='center', path_effects=halo, zorder=5)
#         # Unutilized Labels
#         # Adjust y-offset for the 16MW label to prevent overlap
#         y_offset = -3 if cap == 16 else 2
#         ax.text(cap, unutilized_mw[i] + y_offset, f"{int(unutilized_mw[i])} MW", 
#                 color=C_UNUTIL, fontweight='bold', ha='center', path_effects=halo, zorder=5)

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
print(f"✅ Plot saved to: {save_path}")

plt.show()