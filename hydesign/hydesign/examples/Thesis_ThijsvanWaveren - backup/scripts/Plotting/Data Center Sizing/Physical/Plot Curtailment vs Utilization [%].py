# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:11:31 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.6 - The Macroeconomic Trade-off
Publication-grade plot showing the structural dilemma between renewable 
curtailment and unutilized hardware, normalized as resource underutilization.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

# =============================================================================
# 1. PARAMETERS & DATA
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"

# Installed IT Capacities
IT_CAPACITIES_MW = np.array([16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0])

# Data extracted from optimization results
served_mw = np.array([15.0, 19.0, 27.0, 34.0, 39.0, 49.0, 53.0])
curtailment_mw = np.array([46.0, 41.0, 32.0, 25.0, 19.0, 8.0, 3.0])

# --- Percentage Conversions ---
# Unutilized % = (Installed - Served) / Installed
unutilized_pct = ((IT_CAPACITIES_MW - served_mw) / IT_CAPACITIES_MW) * 100

# Curtailment % = Curtailed MW / Max Average Available Generation (64 MW)
curtailment_pct = (curtailment_mw / 64.0) * 100

# Academic Color Palette 
C_CURTAIL = '#d35400'      # Deep Orange 
C_UNUTIL = '#2c3e50'       # Deep Navy 

print("\n" + "=" * 80)
print(" GENERATING PUBLICATION-GRADE SIZING TRADE-OFF PLOT ".center(80))
print("=" * 80)

# =============================================================================
# 2. VISUALIZATION
# =============================================================================
fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor='white')

# Plot the Main Curves (White marker edges create a premium "cutout" effect)
ax.plot(IT_CAPACITIES_MW, curtailment_pct, color=C_CURTAIL, marker='o', markersize=9, 
        markeredgecolor='white', markeredgewidth=1.5, linewidth=2.5, 
        label='HPP Underutilization (Curtailment)', zorder=4)

ax.plot(IT_CAPACITIES_MW, unutilized_pct, color=C_UNUTIL, marker='s', markersize=9, 
        markeredgecolor='white', markeredgewidth=1.5, linewidth=2.5, 
        label='Data Center Underutilization (Idle Capacity)', zorder=4)

# =============================================================================
# 3. FORMATTING
# =============================================================================
# Refined, precise Y-axis title
ax.set_ylabel("Resource Underutilization (%)", fontsize=11, fontweight='bold', color='#222222')
ax.set_xlabel("Data Center Installed IT Capacity (MW)", fontsize=11, fontweight='bold', color='#222222', labelpad=10)

# Proportional X-Axis Ticks (Bolded to match thesis standard)
ax.set_xticks(IT_CAPACITIES_MW)
ax.set_xticklabels([f"{int(mw)}" for mw in IT_CAPACITIES_MW], fontsize=10, fontweight='bold', color='#222222')
ax.set_xlim(12, 105)

# Format Y-Axis as Percentages with 15% Headroom
max_val = max(curtailment_pct.max(), unutilized_pct.max())
ax.set_ylim(0, max_val * 1.15)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

# Clean, professional grid and spines
ax.grid(axis='y', linestyle='-', alpha=0.3, color='#b0b0b0', zorder=0)
ax.grid(axis='x', linestyle='-', alpha=0.15, color='#b0b0b0', zorder=0) # Faint x-grid helps track the intersection

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('#222222')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('#222222')
ax.tick_params(axis='both', colors='#222222', labelsize=10)

# Legend placed cleanly outside at the bottom
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, 
          frameon=False, fontsize=11)

plt.tight_layout()

# Save
save_path_svg = os.path.join(BASE_FOLDER, 'Thesis_Sizing_Tradeoff_Academic1.svg')
if os.path.exists(BASE_FOLDER):
    plt.savefig(save_path_svg, dpi=300, bbox_inches='tight')
    print(f" ✅ Saved Publication-Grade Trade-Off Plot to: {save_path_svg}")
plt.show()