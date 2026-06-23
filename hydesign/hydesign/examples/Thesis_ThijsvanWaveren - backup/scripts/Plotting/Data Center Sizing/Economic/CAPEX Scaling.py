# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:49:35 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.5/3.6 - Total Annual Fixed Costs (The Linear Penalty)
Publication-grade visualization (Area + Subtle Markers)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PARAMETERS
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITIES_MW = np.array([16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0])

DISCOUNT_RATE = 0.075

FACILITY_CAPEX_PER_MW = 9.60
FACILITY_LIFETIME = 20

IT_CAPEX_PER_MW = 15.33
IT_LIFETIME = 5

FIXED_OPEX_PER_MW = 0.25

# =============================================================================
# 2. CALCULATIONS
# =============================================================================
def calculate_crf(rate, years):
    return (rate * (1 + rate)**years) / ((1 + rate)**years - 1)

crf_fac = calculate_crf(DISCOUNT_RATE, FACILITY_LIFETIME)
crf_it = calculate_crf(DISCOUNT_RATE, IT_LIFETIME)

total_ann_it = (IT_CAPEX_PER_MW * crf_it) * IT_CAPACITIES_MW
total_ann_fac = (FACILITY_CAPEX_PER_MW * crf_fac) * IT_CAPACITIES_MW
total_ann_opex = FIXED_OPEX_PER_MW * IT_CAPACITIES_MW

total_costs = total_ann_it + total_ann_fac + total_ann_opex

# =============================================================================
# 3. PUBLICATION-GRADE PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

x = IT_CAPACITIES_MW
y_it = total_ann_it
y_fac = total_ann_fac
y_opex = total_ann_opex
y_total = total_costs

# --- COLOR PALETTE (Lighter, professional blue gradient) ---
C_IT = '#3a7ca5'        # Medium slate blue (Lighter, easier to read)
C_FACILITY = '#81adc8'  # Soft steel blue
C_OPEX = '#dbe4ed'      # Ultra-light grey-blue

# --- STACKED AREA ---
ax.stackplot(
    x,
    y_it,
    y_fac,
    y_opex,
    colors=[C_IT, C_FACILITY, C_OPEX],
    alpha=0.98,
    edgecolor='white',
    linewidth=0.9,
    zorder=2
)

# --- VERY SUBTLE MARKERS ---
ax.scatter(
    x,
    y_total,
    color='#222222',
    s=14,         # small and subtle
    alpha=0.6,
    zorder=5
)

# --- SMART LABEL POSITIONING (NO OVERLAP) ---
vertical_offsets = [6, 10, 6, 10, 6, 10, 6]  # stagger vertically

# for i in range(len(x)):
#     ax.text(
#         x[i],
#         y_total[i] + vertical_offsets[i],
#         f'€{y_total[i]:.0f}M',
#         ha='center',
#         va='bottom',
#         fontsize=10,
#         fontweight='semibold',
#         color='#222222'
#     )

# =============================================================================
# 4. FORMATTING
# =============================================================================
ax.set_ylabel("Annualized Fixed Cost (Millions € / Year)", fontsize=11, fontweight='bold')
ax.set_xlabel("Installed IT Capacity (MW)", fontsize=11, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([f"{int(v)}" for v in x], fontsize=10)

ax.set_xlim(min(x), max(x))
ax.set_ylim(0, max(y_total) * 1.08)

# --- CLEAN GRID ---
ax.grid(axis='y', linestyle='-', alpha=0.2)
# No vertical grid

# --- SPINES ---
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# --- LEGEND (REVERSED: Matches the bottom-to-top stacking order) ---
handles = [
    plt.Rectangle((0,0),1,1,color=C_IT),
    plt.Rectangle((0,0),1,1,color=C_FACILITY),
    plt.Rectangle((0,0),1,1,color=C_OPEX),
]

labels = [
    'Annualized IT CAPEX (5-year)',
    'Annualized Facility CAPEX (20-year)',
    'Fixed OPEX (Staff, Maint., Insurance)'
]

ax.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=False,
    fontsize=10
)

plt.subplots_adjust(bottom=0.22)
plt.tight_layout()

# --- SAVE ---
save_path = os.path.join(BASE_FOLDER, 'Fixed_Cost_Area_Publication.svg')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()