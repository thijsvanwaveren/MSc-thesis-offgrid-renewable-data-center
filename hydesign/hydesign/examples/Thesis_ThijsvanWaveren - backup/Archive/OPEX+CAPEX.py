# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:57:46 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section X.X - Total Annual Costs (Stacked Area)
Visualizes the Total Annualized CAPEX and OPEX scaling.
Matched to the visual style of the Revenue Stacked Area Chart.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PARAMETERS & INPUTS
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
# os.makedirs(BASE_FOLDER, exist_ok=True) # Uncomment if folder might not exist

IT_CAPACITIES_MW = np.array([16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0])

# --- Financial Inputs (CAPEX) ---
DISCOUNT_RATE = 0.075  # 7.5% WACC
FACILITY_CAPEX_PER_MW = 9.60   # Millions €
FACILITY_LIFETIME = 20         # Years
IT_CAPACITIES_PER_MW = 15.33   # Millions €
IT_LIFETIME = 5                # Years

# --- Cost Color Palette (Distinct from Revenue Blues) ---
C_OPEX = '#eb8c34'   # Warm Orange (Energy/OPEX)
C_CAPEX = '#5b6b78'  # Slate Grey (Infrastructure/CAPEX)
C_GRID = '#e0e0e0'   # Light grey for grid

# =============================================================================
# 2. CALCULATIONS
# =============================================================================
print("\n" + "=" * 60)
print(" CALCULATING ANNUALIZED COSTS ".center(60))
print("=" * 60)

def calculate_crf(rate, years):
    return (rate * (1 + rate)**years) / ((1 + rate)**years - 1)

crf_fac = calculate_crf(DISCOUNT_RATE, FACILITY_LIFETIME)
crf_it = calculate_crf(DISCOUNT_RATE, IT_LIFETIME)

# Component Calculations (Millions €)
total_ann_it = (IT_CAPACITIES_PER_MW * crf_it) * IT_CAPACITIES_MW
total_ann_fac = (FACILITY_CAPEX_PER_MW * crf_fac) * IT_CAPACITIES_MW

# Unified CAPEX Layer
total_capex = total_ann_it + total_ann_fac

# OPEX Layer (Electricity / Variable)
# Interpolating values based on the provided Variable OPEX figure bounds 
# (~€16.29M at 16MW to ~€17.35M at 100MW)
opex_costs = np.interp(IT_CAPACITIES_MW, [16.0, 100.0], [16.29, 17.35])

# Total Costs
totals = opex_costs + total_capex

for i, cap in enumerate(IT_CAPACITIES_MW):
    print(f"{cap:03.0f} MW | OPEX: €{opex_costs[i]:05.2f}M | CAPEX: €{total_capex[i]:05.2f}M | TOTAL: €{totals[i]:.2f}M")

# =============================================================================
# 3. CREATE STACKED AREA VISUALIZATION
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='white')

# 1. Plot the Stacked Area (OPEX on bottom, CAPEX on top)
labels = ['Total Annualized CAPEX', 'Annual OPEX (Electricity)']
colors = [C_CAPEX, C_OPEX]

# Note: order in stackplot is bottom-to-top, so OPEX is first
ax.stackplot(IT_CAPACITIES_MW, opex_costs, total_capex, 
             labels=['Annual OPEX (Electricity & Ops)', 'Total Annualized CAPEX'], 
             colors=[C_OPEX, C_CAPEX], alpha=0.9, edgecolor='white', linewidth=1.0, zorder=3)

# 2. Add Total Line & Annotations on the top ridge
ax.plot(IT_CAPACITIES_MW, totals, color='#333333', linewidth=1, zorder=4, alpha=0.5)

for i, cap in enumerate(IT_CAPACITIES_MW):
    if totals[i] > 0:
        # Add a small dot at the peak of each point
        ax.plot(cap, totals[i], marker='o', markersize=4, color='#333333', zorder=5, alpha=0.35)

# 3. Chart Formatting
ax.set_ylabel("Annual Total Costs (Millions €)", fontsize=12, fontweight='bold', color='#444444')
ax.set_xlabel("Installed IT Capacity (MW)", fontsize=12, fontweight='bold', color='#444444')

ax.set_xlim(min(IT_CAPACITIES_MW), max(IT_CAPACITIES_MW))
ax.set_xticks(IT_CAPACITIES_MW)
ax.set_xticklabels([f"{cap:.0f}" for cap in IT_CAPACITIES_MW], fontsize=11, color='#333333')
ax.set_ylim(0, max(totals) * 1.15)

# Clean Grid & Spines
ax.grid(axis='y', linestyle='-', alpha=0.3, color=C_GRID, zorder=0)
ax.grid(axis='x', linestyle='--', alpha=0.3, color=C_GRID, zorder=0) 

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('#444444')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('#444444')

# 4. Legend matching visual stack order
handles, legend_labels = ax.get_legend_handles_labels()
# Reverse the handles/labels so CAPEX (top visual layer) appears first/left in the legend
ax.legend(handles[::-1], legend_labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.15), 
          ncol=2, fontsize=10, frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9)

plt.subplots_adjust(bottom=0.25)

# Save & Show
save_path = os.path.join(BASE_FOLDER, 'total_costs_stacked_area.svg')
# plt.savefig(save_path, dpi=300, bbox_inches='tight') # Uncomment to save
print("\n" + "=" * 60)
print(f"✅ Plot successfully generated.")
print("=" * 60)
plt.show()