# -*- coding: utf-8 -*-
"""
Section 3.5 - Economies of Scale in Revenue Generation (Stacked Area)
Extracts the actual delivered Energy (GWh) from the 3D sweep CSVs, 
calculates Gross Revenue by workload tier, and plots a continuous Stacked Area Chart.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# =============================================================================
# 1. MASTER PARAMETERS & PATHS
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
os.chdir(BASE_FOLDER)

IT_CAPACITIES_MW = [16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0] 

# The exact optimal combinations determined previously (A, B1, B2)
optimal_mixes = {
    16.0:  (5.0, 6.0, 4.0),
    20.0:  (8.0, 0.0, 11.0),
    30.0:  (8.0, 0.0, 17.0),
    40.0:  (5.0, 6.0, 17.0),
    50.0:  (5.0, 6.0, 20.0),
    75.0:  (5.0, 6.0, 25.0),
    100.0: (5.0, 6.0, 26.0)
}

# Revenue Prices (Millions of € per GWh)
PRICE_A  = 4.0  
PRICE_B1 = 2.8  
PRICE_B2 = 1.6  
PRICE_C  = 0.4  


C_A = '#08306b'      # Deep Navy
C_B1 = '#2879b9'     # Strong Blue
C_B2 = '#73b3d8'     # Soft Blue
C_C = '#c8ddf0'      # Pale Blue
# Consulting-Style Color Palette
C_GRID = '#e0e0e0'

# =============================================================================
# 2. CSV DATA EXTRACTION
# =============================================================================
print("\n" + "=" * 60)
print(" EXTRACTING REVENUE FROM OPTIMAL WORKLOAD MIXES ".center(60))
print("=" * 60)

revenue_data = {}

for cap in IT_CAPACITIES_MW:
    # Handle potential filename variations
    file_path_1 = os.path.join(BASE_FOLDER, f"Feasible_3D_Sweep_Results_99.9pct_IT{cap}.csv")
    file_path_2 = os.path.join(BASE_FOLDER, f"Feasible_3D_Sweep_Results_99.9pct_IT{cap:.1f}.csv")
    
    file_path = file_path_1 if os.path.exists(file_path_1) else file_path_2
    
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Could not find CSV for {cap} MW. Filling with zeros.")
        revenue_data[cap] = {"A": 0, "B1": 0, "B2": 0, "C": 0}
        continue

    df = pd.read_csv(file_path)
    
    # Isolate the exact optimal mix row
    target_a, target_b1, target_b2 = optimal_mixes[cap]
    match = df[(df['Tier_A_MW'] == target_a) & 
               (df['Tier_B1_MW'] == target_b1) & 
               (df['Tier_B2_MW'] == target_b2)]
    
    if not match.empty:
        best_row = match.iloc[0]
    else:
        print(f"⚠️ Optimal mix not found in {cap} MW CSV. Falling back to highest objective value.")
        best_row = df.loc[df["Objective_Value"].idxmax()]
        
    # Extract Annual GWh and convert to Revenue in Millions €
    rev_a = best_row['Energy_A_Annual_GWh'] * PRICE_A
    rev_b1 = best_row['Energy_B1_Annual_GWh'] * PRICE_B1
    rev_b2 = best_row['Energy_B2_Annual_GWh'] * PRICE_B2
    rev_c = best_row['Energy_C_Annual_GWh'] * PRICE_C
    
    revenue_data[cap] = {"A": rev_a, "B1": rev_b1, "B2": rev_b2, "C": rev_c}
    
    print(f"{cap:03.0f} MW | A: €{rev_a:04.1f}M | B1: €{rev_b1:04.1f}M | B2: €{rev_b2:04.1f}M | C: €{rev_c:04.1f}M | TOTAL: €{(rev_a+rev_b1+rev_b2+rev_c):.1f}M")

# Convert to arrays for plotting
df_rev = pd.DataFrame.from_dict(revenue_data, orient='index')
capacities = np.array(IT_CAPACITIES_MW)

rev_a = df_rev["A"].values
rev_b1 = df_rev["B1"].values
rev_b2 = df_rev["B2"].values
rev_c = df_rev["C"].values
totals = rev_a + rev_b1 + rev_b2 + rev_c

# =============================================================================
# 3. CREATE STACKED AREA VISUALIZATION
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='white')

# 1. Plot the Stacked Area
labels = ['Tier A (Firm)', 'Tier B1 (Daily)', 'Tier B2 (Weekly)', 'Tier C (Opportunistic)']
colors = [C_A, C_B1, C_B2, C_C]

ax.stackplot(capacities, rev_a, rev_b1, rev_b2, rev_c, 
             labels=labels, colors=colors, alpha=0.9, edgecolor='white', linewidth=1.0, zorder=3)

# 2. Add Total Line & Annotations on the top ridge
# ax.plot(capacities, totals, color='#333333', linewidth=2, zorder=4)

for i, cap in enumerate(capacities):
    if totals[i] > 0:
        # Add a small dot at the peak of each point
        ax.plot(cap, totals[i], marker='o', markersize=6, color='#333333', zorder=5, alpha = 0.3)
        # Label the total revenue
        # ax.text(cap, totals[i] + max(totals)*0.03, f'€{int(totals[i])}M', ha='center', va='bottom', 
        #         fontweight='bold', fontsize=10, color='#333333')

# 3. Chart Formatting
ax.set_title("Annual Gross Revenue Breakdown by Workload Tier", fontsize=15, fontweight='bold', pad=20, color='#333333')
ax.set_ylabel("Annual Gross Revenue (Millions €)", fontsize=12, fontweight='bold', color='#444444')
ax.set_xlabel("Installed IT Capacity (MW)", fontsize=12, fontweight='bold', color='#444444')

ax.set_xlim(min(capacities), max(capacities))
ax.set_xticks(capacities)
ax.set_xticklabels([f"{cap:.0f}" for cap in capacities], fontsize=11, color='#333333')
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
ax.legend(handles[::-1], legend_labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.15), 
          ncol=4, fontsize=10, frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9)

plt.subplots_adjust(bottom=0.25)

# Save & Show
save_path = os.path.join(BASE_FOLDER, 'revenue_by_tier_area.svg')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print(f"✅ Plot successfully extracted and saved to: {save_path}")
print("=" * 60)
plt.show()