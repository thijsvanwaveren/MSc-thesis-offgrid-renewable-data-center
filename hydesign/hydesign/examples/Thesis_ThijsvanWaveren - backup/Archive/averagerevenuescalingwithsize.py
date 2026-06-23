# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:32:00 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.5 - The Dilution of Average Revenue
Visualizes the Average Revenue per MWh supplied across optimal IT capacities, 
comparing the portfolio value with and without opportunistic Tier C workloads.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. MASTER PARAMETERS
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITIES_MW = [16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0] 

# Financial Assumptions for Revenue (From Thesis Table 2.7)
REVENUE_PER_MWH = {
    "A": 4000.0, 
    "B1": 2800.0, 
    "B2": 1600.0,
    "C": 400.0
}
OTHER_VARIABLE_OPEX_EUR_PER_MWH = 3

# Colors
COLOR_NO_C = '#08306b'     # Deep Navy (Excluding Tier C)
COLOR_WITH_C = '#f39c12'   # Gold/Orange (Including Tier C)

# =============================================================================
# 2. DATA EXTRACTION & OPTIMIZATION
# =============================================================================
avg_rev_data = []

print("Extracting Average Unit Revenue for optimal mixes...")

for cap in IT_CAPACITIES_MW:
    file_name = f"Feasible_3D_Sweep_Results_99.9pct_IT{cap:.1f}.csv"
    file_path = os.path.join(BASE_FOLDER, file_name)
    
    # Load or generate synthetic data
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Synthetic fallback
        rows = []
        max_a, max_b1, max_b2 = int(cap/1.5)+1, int(cap/1.1)+1, int(cap/1.3)+1
        for a in range(0, max_a):
            for b1 in range(0, max_b1):
                for b2 in range(0, max_b2):
                    if (a*1.5 + b1*1.1 + b2*1.3) <= cap: 
                        rows.append({
                            'Tier_A_MW': a, 'Tier_B1_MW': b1, 'Tier_B2_MW': b2,
                            'Energy_A_Annual_GWh': a * 8.76 * 0.9,
                            'Energy_B1_Annual_GWh': b1 * 8.76 * 0.5,
                            'Energy_B2_Annual_GWh': b2 * 8.76 * 0.3,
                            'Total_Delivered_Annual_GWh': (a*0.9 + b1*0.5 + b2*0.3 + (cap*0.1)) * 8.76, 
                            'LCOE_delivered': np.random.uniform(40, 60)
                        })
        df = pd.DataFrame(rows)

    # Standardize Energy Columns
    for col in df.columns:
        if "Energy" in col and "MWh" in col:
            df[col.replace("_MWh", "_GWh")] = df[col] / 1000.0

    e_a = df.get('Energy_A_Annual_GWh', 0)
    e_b1 = df.get('Energy_B1_Annual_GWh', 0)
    e_b2 = df.get('Energy_B2_Annual_GWh', 0)
    
    e_total_with_c = df.get('Total_Delivered_Annual_GWh', e_a + e_b1 + e_b2)
    e_committed = e_a + e_b1 + e_b2
    e_c = np.maximum(0, e_total_with_c - e_committed)

    # Calculate Revenues
    rev_a = e_a * 1000.0 * REVENUE_PER_MWH["A"]
    rev_b1 = e_b1 * 1000.0 * REVENUE_PER_MWH["B1"]
    rev_b2 = e_b2 * 1000.0 * REVENUE_PER_MWH["B2"]
    rev_c = e_c * 1000.0 * REVENUE_PER_MWH["C"]
    
    df["Revenue_No_C_EUR"] = rev_a + rev_b1 + rev_b2
    df["Revenue_With_C_EUR"] = df["Revenue_No_C_EUR"] + rev_c
    
    # Find the Optimum based on the Committed Contribution Margin
    df["Total_HPP_Cost_EUR"] = df["LCOE_delivered"] * (e_total_with_c * 1000.0) 
    df["DC_Variable_OPEX_EUR"] = OTHER_VARIABLE_OPEX_EUR_PER_MWH * (e_committed * 1000.0)
    df["Contribution_Margin_M_EUR"] = (df["Revenue_No_C_EUR"] - (df["Total_HPP_Cost_EUR"] + df["DC_Variable_OPEX_EUR"])) / 1e6

    best_idx = df["Contribution_Margin_M_EUR"].idxmax()
    best_row = df.loc[best_idx]
    
    # Calculate Average Revenue per MWh
    # Ensure no division by zero
    total_committed_mwh = e_committed[best_idx] * 1000.0
    total_energy_with_c_mwh = e_total_with_c[best_idx] * 1000.0
    
    avg_rev_no_c = best_row["Revenue_No_C_EUR"] / total_committed_mwh if total_committed_mwh > 0 else 0
    avg_rev_with_c = best_row["Revenue_With_C_EUR"] / total_energy_with_c_mwh if total_energy_with_c_mwh > 0 else 0
    
    avg_rev_data.append({
        "IT_Capacity_MW": cap,
        "Avg_Rev_No_C": avg_rev_no_c,
        "Avg_Rev_With_C": avg_rev_with_c
    })

df_avg_rev = pd.DataFrame(avg_rev_data)

# =============================================================================
# 3. CREATE VISUALIZATION
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

capacities = df_avg_rev["IT_Capacity_MW"].values
avg_rev_no_c = df_avg_rev["Avg_Rev_No_C"].values
avg_rev_with_c = df_avg_rev["Avg_Rev_With_C"].values

# Plot the lines
ax.plot(capacities, avg_rev_no_c, color=COLOR_NO_C, marker='o', markersize=8, linestyle='-', 
        linewidth=2.5, zorder=5, label='Average Unit Revenue (Excluding Tier C)')

ax.plot(capacities, avg_rev_with_c, color=COLOR_WITH_C, marker='D', markersize=8, linestyle='--', 
        linewidth=2.5, zorder=5, label='Average Unit Revenue (Including Tier C)')

# Add data labels
for i, txt in enumerate(avg_rev_no_c):
    ax.annotate(f'€{int(txt)}', (capacities[i], avg_rev_no_c[i]), 
                textcoords="offset points", xytext=(0, 10), ha='center', 
                color=COLOR_NO_C, fontweight='bold', fontsize=10)

for i, txt in enumerate(avg_rev_with_c):
    # Shift text down to prevent overlapping
    ax.annotate(f'€{int(txt)}', (capacities[i], avg_rev_with_c[i]), 
                textcoords="offset points", xytext=(0, -18), ha='center', 
                color='#d35400', fontweight='bold', fontsize=10)

# Formatting
ax.set_title("Average Revenue Scaling", fontsize=15, fontweight='bold', pad=20, color='#333333')
ax.set_ylabel("Average Revenue (€/MWh)", fontsize=12, fontweight='bold', color='grey')
ax.set_xlabel("Installed IT Capacity (MW)", fontsize=12, fontweight='bold', color='grey')

ax.set_xticks(capacities)
ax.set_xticklabels([f"{cap:.0f}" for cap in capacities], fontsize=11)

# Dynamic Y-axis
y_max = max(avg_rev_no_c) * 1.15
ax.set_ylim(0, y_max)

ax.grid(axis='y', linestyle='-', alpha=0.3)
ax.grid(axis='x', linestyle='--', alpha=0.1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('grey')
ax.spines['bottom'].set_color('grey')

ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()

# Save
save_path = os.path.join(BASE_FOLDER, 'averagerevenuescalingwithsize.svg')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"-> Plot saved to: {save_path}")
plt.show()