# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:00:23 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.4 - Optimal Energy Balance by Data Center Size
Finds the financially optimal workload mix for each IT capacity, 
including opportunistic Tier C workloads in the revenue and margin optimization,
and visualizes the physical energy balance (Average Power in MW), 
including exact MW labels inside each bar tier.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. MASTER USER INPUTS & PARAMETERS
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITIES_MW = [16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0] 

# Financial Assumptions (Now explicitly including Tier C)
REVENUE_PER_MWH = {
    "A": 4000.0, 
    "B1": 2800.0, 
    "B2": 1600.0,
    "C": 400.0  # Added Tier C Revenue
}
OTHER_VARIABLE_OPEX_EUR_PER_MWH = 3

# Exact Color Palette Requested
C_A = '#08306b'        # Deep Navy (Tier A)
C_B1 = '#2879b9'       # Strong Blue (Tier B1)
C_B2 = '#73b3d8'       # Soft Blue (Tier B2)
C_C = '#c8ddf0'        # Pale Blue (Tier C)

# Styling for Curtailment ("Waste")
C_WASTE_FACE = 'white'
C_WASTE_EDGE = '#e74c3c' # Red outline
HATCH_STYLE = '//'

# =============================================================================
# 2. DATA PROCESSING & OPTIMIZATION FINDER
# =============================================================================
optimal_mixes = []

print("Scanning for optimal energy balances across all IT Capacities...")

for cap in IT_CAPACITIES_MW:
    file_name = f"Feasible_3D_Sweep_Results_99.9pct_IT{cap:.1f}.csv"
    file_path = os.path.join(BASE_FOLDER, file_name)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        print(f"  [WARNING] File not found: {file_name}. Generating synthetic data...")
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
                            'Total_Delivered_Annual_GWh': (a*0.9 + b1*0.5 + b2*0.3 + (cap*0.1)) * 8.76, # Included cap-scaled Tier C
                            'Total_Produced_Annual_GWh': cap * 8.76 * 0.6, 
                            'LCOE_delivered': np.random.uniform(40, 60)
                        })
        df = pd.DataFrame(rows)

    # 1. Standardize Columns to GWh
    for col in df.columns:
        if "Energy" in col and "MWh" in col:
            df[col.replace("_MWh", "_GWh")] = df[col] / 1000.0

    # Ensure Core Energy Columns Exist
    e_a = df.get('Energy_A_Annual_GWh', 0)
    e_b1 = df.get('Energy_B1_Annual_GWh', 0)
    e_b2 = df.get('Energy_B2_Annual_GWh', 0)
    e_total_del = df.get('Total_Delivered_Annual_GWh', e_a + e_b1 + e_b2)
    
    # Calculate Tier C (Uncommitted Energy)
    e_c = df.get('Energy_C_Annual_GWh', e_total_del - (e_a + e_b1 + e_b2))
    df['Energy_C_Annual_GWh'] = np.maximum(0, e_c) # Prevent negative floats
    e_c = df['Energy_C_Annual_GWh']
    
    # Calculate Curtailment
    if 'Curtailment_Annual_GWh' in df.columns:
        curtailment = df['Curtailment_Annual_GWh']
    elif 'Curtailment_GWh' in df.columns:
        curtailment = df['Curtailment_GWh']
    elif 'Total_Produced_Annual_GWh' in df.columns:
        curtailment = df['Total_Produced_Annual_GWh'] - e_total_del
    else:
        curtailment = 0 # Default fallback
    df['Curtailment_Annual_GWh'] = np.maximum(0, curtailment)

    # 2. Calculate Financials to Find the Optimum
    rev_a = e_a * 1000.0 * REVENUE_PER_MWH["A"]
    rev_b1 = e_b1 * 1000.0 * REVENUE_PER_MWH["B1"]
    rev_b2 = e_b2 * 1000.0 * REVENUE_PER_MWH["B2"]
    rev_c = e_c * 1000.0 * REVENUE_PER_MWH["C"]
    
    # Revenue now includes Tier C
    df["Revenue_EUR"] = rev_a + rev_b1 + rev_b2 + rev_c

    # Total HPP cost dynamically includes Tier C because e_total_del encompasses it
    df["Total_HPP_Cost_EUR"] = df["LCOE_delivered"] * (e_total_del * 1000.0)
    
    # Variable OPEX applies strictly to committed workloads (Tier A, B1, B2)
    committed_mwh = (e_a + e_b1 + e_b2) * 1000.0
    df["DC_Variable_OPEX_EUR"] = OTHER_VARIABLE_OPEX_EUR_PER_MWH * committed_mwh
    
    # Optimization metric now naturally considers Tier C's contribution
    df["Contribution_Margin_M_EUR"] = (df["Revenue_EUR"] - (df["Total_HPP_Cost_EUR"] + df["DC_Variable_OPEX_EUR"])) / 1e6

    # 3. Extract the Single Global Optimum
    best_idx = df["Contribution_Margin_M_EUR"].idxmax()
    best_row = df.loc[best_idx]
    
    optimal_mixes.append({
        "IT_Capacity_MW": cap,
        "Tier_A_GWh": best_row["Energy_A_Annual_GWh"],
        "Tier_B1_GWh": best_row["Energy_B1_Annual_GWh"],
        "Tier_B2_GWh": best_row["Energy_B2_Annual_GWh"],
        "Tier_C_GWh": best_row["Energy_C_Annual_GWh"],
        "Curtailment_GWh": best_row["Curtailment_Annual_GWh"]
    })

df_opt = pd.DataFrame(optimal_mixes)

# =============================================================================
# 3. CREATE STACKED BAR CHART VISUALIZATION
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

labels = [f"{mw:.0f} MW" for mw in df_opt['IT_Capacity_MW']]
x = np.arange(len(labels))
width = 0.55

# Convert Annual GWh to Average Power (MW)
avg_power_a = df_opt['Tier_A_GWh'].values / 8.76
avg_power_b1 = df_opt['Tier_B1_GWh'].values / 8.76
avg_power_b2 = df_opt['Tier_B2_GWh'].values / 8.76
avg_power_c = df_opt['Tier_C_GWh'].values / 8.76
avg_power_curtail = df_opt['Curtailment_GWh'].values / 8.76

bottom_tracker = np.zeros(len(labels))

# Plot Tiers (Saving the rect objects so we can attach text to them later)
p_a = ax.bar(x, avg_power_a, width, label='Tier A', color=C_A, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += avg_power_a

p_b1 = ax.bar(x, avg_power_b1, width, label='Tier B1', color=C_B1, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += avg_power_b1

p_b2 = ax.bar(x, avg_power_b2, width, label='Tier B2', color=C_B2, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += avg_power_b2

p_c = ax.bar(x, avg_power_c, width, label='Tier C', color=C_C, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += avg_power_c

p_curtail = ax.bar(x, avg_power_curtail, width, bottom=bottom_tracker, 
                   facecolor=C_WASTE_FACE, edgecolor=C_WASTE_EDGE, hatch=HATCH_STYLE, linewidth=1.5, 
                   label='Curtailment')

# =============================================================================
# 4. ADD TEXT ANNOTATIONS (THE MW LABELS)
# =============================================================================
def annotate_bars(rects, text_color='white', apply_bg=False):
    """
    Places the MW value in the center of the bar.
    apply_bg: Creates a tiny semi-transparent background box to make text readable over hatched lines.
    """
    for rect in rects:
        height = rect.get_height()
        if height > 1.5:  # Only label if the bar is tall enough to fit the text
            bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8) if apply_bg else None
            
            ax.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2),
                        ha='center', va='center', color=text_color, fontweight='bold', fontsize=9,
                        bbox=bbox_props)

# Apply labels with correct contrasting text colors
annotate_bars(p_a, text_color='white')
annotate_bars(p_b1, text_color='white')
annotate_bars(p_b2, text_color='black')
annotate_bars(p_c, text_color='black')

# Apply a text background for curtailment so the red hash lines don't ruin readability
annotate_bars(p_curtail, text_color='#333333', apply_bg=True)


# =============================================================================
# 5. CHART FORMATTING & LEGEND
# =============================================================================
ax.set_title("Data Center Energy Balance at Optimal Workload Mix", fontsize=14, pad=15, color='#333333', fontweight='bold')
ax.set_ylabel("Average Power (MW)", color='grey', fontsize=11, fontweight='bold')
ax.set_xlabel("Installed IT Capacity (MW)", color='grey', fontweight='bold', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.grid(axis='y', linestyle='-', alpha=0.3)

# Remove specific spines to match exactly
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('grey')
ax.spines['bottom'].set_color('grey')

# Match the legend style and placement exactly
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False, fontsize=10)

plt.tight_layout()

# Save and Show
save_path = os.path.join(BASE_FOLDER, 'Thesis_Energy_Balance_Optimal_Mix_Labeled.SVG')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"-> Plot saved to: {save_path}")
plt.show()