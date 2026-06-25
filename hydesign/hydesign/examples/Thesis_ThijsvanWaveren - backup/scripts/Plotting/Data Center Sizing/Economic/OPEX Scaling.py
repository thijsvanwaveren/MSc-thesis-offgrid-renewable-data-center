# -*- coding: utf-8 -*-
"""
Visualizes the variable OPEX breakdown using a stacked area chart.

Demonstrates the concept of "Zero Marginal Cost" for opportunistic workloads 
by plotting the breakdown of annual operating expenditures. Shows that while the 
base hybrid power plant costs remain relatively fixed, the variable water and 
operational costs scale dynamically with the dispatch of different workload tiers.
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# =============================================================================
# 1. MASTER PARAMETERS
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITIES_MW = [16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0] 

OTHER_VARIABLE_OPEX_EUR_PER_MWH = 3  # Water / Minor Variable OPEX

# Academic-Consulting Color Palette
C_HPP = '#94a3b8'        # Professional Slate Grey (Visible, solid, but not overpowering)
C_CORE = '#2b6cb0'       # Editorial Steel Blue (Core committed OPEX)
C_TIER_C = '#e28743'     # Muted Terracotta / Orange (Opportunistic extra OPEX)

# =============================================================================
# 2. DATA EXTRACTION
# =============================================================================
var_cost_data = []

for cap in IT_CAPACITIES_MW:
    file_name = f"Feasible_3D_Sweep_Results_99.9pct_IT{cap:.1f}.csv"
    file_path = os.path.join(BASE_FOLDER, file_name)
    df = pd.read_csv(file_path)
   

    for col in df.columns:
        if "Energy" in col and "MWh" in col:
            df[col.replace("_MWh", "_GWh")] = df[col] / 1000.0

    # 1. Extract Fixed HPP Electricity Cost (Tier A Anchor)
    tier_a_only = df[(df["Tier_B1_MW"] == 0) & (df["Tier_B2_MW"] == 0) & (df["Tier_A_MW"] > 0)].copy()
    if not tier_a_only.empty:
        total_delivered_base = (tier_a_only.get('Energy_A_Annual_GWh', 0) + 
                                tier_a_only.get('Energy_B1_Annual_GWh', 0) + 
                                tier_a_only.get('Energy_B2_Annual_GWh', 0) + 
                                tier_a_only.get('Energy_C_Annual_GWh', 0)) * 1000.0
        tier_a_only["HPP_Base_Cost"] = tier_a_only["LCOE_delivered"] * total_delivered_base
        hpp_sunk_cost_m = tier_a_only["HPP_Base_Cost"].mean() / 1e6
    else:
        hpp_sunk_cost_m = 15.95 # Fallback

    df["Total_HPP_Cost_EUR"] = hpp_sunk_cost_m * 1e6

    # 2. Extract Water Costs for the Optimal Mix
    e_a = df.get('Energy_A_Annual_GWh', 0)
    e_b1 = df.get('Energy_B1_Annual_GWh', 0)
    e_b2 = df.get('Energy_B2_Annual_GWh', 0)
    e_total_del = df.get('Total_Delivered_Annual_GWh', e_a + e_b1 + e_b2)
    e_c = np.maximum(0, e_total_del - (e_a + e_b1 + e_b2))

    planned_mwh = (df["Tier_A_MW"] + df["Tier_B1_MW"] + df["Tier_B2_MW"]) * 8760.0
    df["DC_Variable_OPEX_EUR"] = OTHER_VARIABLE_OPEX_EUR_PER_MWH * planned_mwh
    
    df["Revenue_No_C_EUR"] = (e_a * 4000 + e_b1 * 2800 + e_b2 * 1600) * 1000
    df["CM_M"] = (df["Revenue_No_C_EUR"] - (df["Total_HPP_Cost_EUR"] + df["DC_Variable_OPEX_EUR"])) / 1e6
    
    best_idx = df["CM_M"].idxmax()
    best_row = df.loc[best_idx]

    water_core_m = (planned_mwh.loc[best_idx]) * OTHER_VARIABLE_OPEX_EUR_PER_MWH / 1e6
    water_tier_c_m = (e_c.loc[best_idx] * 1000.0) * OTHER_VARIABLE_OPEX_EUR_PER_MWH / 1e6

    var_cost_data.append({
        "IT_Capacity_MW": cap,
        "HPP_Elec_Cost": hpp_sunk_cost_m,
        "Water_Core": water_core_m,
        "Water_Tier_C": water_tier_c_m
    })

df_vc = pd.DataFrame(var_cost_data)

# =============================================================================
# 3. CREATE VISUALIZATION (Stacked Area Chart)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

x = df_vc['IT_Capacity_MW'].values
y_elec = df_vc['HPP_Elec_Cost'].values
y_core = df_vc['Water_Core'].values
y_tier_c = df_vc['Water_Tier_C'].values
y_total = y_elec + y_core + y_tier_c

# Create Stacked Area
ax.stackplot(
    x, y_elec, y_core, y_tier_c,
    colors=[C_HPP, C_CORE, C_TIER_C],
    edgecolor='white',
    linewidth=0.8,
    alpha=0.95,
    zorder=2
)

# Subtle Markers on the top edge
ax.plot(x, y_total, color='#555555', marker='o', markersize=5, linewidth=0, zorder=4, alpha=0.8)

# Formatting & Titles
ax.set_ylabel("Annual Cost (Millions € / yr)", fontsize=11, fontweight='bold', color='#222222')
ax.set_xlabel("Installed IT Capacity (MW$_{{IT}}$)", fontsize=11, fontweight='bold', color='#222222')

# Proportional continuous X-axis
ax.set_xlim(min(x), max(x))
ax.set_xticks(x)
ax.set_xticklabels([f"{cap:.0f}" for cap in x], fontsize=10, color='#222222')

# Set Axis Limits dynamically based on max total height
ax.set_ylim(0, max(y_total) * 1.15) 

# Minimalist Grid & Bold Spines
ax.grid(axis='y', linestyle='-', alpha=0.3, color='#b0b0b0', zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('#222222')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('#222222')
ax.tick_params(axis='y', colors='#222222', labelsize=10)

# Legend: Reordered to match the visual stack (Bottom -> Middle -> Top)
legend_handles = [
    mpatches.Patch(facecolor=C_HPP, label='HPP Electricity Generation'),
    mpatches.Patch(facecolor=C_CORE, label='Committed Workload OPEX (Water)'),
    mpatches.Patch(facecolor=C_TIER_C, label='Tier C Opportunistic OPEX (Water)')
]

ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
          ncol=3, frameon=False, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2) # Give extra room for the legend

# Save & Show
save_path = os.path.join(BASE_FOLDER, 'variable_cost_area_refined.svg')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()