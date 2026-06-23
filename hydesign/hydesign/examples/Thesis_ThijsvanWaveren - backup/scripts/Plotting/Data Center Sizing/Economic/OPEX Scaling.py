# -*- coding: utf-8 -*-
"""
Section 3.5 - The Zero Marginal Cost of Opportunistic Workloads
Visualizes the Variable OPEX breakdown using a Stacked Bar Chart.
(Refined Academic Consulting Standard with Integrated Base)
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

OTHER_VARIABLE_OPEX_EUR_PER_MWH = 3  # Water / Minor Variable OPEX

# Academic-Consulting Color Palette
C_HPP = '#94a3b8'        # Professional Slate Grey (Visible, solid, but not overpowering)
C_HPP_EDGE = '#64748b'   # Darker slate for crisp definition
C_CORE = '#2b6cb0'       # Editorial Steel Blue (Core committed OPEX)
C_TIER_C = '#e28743'     # Muted Terracotta / Orange (Opportunistic extra OPEX)

# =============================================================================
# 2. DATA EXTRACTION
# =============================================================================
var_cost_data = []
print("Calculating Variable Cost Breakdown (Refined Bar Chart)...")

for cap in IT_CAPACITIES_MW:
    file_name = f"Feasible_3D_Sweep_Results_99.9pct_IT{cap:.1f}.csv"
    file_path = os.path.join(BASE_FOLDER, file_name)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Synthetic fallback for demonstration
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
# 3. CREATE VISUALIZATION (Stacked Bar Chart)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='white')

x = df_vc['IT_Capacity_MW'].values
y_elec = df_vc['HPP_Elec_Cost'].values
y_core = df_vc['Water_Core'].values
y_tier_c = df_vc['Water_Tier_C'].values
y_total = y_elec + y_core + y_tier_c

# Use a categorical X-axis for evenly spaced bars
x_pos = np.arange(len(x))
bar_width = 0.65

# Stack 1: HPP Generation Cost (Now a solid, clear part of the stack)
ax.bar(x_pos, y_elec, width=bar_width, color=C_HPP, edgecolor=C_HPP_EDGE, 
       linewidth=1.0, label='HPP Electricity Generation', zorder=3)

# Stack 2: Core Workload Variable OPEX
ax.bar(x_pos, y_core, width=bar_width, bottom=y_elec, color=C_CORE, 
       edgecolor='#1a426b', linewidth=1.0, label='Committed Workload OPEX (Water)', zorder=3)

# Stack 3: Tier C Variable OPEX
ax.bar(x_pos, y_tier_c, width=bar_width, bottom=y_elec+y_core, color=C_TIER_C, 
       edgecolor='#a05d2c', linewidth=1.0, label='Tier C Opportunistic OPEX (Water)', zorder=3)

# Annotate totals neatly on top of each bar
for i in range(len(x_pos)):
    ax.annotate(f'€{y_total[i]:.1f}M', 
                xy=(x_pos[i], y_total[i]), 
                xytext=(0, 6), # Offset slightly above the bar
                textcoords="offset points", 
                ha='center', va='bottom', 
                fontsize=9.5, fontweight='bold', color='#222222')

# Formatting & Titles (No Main Title)
ax.set_ylabel("Annual Cost (Millions € / yr)", fontsize=11, fontweight='bold', color='#222222')
ax.set_xlabel("Data Center Installed IT Capacity", fontsize=11, fontweight='bold', color='#222222', labelpad=10)

# Clean proportional ticks corresponding to actual data points
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{cap:.0f} MW" for cap in x], fontsize=10, fontweight='bold', color='#222222')

# Set Axis Limits dynamically based on max total height
ax.set_ylim(0, max(y_total) * 1.25) # Extra headroom for legend and labels

# Minimalist Grid & Bold Spines
ax.grid(axis='y', linestyle='-', alpha=0.3, color='#b0b0b0', zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('#222222')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('#222222')
ax.tick_params(axis='y', colors='#222222', labelsize=10)
ax.tick_params(axis='x', length=0) # Remove x-axis tick marks for cleaner look

# Legend: Reverse order to perfectly match the visual top-to-bottom stack
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=10, 
          frameon=False, borderpad=1)

plt.tight_layout()
save_path = os.path.join(BASE_FOLDER, 'variable_cost_bar_refined.svg')
if os.path.exists(BASE_FOLDER):
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"-> Plot saved to: {save_path}")
plt.show()