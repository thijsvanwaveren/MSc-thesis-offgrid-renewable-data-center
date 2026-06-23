# -*- coding: utf-8 -*-
"""
Section 3.4 - Data Center Workload Optimization
Unified Script: Generates both the Energy Balance and Hardware Efficiency plots.
(Updated with Tier C Revenue and Explicit Tier A Tie-Breaker Logic)
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

# Financial Assumptions (Added Tier C at 400 EUR/MWh)
REVENUE_PER_MWH = {"A": 4000, "B1": 2800, "B2": 1600, "C": 416.72}
OTHER_VARIABLE_OPEX_EUR_PER_MWH = 3

# Academic-Consulting Color Palette
C_A = '#08306b'        # Deep Navy (Tier A)
C_B1 = '#2879b9'       # Strong Blue (Tier B1)
C_B2 = '#73b3d8'       # Soft Blue (Tier B2)
C_C = '#c8ddf0'        # Pale Blue (Tier C)

# Refined Inefficiency Styling
C_WASTE_FACE = '#fdf4f4'       
C_WASTE_EDGE = '#e8b7b2'       
HATCH_CURTAIL = '////'         

C_UNUTILIZED_FACE = '#f8f9f9'  
C_UNUTILIZED_EDGE = '#a6acaf'  
HATCH_UNUTILIZED = '////'      

# =============================================================================
# 2. DATA PROCESSING & OPTIMIZATION ENGINE
# =============================================================================
optimal_mixes = []
print("Scanning for optimal energy balances and hardware utilization...")

for cap in IT_CAPACITIES_MW:
    file_name = f"Feasible_3D_Sweep_Results_99.9pct_IT{cap:.1f}.csv"
    file_path = os.path.join(BASE_FOLDER, file_name)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # 1. Standardize Columns to GWh 
        for col in list(df.columns):
            if "MWh" in col:
                df[col.replace("_MWh", "_GWh")] = df[col] / 1000.0
    else:
        # Synthetic fallback
        print(f"Creating synthetic data for {cap} MW")
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
                            'Total_Produced_Annual_GWh': 61.0 * 8.76, 
                            'LCOE_delivered': 50.0
                        })
        df = pd.DataFrame(rows)

    # Extract Core Energy 
    e_a = df.get('Energy_A_Annual_GWh', 0)
    e_b1 = df.get('Energy_B1_Annual_GWh', 0)
    e_b2 = df.get('Energy_B2_Annual_GWh', 0)
    e_total_del = df.get('Total_Delivered_Annual_GWh', e_a + e_b1 + e_b2)
    
    # Calculate Tier C
    if 'Energy_C_Annual_GWh' not in df.columns:
        df['Energy_C_Annual_GWh'] = np.maximum(0, e_total_del - (e_a + e_b1 + e_b2))
    e_c = df['Energy_C_Annual_GWh']
    
    # Robust Curtailment Extraction 
    if 'Curtailment_Annual_GWh' in df.columns:
        curtailment = df['Curtailment_Annual_GWh']
    elif 'Curtailment_GWh' in df.columns:
        curtailment = df['Curtailment_GWh']
    elif 'Total_Produced_Annual_GWh' in df.columns:
        curtailment = df['Total_Produced_Annual_GWh'] - e_total_del
    else:
        curtailment = (61.0 * 8.76) - e_total_del
        
    df['Curtailment_Annual_GWh'] = np.maximum(0, curtailment)
    
    # Financials to find the optimum mix
    rev_a = e_a * 1000.0 * REVENUE_PER_MWH["A"]
    rev_b1 = e_b1 * 1000.0 * REVENUE_PER_MWH["B1"]
    rev_b2 = e_b2 * 1000.0 * REVENUE_PER_MWH["B2"]
    rev_c = e_c * 1000.0 * REVENUE_PER_MWH["C"] # NEW: Tier C Revenue included
    
    # OPEX based on ACTUAL energy processed (as per Fig 3.15 methodology)
    actual_mwh_processed = (e_a + e_b1 + e_b2 + e_c) * 1000.0
    dc_var_opex = OTHER_VARIABLE_OPEX_EUR_PER_MWH * actual_mwh_processed
    
    # Base HPP anchor (Simulated baseline sunk cost)
    hpp_sunk = 15.95e6 
    
    # Calculate total Contribution Margin
    df["Contribution_Margin_M_EUR"] = ((rev_a + rev_b1 + rev_b2 + rev_c) - (hpp_sunk + dc_var_opex)) / 1e6
    
    # --- NEW: EXPLICIT TIE-BREAKER LOGIC ---
    # Find the maximum margin (with a tiny tolerance for floating point rounding)
    max_margin = df["Contribution_Margin_M_EUR"].max()
    top_candidates = df[df["Contribution_Margin_M_EUR"] >= max_margin - 1e-6]
    
    # If there are degenerate solutions, sort them to maximize Tier A, then Tier B1
    best_row = top_candidates.sort_values(by=["Tier_A_MW", "Tier_B1_MW"], ascending=[False, False]).iloc[0]
    
    optimal_mixes.append({
        "IT_Capacity_MW": cap,
        "Tier_A_MW": best_row["Energy_A_Annual_GWh"] / 8.76,
        "Tier_B1_MW": best_row["Energy_B1_Annual_GWh"] / 8.76,
        "Tier_B2_MW": best_row["Energy_B2_Annual_GWh"] / 8.76,
        "Tier_C_MW": best_row["Energy_C_Annual_GWh"] / 8.76,
        "Curtailment_MW": best_row["Curtailment_Annual_GWh"] / 8.76
    })

df_opt = pd.DataFrame(optimal_mixes)

# Calculate Unutilized IT Capacity
df_opt['Utilized_MW'] = df_opt['Tier_A_MW'] + df_opt['Tier_B1_MW'] + df_opt['Tier_B2_MW'] + df_opt['Tier_C_MW']
df_opt['Unutilized_MW'] = df_opt['IT_Capacity_MW'] - df_opt['Utilized_MW']

# Formatting logic
labels = [f"{mw:.0f} MW" for mw in df_opt['IT_Capacity_MW']]
x = np.arange(len(labels))
width = 0.45

def annotate_bars(ax, rects, text_color='white', apply_bg=False):
    """Places the MW value in the center of the bar."""
    for rect in rects:
        height = rect.get_height()
        if height > 1.5:  
            bbox_props = dict(boxstyle="round,pad=0.2", facecolor=C_WASTE_FACE, edgecolor="none", alpha=0.8) if apply_bg else None
            ax.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2),
                        ha='center', va='center', color=text_color, fontweight='bold', fontsize=9,
                        bbox=bbox_props)

def clean_axes(ax):
    """Applies academic-consulting spine formatting."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

# =============================================================================
# 3. PLOT 1: THE ENERGY BALANCE (HPP PERSPECTIVE)
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor='white')
bottom_tracker = np.zeros(len(labels))

p_a = ax1.bar(x, df_opt['Tier_A_MW'], width, label='Tier A', color=C_A, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += df_opt['Tier_A_MW']

p_b1 = ax1.bar(x, df_opt['Tier_B1_MW'], width, label='Tier B1', color=C_B1, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += df_opt['Tier_B1_MW']

p_b2 = ax1.bar(x, df_opt['Tier_B2_MW'], width, label='Tier B2', color=C_B2, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += df_opt['Tier_B2_MW']

p_c = ax1.bar(x, df_opt['Tier_C_MW'], width, label='Tier C', color=C_C, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += df_opt['Tier_C_MW']

p_curtail = ax1.bar(x, df_opt['Curtailment_MW'], width, bottom=bottom_tracker, 
                    facecolor=C_WASTE_FACE, edgecolor=C_WASTE_EDGE, hatch=HATCH_CURTAIL, linewidth=1.5, label='Curtailment')

annotate_bars(ax1, p_a, 'white')
annotate_bars(ax1, p_b1, 'white')
annotate_bars(ax1, p_b2, 'black')
annotate_bars(ax1, p_c, 'black')
annotate_bars(ax1, p_curtail, '#333333', apply_bg=True)

#ax1.set_title("Data Center Energy Balance", fontsize=15, pad=15, color='#333333', fontweight='bold')
ax1.set_ylabel("Average Power (MW)", color='#444444', fontsize=12, fontweight='bold')
ax1.set_xlabel("Installed IT Capacity (MW)", color='#444444', fontweight='bold', fontsize=12)
clean_axes(ax1)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, frameon=False, fontsize=10)

plt.tight_layout()
save_path_1 = os.path.join(BASE_FOLDER, 'Thesis_Energy_Balance_Optimal_Mix_Labeled_adjusted.svg')
fig1.savefig(save_path_1, dpi=300, bbox_inches='tight')
print(f"✅ Plot 1 saved to: {save_path_1}")

# =============================================================================
# 4. PLOT 2: HARDWARE EFFICIENCY (DC PERSPECTIVE)
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6.5), facecolor='white')
bottom_tracker = np.zeros(len(labels))

p_a2 = ax2.bar(x, df_opt['Tier_A_MW'], width, label='Tier A', color=C_A, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += df_opt['Tier_A_MW']

p_b1_2 = ax2.bar(x, df_opt['Tier_B1_MW'], width, label='Tier B1', color=C_B1, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += df_opt['Tier_B1_MW']

p_b2_2 = ax2.bar(x, df_opt['Tier_B2_MW'], width, label='Tier B2', color=C_B2, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += df_opt['Tier_B2_MW']

p_c2 = ax2.bar(x, df_opt['Tier_C_MW'], width, label='Tier C', color=C_C, bottom=bottom_tracker, edgecolor='white', linewidth=0.5)
bottom_tracker += df_opt['Tier_C_MW']

p_unutilized = ax2.bar(x, df_opt['Unutilized_MW'], width, bottom=bottom_tracker, 
       facecolor=C_UNUTILIZED_FACE, edgecolor=C_UNUTILIZED_EDGE, hatch=HATCH_UNUTILIZED, linewidth=1.2, 
       label='Unutilized Hardware')

# Explicitly label the stranded capacity
for rect in p_unutilized:
    height = rect.get_height()
    if height > 2: 
        ax2.annotate(f'{height:.0f} MW',
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2 ),
                    ha='center', va='center', color='#555555', fontweight='bold', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.0", facecolor=C_UNUTILIZED_FACE, edgecolor="none", alpha=0.9))
        
#ax2.set_title("Data Center Hardware Efficiency", fontsize=15, pad=20, color='#333333', fontweight='bold')
ax2.set_ylabel("Average Power (MW)", color='#444444', fontsize=12, fontweight='bold')
ax2.set_xlabel("Installed IT Capacity (MW)", color='#444444', fontweight='bold', fontsize=12)
ax2.set_ylim(0, 105)
clean_axes(ax2)

# Force 5 columns to perfectly match the top plot's legend structure
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False, fontsize=10)
plt.tight_layout()
save_path_2 = os.path.join(BASE_FOLDER, 'Thesis_Hardware_Efficiency_Optimal_Mix_adjusted.svg')
fig2.savefig(save_path_2, dpi=300, bbox_inches='tight')
print(f"✅ Plot 2 saved to: {save_path_2}")

# Display both plots
plt.show()