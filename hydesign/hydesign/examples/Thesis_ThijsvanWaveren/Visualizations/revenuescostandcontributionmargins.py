# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:19:37 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.4 - Economic Story Arc
Generates 3 sequential scatter plots: Gross Revenue, Variable Cost, Contribution Margin.
Executes twice: Once excluding Tier C economics, and once including Tier C economics.
Maintains true, dynamic LCOE-based HPP costs to preserve real simulation spread.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import warnings
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# =============================================================================
# 1. USER INPUTS & PARAMETERS
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITY = 16.0  

# Financial Assumptions
REVENUE_PER_MWH = {"A": 4000.0, "B1": 2800.0, "B2": 1600.0, "C": 400.0}
OTHER_VARIABLE_OPEX_EUR_PER_MWH = 3

FILE_NAME = f"Feasible_3D_Sweep_Results_99.9pct_IT{IT_CAPACITY:.1f}.csv"
file_path = os.path.join(BASE_FOLDER, FILE_NAME)

# =============================================================================
# 2. LOAD DATA & EXTRACT NATIVE SIMULATION VOLUMES
# =============================================================================
print(f"Loading data from: {FILE_NAME}...")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find the file at {file_path}. Please check the path.")

df_raw = pd.read_csv(file_path)

# Ensure columns are in MWh
for col in ["Energy_A_Annual_GWh", "Energy_B1_Annual_GWh", "Energy_B2_Annual_GWh", "Energy_C_Annual_GWh", "Total_Delivered_Annual_GWh"]:
    if col in df_raw.columns:
        df_raw[col.replace("_GWh", "_MWh")] = df_raw[col] * 1000.0

# Extract physical energy volumes safely
vol_A = df_raw.get("Energy_A_Annual_MWh", 0)
vol_B1 = df_raw.get("Energy_B1_Annual_MWh", 0)
vol_B2 = df_raw.get("Energy_B2_Annual_MWh", 0)
vol_C = df_raw.get("Energy_C_Annual_MWh", 0) 

# =============================================================================
# 3. DERIVE FIXED HPP BASELINE (Tier A 1-8 MW Anchor)
# =============================================================================
# We isolate mixes that are strictly Tier A (B1=0, B2=0) from 1 to 8 MW to find the clean cost baseline
df_raw["Portfolio_Power_MW"] = (vol_A + vol_B1 + vol_B2) / 8760.0
tier_a_only = df_raw[(df_raw["Tier_B1_MW"] == 0) & (df_raw["Tier_B2_MW"] == 0) & (df_raw["Tier_A_MW"] > 0) & (df_raw["Tier_A_MW"] <= 8)].copy()

# HPP Cost = LCOE * Total energy delivered in those specific clean simulation runs
tier_a_only["HPP_Base_Cost"] = tier_a_only["LCOE_delivered"] * (tier_a_only["Energy_A_Annual_MWh"] + tier_a_only["Energy_C_Annual_MWh"])

# Average baseline cost establishes our Ground Truth (approx 15.9M)
FIXED_HPP_BASELINE_EUR = tier_a_only["HPP_Base_Cost"].mean()

print(f"Fixed HPP Baseline derived from Tier A (1-8MW): €{FIXED_HPP_BASELINE_EUR/1e6:.3f} Million")

# =============================================================================
# 4. DEFINE DISTINCT COLORMAPS
# =============================================================================
cmap_revenue = mcolors.LinearSegmentedColormap.from_list("Rev_cmap", ["#c8ddf0", "#4b7ab5", "#08306b"])
cmap_cost = mcolors.LinearSegmentedColormap.from_list("Cost_cmap", ["#d4e6df", "#569e76", "#184f2d"])
cmap_margin = mcolors.LinearSegmentedColormap.from_list("Margin_cmap", ["#e6d4e1", "#946386", "#4a1e3e"])

# =============================================================================
# 5. MASTER PLOTTING FUNCTION
# =============================================================================
def generate_scatter_plot(df_plot, y_col, title, ylabel, filename, colormap, highlight_max=False, is_cost_plot=False):
    fig, ax = plt.subplots(figsize=(15, 7), facecolor='white')

    # Scatter Cloud (X-Axis is ALWAYS A+B1+B2, representing the core portfolio)
    if is_cost_plot:
        scatter = ax.scatter(df_plot["Portfolio_Power_MW"], 
                             df_plot[y_col], 
                             color='#184f2d', s=35, alpha=0.8, edgecolors='none', zorder=2)
    else:
        scatter = ax.scatter(df_plot["Portfolio_Power_MW"], 
                             df_plot[y_col], 
                             c=df_plot["Tier_A_MW"], 
                             cmap=colormap, s=35, alpha=0.8, edgecolors='none', 
                             vmin=0, vmax=8, zorder=2)

    if highlight_max:
        opt_idx = df_plot[y_col].idxmax()
        opt_x = df_plot.loc[opt_idx, "Portfolio_Power_MW"]
        opt_y = df_plot.loc[opt_idx, y_col]
        opt_a = df_plot.loc[opt_idx, "Tier_A_MW"]
        opt_b1 = df_plot.loc[opt_idx, "Tier_B1_MW"]
        opt_b2 = df_plot.loc[opt_idx, "Tier_B2_MW"]

        ax.scatter(opt_x, opt_y, facecolors='none', edgecolors='#e74c3c', s=200, linewidth=2.5, zorder=4)

        box_title = "Peak Revenue" if "Revenue" in ylabel else "Optimal Mix"

        annotation_text = (f"{box_title}:\n"
                           f"A: {int(opt_a)} MW\n"
                           f"B1: {int(opt_b1)} MW\n"
                           f"B2: {int(opt_b2)} MW")

        ax.annotate(annotation_text, 
                    xy=(opt_x, opt_y), 
                    xytext=(opt_x+0.3, opt_y - (max(df_plot[y_col])*0.05)), 
                    arrowprops=dict(facecolor='black', arrowstyle="-|>", lw=1.5, color='black'),
                    fontsize=10, fontweight='bold', color='#333333',
                    bbox=dict(facecolor='white', edgecolor='#e74c3c', alpha=0.9, pad=0.4))

    ax.set_title(title, fontsize=14, fontweight='bold', color='#333333', pad=15)
    ax.set_xlabel("Total Workload Volume Served (Average MW)", fontweight='bold', color='#444444', fontsize=12)
    ax.set_ylabel(ylabel, fontweight='bold', color='#444444', fontsize=12)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if not is_cost_plot:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Allocated Tier A Capacity (MW)', rotation=270, labelpad=15, fontsize=11, fontweight='bold', color='#444444')
        cbar.set_ticks(np.arange(0, 9, 1))

    plt.tight_layout()
    save_path = os.path.join(BASE_FOLDER, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filename}")

# =============================================================================
# 6. EXECUTE PLOTS (LOOP FOR BOTH MODES)
# =============================================================================

modes = [
    {"include_tier_c": False, "file_suffix": "_No_TierC", "title_suffix": " (Excl. Tier C)"},
    {"include_tier_c": True,  "file_suffix": "_With_TierC", "title_suffix": " (Incl. Tier C)"}
]

for mode in modes:
    print(f"\n--- Generating Plots for Mode: {mode['file_suffix']} ---")
    
    df = df_raw.copy()
    
    # 1. Define the X-Axis (ALWAYS A + B1 + B2) to show portfolio spread
    df["Portfolio_Power_MW"] = (vol_A + vol_B1 + vol_B2) / 8760.0
    df = df[df["Portfolio_Power_MW"] > 0] # Filter zeroes
    
    # 2. Calculate Economics based on Mode
    if mode["include_tier_c"]:
        # Revenue includes Tier C
        revenue_eur = (vol_A * REVENUE_PER_MWH["A"] + vol_B1 * REVENUE_PER_MWH["B1"] + 
                       vol_B2 * REVENUE_PER_MWH["B2"] + vol_C * REVENUE_PER_MWH["C"])
        c_vol = vol_C[df.index]
    else:
        # Revenue excludes Tier C
        revenue_eur = (vol_A * REVENUE_PER_MWH["A"] + vol_B1 * REVENUE_PER_MWH["B1"] + 
                       vol_B2 * REVENUE_PER_MWH["B2"])
        c_vol = 0

    # 3. Compile Final Metrics
    df["Revenue_M_EUR"] = revenue_eur / 1e6
    
    # Linear Cost Calculation: Fixed Baseline + (Total Volume * 3 EUR/MWh)
    df["Total_Variable_OPEX_M_EUR"] = (FIXED_HPP_BASELINE_EUR + (OTHER_VARIABLE_OPEX_EUR_PER_MWH * (df["Portfolio_Power_MW"] * 8760 + c_vol))) / 1e6
    
    df["Contribution_Margin_M_EUR"] = df["Revenue_M_EUR"] - df["Total_Variable_OPEX_M_EUR"]

    # ---------------- GENERATE THE 3 PLOTS ----------------
    
    # Plot 1: Revenue
    generate_scatter_plot(df, 
                          y_col="Revenue_M_EUR", 
                          title=f"Gross Revenues ({int(IT_CAPACITY)} MW Data Center)", 
                          ylabel="Gross Annual Revenue (Million €/yr)", 
                          filename=f"Thesis_1_Gross_Revenue_{IT_CAPACITY}MW{mode['file_suffix']}.svg", 
                          colormap=cmap_revenue,
                          highlight_max=True)

    # Plot 2: Cost (is_cost_plot=True applies dark green and removes colorbar)
    generate_scatter_plot(df, 
                          y_col="Total_Variable_OPEX_M_EUR", 
                          title=f"Variable OPEX ({int(IT_CAPACITY)} MW Data Center)", 
                          ylabel="Total Variable OPEX (Million €/yr)", 
                          filename=f"Thesis_2_Variable_Cost_{IT_CAPACITY}MW{mode['file_suffix']}.svg", 
                          colormap=cmap_cost,
                          highlight_max=False,
                          is_cost_plot=True)

    # Plot 3: Margin
    generate_scatter_plot(df, 
                          y_col="Contribution_Margin_M_EUR", 
                          title=f"Contribution Margin ({int(IT_CAPACITY)} MW Data Center)", 
                          ylabel="Total Contribution Margin (Million €/yr)", 
                          filename=f"Thesis_3_Contribution_Margin_{IT_CAPACITY}MW{mode['file_suffix']}.svg", 
                          colormap=cmap_margin,
                          highlight_max=True)

print("\nAll 6 plots generated successfully!")