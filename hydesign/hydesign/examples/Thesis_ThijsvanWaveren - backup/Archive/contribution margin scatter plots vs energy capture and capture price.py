# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:19:01 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.4: Optimizing the Workload Portfolio (Micro-Economics)
Generates 2x3 grid visualizations analyzing workload mix trade-offs:
1. Energy Delivered vs. Contribution Margin
2. Energy Delivered vs. Average Capture Price
Color metric: Average Deferral Time (Days) - measures portfolio flexibility.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. MASTER USER INPUTS & ASSUMPTIONS
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITIES_MW = [16.0, 20.0, 30.0, 40.0, 50.0, 75.0] 

HOURS_PER_YEAR = 8760
REVENUE_PER_MWH = {"A": 3000.0, "B1": 2100.0, "B2": 1200.0} # Tier C excluded
OTHER_VARIABLE_OPEX_EUR_PER_MWH = 5.50 

# =============================================================================
# 2. DATA AGGREGATION & PROCESSING
# =============================================================================
all_scenarios = {}

print("Processing data and recalculating financials without Tier C...")

for cap in IT_CAPACITIES_MW:
    file_name = f"Feasible_3D_Sweep_Results_99.9pct_IT{cap:.1f}.csv"
    file_path = os.path.join(BASE_FOLDER, file_name)
    
    if not os.path.exists(file_path):
        print(f"  [WARNING] File not found, generating synthetic data for IT {cap} MW")
        df = pd.DataFrame({
            "Total_Delivered_Annual_GWh": np.random.uniform(30, cap*8.76*0.9, 1000),
            "Energy_A_Annual_GWh": np.random.uniform(0, 8*8.76, 1000),
            "Energy_B1_Annual_GWh": np.random.uniform(0, cap*8.76*0.3, 1000),
            "Energy_B2_Annual_GWh": np.random.uniform(0, cap*8.76*0.5, 1000),
            "LCOE_delivered": np.random.uniform(40, 90, 1000)
        })
        df["Energy_B2_Annual_GWh"] = df["Total_Delivered_Annual_GWh"] * 0.4 
    else:
        df = pd.read_csv(file_path)

    # Convert GWh to MWh
    for col in ["Energy_A_Annual_GWh", "Energy_B1_Annual_GWh", "Energy_B2_Annual_GWh", "Total_Delivered_Annual_GWh"]:
        if col in df.columns:
            df[col.replace("_GWh", "_MWh")] = df[col] * 1000.0

    # 1. Committed Energy (No Tier C)
    df["Energy_Delivered_No_C_MWh"] = df["Energy_A_Annual_MWh"] + df["Energy_B1_Annual_MWh"] + df["Energy_B2_Annual_MWh"]
    df["Energy_Delivered_No_C_GWh"] = df["Energy_Delivered_No_C_MWh"] / 1000.0

    # 2. Financials (No Tier C)
    df["Revenue_No_C_EUR"] = (df["Energy_A_Annual_MWh"] * REVENUE_PER_MWH["A"] + 
                              df["Energy_B1_Annual_MWh"] * REVENUE_PER_MWH["B1"] + 
                              df["Energy_B2_Annual_MWh"] * REVENUE_PER_MWH["B2"])

    df["Total_HPP_Cost_EUR"] = df["LCOE_delivered"] * df["Total_Delivered_Annual_MWh"] 
    df["DC_Variable_OPEX_No_C_EUR"] = OTHER_VARIABLE_OPEX_EUR_PER_MWH * df["Energy_Delivered_No_C_MWh"]
    df["Total_Variable_OPEX_No_C_EUR"] = df["Total_HPP_Cost_EUR"] + df["DC_Variable_OPEX_No_C_EUR"]

    # 3. Margins & Unit Economics
    df["Contribution_Margin_No_C_M_EUR"] = (df["Revenue_No_C_EUR"] - df["Total_Variable_OPEX_No_C_EUR"]) / 1e6
    df["Average_Capture_Price_No_C"] = np.where(df["Energy_Delivered_No_C_MWh"] > 0, 
                                                df["Revenue_No_C_EUR"] / df["Energy_Delivered_No_C_MWh"], 0)
    
    # 4. Color Metric: Average Deferral Time (Days) -> Tier A=0, B1=1, B2=7
    df["Avg_Deferral_Days"] = np.where(
        df["Energy_Delivered_No_C_MWh"] > 0, 
        (df["Energy_A_Annual_MWh"] * 0 + df["Energy_B1_Annual_MWh"] * 1 + df["Energy_B2_Annual_MWh"] * 7) / df["Energy_Delivered_No_C_MWh"], 
        0
    )

    all_scenarios[cap] = df

# =============================================================================
# 3. MASTER PLOTTING FUNCTION
# =============================================================================
def plot_pareto_grid(x_col, y_col, x_label, y_label, title, filename):
    """
    Generates a 2x3 grid of scatter plots with an exact upper envelope 
    and high-contrast color-coding locked between 0 and 7 days.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False, facecolor='white')
    axes = axes.flatten()
    
    # 'plasma' colormap: Dark Purple (Rigid/0 Days) -> Bright Yellow (Flexible/~7 Days)
    cmap = plt.get_cmap('plasma')

    for idx, cap in enumerate(IT_CAPACITIES_MW):
        ax = axes[idx]
        df = all_scenarios[cap].copy()
        
        # 1. Scatter Cloud
        # Locked vmin=0, vmax=7 for consistent coloring across all subplots
        scatter = ax.scatter(df[x_col], df[y_col], 
                             c=df["Avg_Deferral_Days"], 
                             cmap=cmap, s=15, alpha=0.8, edgecolors='none', vmin=0, vmax=7)

        # 2. EXACT Upper Envelope (Pareto Front)
        bin_edges = np.linspace(df[x_col].min(), df[x_col].max(), 25)
        df['Bin'] = pd.cut(df[x_col], bins=bin_edges)
        
        # observed=True prevents pandas ValueError for empty bins
        idx_max = df.groupby('Bin', observed=True)[y_col].idxmax().dropna()
        upper_bound = df.loc[idx_max].sort_values(by=x_col)
        
        ax.plot(upper_bound[x_col], upper_bound[y_col], 
                color='#e74c3c', linewidth=2.5, label='Pareto Frontier\n(Exact Maxima)')

        # 3. Highlight Absolute Optimum (Max Contribution Margin)
        optimum_idx = df["Contribution_Margin_No_C_M_EUR"].idxmax()
        ax.scatter(df.loc[optimum_idx, x_col], df.loc[optimum_idx, y_col], 
                   color='none', edgecolors='#e74c3c', s=100, linewidth=2, zorder=5, label='Max Contribution Margin')

        # 4. Subplot Formatting
        ax.set_title(f"IT Capacity: {cap} MW", fontsize=12, fontweight='bold', color='#333333')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            
        if idx >= 3: ax.set_xlabel(x_label, fontweight='bold', color='grey')
        if idx % 3 == 0: ax.set_ylabel(y_label, fontweight='bold', color='grey')
        if idx == 0: ax.legend(loc='lower left', fontsize=9)

    # Master Formatting
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02, color='#333333')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Average Deferral Time of Portfolio (Days)', rotation=270, labelpad=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save and Show
    save_path = os.path.join(BASE_FOLDER, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 4. GENERATE THE PLOTS
# =============================================================================

print("Generating visualizations...")

# PLOT 1: Energy Delivered vs Contribution Margin
plot_pareto_grid(
    x_col="Energy_Delivered_No_C_GWh", 
    y_col="Contribution_Margin_No_C_M_EUR", 
    x_label="Committed Energy Delivered (GWh/yr)", 
    y_label="Contribution Margin (M€/yr)", 
    title="Economic Envelope: Energy Capture vs. Contribution Margin", 
    filename="Thesis_Pareto_Margin_vs_Energy_Grid.SVG"
)

# PLOT 2: Energy Delivered vs Average Capture Price
plot_pareto_grid(
    x_col="Energy_Delivered_No_C_GWh", 
    y_col="Average_Capture_Price_No_C", 
    x_label="Committed Energy Delivered (GWh/yr)", 
    y_label="Average Capture Price (€/MWh)", 
    title="Pricing Frontier: Energy Volume vs. Average Capture Price", 
    filename="Thesis_Pareto_Price_vs_Energy_Grid.SVG"
)

print("All plots successfully generated!")