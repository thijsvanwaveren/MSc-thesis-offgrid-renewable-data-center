# -*- coding: utf-8 -*-
"""
Individual Efficiency Frontiers (Strictly Excluding Tier C)
Plots individual visualizations of Gross Revenue vs. Effective LCOED 
for EACH IT capacity separately.
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
REVENUE_PER_MWH = {"A": 4000.0, "B1": 2800.0, "B2": 1600.0}
OTHER_VARIABLE_OPEX_EUR_PER_MWH = 3


# =============================================================================
# 2. DATA AGGREGATION & PROCESSING
# =============================================================================
all_scenarios = {}

print("Processing data across all IT Capacities...")

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

    # Strict Committed Energy (No C)
    df["Energy_Delivered_No_C_MWh"] = df["Energy_A_Annual_MWh"] + df["Energy_B1_Annual_MWh"] + df["Energy_B2_Annual_MWh"]
    
    # Financials
    df["Revenue_No_C_EUR"] = (df["Energy_A_Annual_MWh"] * REVENUE_PER_MWH["A"] + 
                              df["Energy_B1_Annual_MWh"] * REVENUE_PER_MWH["B1"] + 
                              df["Energy_B2_Annual_MWh"] * REVENUE_PER_MWH["B2"])
    df["Revenue_No_C_M_EUR"] = df["Revenue_No_C_EUR"] / 1e6

    # Effective LCOED (Scaling the HPP cost entirely onto the committed loads)
    df["LCOED_No_C"] = np.where(df["Energy_Delivered_No_C_MWh"] > 0, 
                                df["LCOE_delivered"] * (df["Total_Delivered_Annual_MWh"] / df["Energy_Delivered_No_C_MWh"]), 0)

    # Flexibility Metric
    df["Avg_Deferral_Days"] = np.where(
        df["Energy_Delivered_No_C_MWh"] > 0, 
        (df["Energy_A_Annual_MWh"] * 0 + df["Energy_B1_Annual_MWh"] * 1 + df["Energy_B2_Annual_MWh"] * 7) / df["Energy_Delivered_No_C_MWh"], 
        0
    )

    # Tag capacity and calculate Contribution Margin for finding the optimums
    df["IT_Capacity"] = cap
    df["Total_HPP_Cost_EUR"] = df["LCOE_delivered"] * df["Total_Delivered_Annual_MWh"] 
    df["DC_Variable_OPEX_No_C_EUR"] = OTHER_VARIABLE_OPEX_EUR_PER_MWH * df["Energy_Delivered_No_C_MWh"]
    df["Total_Variable_OPEX_No_C_EUR"] = df["Total_HPP_Cost_EUR"] + df["DC_Variable_OPEX_No_C_EUR"]
    df["Contribution_Margin_No_C_M_EUR"] = (df["Revenue_No_C_EUR"] - df["Total_Variable_OPEX_No_C_EUR"]) / 1e6

    # Store in dictionary and clean out zeroes
    df = df[df["LCOED_No_C"] > 0]
    all_scenarios[cap] = df

# =============================================================================
# 3. PLOTTING THE INDIVIDUAL FRONTIERS
# =============================================================================

print("Generating individual plots...")

cmap = plt.get_cmap('plasma')

for cap in IT_CAPACITIES_MW:
    cap_df = all_scenarios[cap].copy()
    
    if cap_df.empty:
        continue
        
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')

    # 1. The Scatter Cloud
    # Fixed vmin=0, vmax=7 so the color meaning is locked across all capacities
    scatter = ax.scatter(cap_df["Revenue_No_C_M_EUR"], cap_df["LCOED_No_C"], 
                         c=cap_df["Avg_Deferral_Days"], 
                         cmap=cmap, s=35, alpha=0.8, edgecolors='none',
                         vmin=0, vmax=7)

    # 2. Draw the Pareto Frontier (Minimum LCOED for given Revenue)
    bin_edges = np.linspace(cap_df["Revenue_No_C_M_EUR"].min(), cap_df["Revenue_No_C_M_EUR"].max(), 25)
    cap_df['Rev_Bin'] = pd.cut(cap_df["Revenue_No_C_M_EUR"], bins=bin_edges)

    idx_min = cap_df.groupby('Rev_Bin', observed=True)["LCOED_No_C"].idxmin().dropna()
    lower_bound = cap_df.loc[idx_min].sort_values(by="Revenue_No_C_M_EUR")

    ax.plot(lower_bound["Revenue_No_C_M_EUR"], lower_bound["LCOED_No_C"], 
            color='#e74c3c', linewidth=2.5, label='Efficiency Frontier (Lowest Cost)')

    # 3. Highlight the Optimum Contribution Margin
    opt_idx = cap_df["Contribution_Margin_No_C_M_EUR"].idxmax()
    
    ax.scatter(cap_df.loc[opt_idx, "Revenue_No_C_M_EUR"], cap_df.loc[opt_idx, "LCOED_No_C"], 
               color='none', edgecolors='#e74c3c', s=150, linewidth=2.5, zorder=5, 
               label='Optimal Mix (Max Contribution Margin)')

    # 4. Styling and Formatting
    ax.set_title(f"Revenue vs. Effective LCOED\n(IT Capacity: {cap} MW)", 
                 fontsize=16, fontweight='bold', color='#333333', pad=15)
    ax.set_xlabel("Gross Revenue (Excluding Tier C) [Millions € / Year]", fontweight='bold', color='grey', fontsize=11)
    ax.set_ylabel("Effective LCOED (Excluding Tier C) [€/MWh]", fontweight='bold', color='grey', fontsize=11)

    ax.grid(True, linestyle='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.03)
    cbar.set_label('Average Deferral Time of Portfolio (Days)', rotation=270, labelpad=20, fontweight='bold')

    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    
    # Save the plot
    filename = f"Thesis_Revenue_vs_LCOED_{cap}MW.svg"
    save_path = os.path.join(BASE_FOLDER, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Prevent overlapping memory issues
    
    print(f"  -> Saved {filename}")

print("All individual plots generated successfully!")