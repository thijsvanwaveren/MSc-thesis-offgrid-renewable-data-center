# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:44:52 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.4 - The Margin Story
Plots Contribution Margin vs. Committed Energy Delivered for the 16 MW data center.
Highlights the upper Pareto frontier (maximum margin for a given energy volume)
and pinpoints the absolute global optimum.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
REVENUE_PER_MWH = {"A": 4000.0, "B1": 2800.0, "B2": 1600.0}
OTHER_VARIABLE_OPEX_EUR_PER_MWH = 3

FILE_NAME = f"Feasible_3D_Sweep_Results_99.9pct_IT{IT_CAPACITY:.1f}.csv"
file_path = os.path.join(BASE_FOLDER, FILE_NAME)

# =============================================================================
# 2. LOAD AND PROCESS DATA
# =============================================================================
print(f"Loading data from: {FILE_NAME}...")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find the file at {file_path}. Please check the path.")

df = pd.read_csv(file_path)

# Ensure columns are in MWh
for col in ["Energy_A_Annual_GWh", "Energy_B1_Annual_GWh", "Energy_B2_Annual_GWh", "Total_Delivered_Annual_GWh"]:
    if col in df.columns:
        df[col.replace("_GWh", "_MWh")] = df[col] * 1000.0

# 1. Calculate Total Committed Energy Delivered (X-Axis)
df["Energy_Delivered_No_C_MWh"] = (df.get("Energy_A_Annual_MWh", 0) + 
                                   df.get("Energy_B1_Annual_MWh", 0) + 
                                   df.get("Energy_B2_Annual_MWh", 0))

df["Energy_Delivered_No_C_GWh"] = df["Energy_Delivered_No_C_MWh"] / 1000.0

# Filter out edge cases with zero energy
df = df[df["Energy_Delivered_No_C_MWh"] > 0].copy()

# 2. Calculate Gross Revenue
df["Revenue_No_C_EUR"] = (df.get("Energy_A_Annual_MWh", 0) * REVENUE_PER_MWH["A"] + 
                          df.get("Energy_B1_Annual_MWh", 0) * REVENUE_PER_MWH["B1"] + 
                          df.get("Energy_B2_Annual_MWh", 0) * REVENUE_PER_MWH["B2"])

# 3. Calculate Total Costs
energy_col = "Total_Delivered_Annual_MWh" if "Total_Delivered_Annual_MWh" in df.columns else "Energy_Delivered_No_C_MWh"
df["Total_HPP_Cost_EUR"] = df["LCOE_delivered"] * df[energy_col] 
df["DC_Variable_OPEX_EUR"] = OTHER_VARIABLE_OPEX_EUR_PER_MWH * df["Energy_Delivered_No_C_MWh"]
df["Total_Variable_OPEX_EUR"] = df["Total_HPP_Cost_EUR"] + df["DC_Variable_OPEX_EUR"]

# 4. Target Metric: Contribution Margin (Y-Axis)
df["Contribution_Margin_M_EUR"] = (df["Revenue_No_C_EUR"] - df["Total_Variable_OPEX_EUR"]) / 1e6

# 5. Calculate Average Deferral Time in Days (Color Scale)
df["Avg_Deferral_Days"] = (df.get("Energy_A_Annual_MWh", 0) * 0 + 
                           df.get("Energy_B1_Annual_MWh", 0) * 1 + 
                           df.get("Energy_B2_Annual_MWh", 0) * 7) / df["Energy_Delivered_No_C_MWh"]

# =============================================================================
# 3. GENERATE VISUALIZATION
# =============================================================================
print("Generating Contribution Margin Scatter Plot...")

fig, ax = plt.subplots(figsize=(11, 7), facecolor='white')
cmap = plt.get_cmap('plasma')

# --- 3a. Plot the Scatter Cloud ---
scatter = ax.scatter(df["Energy_Delivered_No_C_GWh"], 
                     df["Contribution_Margin_M_EUR"], 
                     c=df["Avg_Deferral_Days"], 
                     cmap=cmap, s=35, alpha=0.9, edgecolors='none', 
                     vmin=0, vmax=7, zorder=2)

# --- 3b. Calculate and Plot the Smooth Upper Pareto Frontier ---
# Round the X-axis to the nearest integer to group the distinct vertical columns
# and find the absolute maximum Y (Margin) inside each column.
df['Energy_Rounded'] = df["Energy_Delivered_No_C_GWh"].round(0)
idx_max = df.groupby('Energy_Rounded')["Contribution_Margin_M_EUR"].idxmax().dropna()
upper_bound = df.loc[idx_max].sort_values(by="Energy_Delivered_No_C_GWh")

# Plot the perfectly smooth frontier line
ax.plot(upper_bound["Energy_Delivered_No_C_GWh"], 
        upper_bound["Contribution_Margin_M_EUR"], 
        color='#e74c3c', linewidth=2.5, zorder=3)

# Highlight the Global Optimum (Maximum Contribution Margin)
opt_idx = df["Contribution_Margin_M_EUR"].idxmax()
ax.scatter(df.loc[opt_idx, "Energy_Delivered_No_C_GWh"], 
           df.loc[opt_idx, "Contribution_Margin_M_EUR"], 
           color='none', edgecolors='#e74c3c', s=150, linewidth=2.5, zorder=4)

# --- 3c. Styling and Formatting ---
ax.set_title(f"Profitability Dynamics: Contribution Margin vs. Committed Energy\n(IT Capacity: {IT_CAPACITY} MW)", 
             fontsize=16, fontweight='bold', color='#333333', pad=15)

ax.set_xlabel("Committed Energy Delivered (GWh/Year)", fontweight='bold', color='#595959', fontsize=12)
ax.set_ylabel("Contribution Margin [Millions € / Year]", fontweight='bold', color='#595959', fontsize=12)

# Clean grid
ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Custom Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#8e44ad', markersize=9, label='Feasible Workload Mixes'),
    Line2D([0], [0], color='#e74c3c', lw=2.5, label='Efficiency Frontier (Max Margin)'),
    Line2D([0], [0], marker='o', color='w', markeredgecolor='#e74c3c', markersize=12, markeredgewidth=2.5, label='Global Optimum')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.95)

# Colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Average Deferral Time of Portfolio (Days)', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

plt.tight_layout()

# Save the plot
save_path = os.path.join(BASE_FOLDER, f"Thesis_Margin_vs_Energy_{IT_CAPACITY}MW.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"-> Successfully saved plot to: {save_path}")