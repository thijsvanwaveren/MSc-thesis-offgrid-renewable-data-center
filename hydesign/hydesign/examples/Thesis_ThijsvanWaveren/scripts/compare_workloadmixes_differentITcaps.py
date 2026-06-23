# -*- coding: utf-8 -*-
"""
Script 2: Compare multiple optimal scenarios across different Installed IT Capacities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================
# USER INPUTS
# =========================
# Point this to the folder containing your exported "Optimal_Workload_Mix_*.csv" files
FOLDER_PATH = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
HOURS_PER_YEAR = 8760

# =========================
# DATA AGGREGATION
# =========================
# Find all CSV files that match our optimal export naming convention
file_pattern = os.path.join(FOLDER_PATH, "Optimal_Workload_Mix_*MW.csv")
file_list = glob.glob(file_pattern)

if not file_list:
    print(f"No files found matching pattern: {file_pattern}")
    exit()

# Read and combine all files
dfs = [pd.read_csv(f) for f in file_list]
df_comp = pd.concat(dfs, ignore_index=True)

# Sort by Installed IT Capacity to ensure the X-axis is chronological/logical
df_comp = df_comp.sort_values("Installed_IT_Capacity_MW").reset_index(drop=True)

# Convert x-axis labels to strings for categorical plotting
x_labels = [f"{cap} MW" for cap in df_comp["Installed_IT_Capacity_MW"]]
x = np.arange(len(df_comp))
# =========================
# PLOTTING & VISUALIZATION (UPDATED)
# =========================
import matplotlib.ticker as mtick

# Assumed Power Usage Effectiveness for liquid-cooled AI Data Center
PUE = 1.15 

# Calculate Total Energy Served and Cooling Energy in GWh
df_comp["Energy_Served_Total_GWh"] = (
    df_comp["Energy_A_Annual_GWh"] + 
    df_comp["Energy_B1_Annual_GWh"] + 
    df_comp["Energy_B2_Annual_GWh"] + 
    df_comp["Energy_C_Annual_GWh"]
)
df_comp["Cooling_Energy_GWh"] = df_comp["Energy_Served_Total_GWh"] * (PUE - 1.0)

# Define the exact colors from the theme
COLOR_BAR = '#19607D'    # Dark Teal
COLOR_BAR_2 = '#5E93A5'  # Lighter Teal 
COLOR_LINE = '#E86624'   # Vibrant Orange

def apply_clean_style(ax1, ax2=None):
    """Helper function to apply the clean, modern aesthetic."""
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)
    ax1.xaxis.grid(False)
    ax1.set_axisbelow(True) 
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('grey')
    ax1.spines['bottom'].set_color('grey')
    
    if ax2:
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('grey')
        ax2.spines['bottom'].set_color('grey')

# --- PLOT 1: Utilization Rate vs Contribution Margin ---
fig, ax1 = plt.subplots(figsize=(10, 6))

bars1 = ax1.bar(x, df_comp["Realized_Utilization"] * 100, color=COLOR_BAR, width=0.4, label="Utilization rate")
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("IT Capacity (MW)", color='grey', fontweight='bold')
ax1.set_ylabel("Utilization rate (%)", color='grey')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_ylim(0, 100) 

ax2 = ax1.twinx()
line1 = ax2.plot(x, df_comp["Contribution_Margin_EUR"] / 1e6, color=COLOR_LINE, linewidth=2.5, label="Contribution Margin (M€)")
ax2.set_ylabel("Contribution Margin (M€ / Year)", color='grey')
ax2.set_ylim(bottom=0) # FORCED TO ZERO

apply_clean_style(ax1, ax2)
plt.title("Utilization Rate vs Contribution Margin", fontsize=14, pad=20, color='#333333')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
plt.tight_layout()
plt.show()


# --- PLOT 2: Financial Plot (Revenue, Full Cost Stack, LCOE) ---
fig, ax1 = plt.subplots(figsize=(10, 6))
width = 0.35

# 1. Stack all costs (Facility CAPEX -> IT CAPEX -> Fixed OPEX -> Var OPEX)
b_fac = df_comp["Annualized_Facility_CAPEX_EUR"] / 1e6
b_it = df_comp["Annualized_IT_CAPEX_EUR"] / 1e6
b_fix_op = df_comp["Annual_Fixed_OPEX_EUR"] / 1e6
b_var_op = df_comp["Total_Variable_OPEX_EUR"] / 1e6

ax1.bar(x - width/2, b_fac, width, color='#8FA8B3', label="Facility CAPEX")
ax1.bar(x - width/2, b_it, width, bottom=b_fac, color=COLOR_BAR_2, label="IT CAPEX")
ax1.bar(x - width/2, b_fix_op, width, bottom=b_fac + b_it, color='#B0C4DE', label="Fixed OPEX")
ax1.bar(x - width/2, b_var_op, width, bottom=b_fac + b_it + b_fix_op, color='#D3D3D3', label="Variable OPEX")

# 2. Plot Revenue next to the stack
ax1.bar(x + width/2, df_comp["Revenue_Total_EUR"] / 1e6, width, color=COLOR_BAR, label="Total Revenue")

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("IT Capacity (MW)", color='grey', fontweight='bold')
ax1.set_ylabel("Millions € / Year", color='grey')

# 3. LCOE Line
ax2 = ax1.twinx()
ax2.plot(x, df_comp["LCOE_delivered"], color=COLOR_LINE, linewidth=2.5, marker='o', markersize=6, label="LCOE Delivered (€/MWh)")
ax2.set_ylabel("LCOE Delivered (€/MWh)", color='grey')
ax2.set_ylim(bottom=0) # FORCED TO ZERO

apply_clean_style(ax1, ax2)
plt.title("Financial Overview: Total Costs vs Revenue vs LCOE", fontsize=14, pad=20, color='#333333')
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
plt.tight_layout()
plt.show()


# --- PLOT 3: Energy Balance (GWh) Including Cooling ---
fig, ax1 = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(df_comp))

# Colors: A, B1, B2, C, Cooling (Cyan), Curtailment (Red)
plot_colors = [COLOR_BAR, COLOR_BAR_2, '#A8C2CB', 'lightgrey', '#42bbc9', '#D94833'] 

for col, label, color in zip(
    ["Energy_A_Annual_GWh", "Energy_B1_Annual_GWh", "Energy_B2_Annual_GWh", "Energy_C_Annual_GWh", "Cooling_Energy_GWh", "Curtailment_Annual_GWh"],
    ["Tier A", "Tier B1", "Tier B2", "Tier C", f"Cooling (PUE {PUE})", "Curtailed"],
    plot_colors
):
    bars = ax1.bar(x, df_comp[col], bottom=bottom, label=label, color=color, width=0.5)
    ax1.bar_label(bars, label_type='center', fmt=lambda val: f'{val:.0f}' if val > 2 else '', color='white' if color in [COLOR_BAR, COLOR_BAR_2, '#42bbc9', '#D94833'] else 'black')
    bottom += df_comp[col].values

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("IT Capacity (MW)", color='grey', fontweight='bold')
ax1.set_ylabel("Annual Energy (GWh)", color='grey')

apply_clean_style(ax1)
plt.title("Power Fate: Energy Balance Including Cooling", fontsize=14, pad=20, color='#333333')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
plt.tight_layout()
plt.show()

# --- PLOT 4: Absolute Utilization (Avg MW vs Max Capacity) ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Calculate Average MW served
df_comp["Avg_Committed_MW"] = (df_comp["Total_Committed_Delivered_Annual_MWh"]) / HOURS_PER_YEAR
df_comp["Avg_C_MW"] = (df_comp["Energy_C_Annual_MWh"]) / HOURS_PER_YEAR

# 1. Primary Axis (Bars): Plot the "Ghost" bar showing Total Installed Capacity
ax1.bar(x, df_comp["Installed_IT_Capacity_MW"], color='#f4f4f4', edgecolor='darkgrey', linestyle='--', linewidth=1.5, width=0.5, label="Max IT Cap")

# Fill the bar with the actual utilized Average MW
ax1.bar(x, df_comp["Avg_Committed_MW"], color=COLOR_BAR, width=0.5, label="Committed (A+B1+B2)")
ax1.bar(x, df_comp["Avg_C_MW"], bottom=df_comp["Avg_Committed_MW"], color='lightgrey', edgecolor='grey', width=0.5, label="Tier C")

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("IT Capacity (MW)", color='grey', fontweight='bold')
ax1.set_ylabel("Average Power (MW)", color='grey')

# 2. Secondary Axis (Lines): Plot Realized Utilization (With and Without C)
ax2 = ax1.twinx()

# Line 1: Total Utilization (With C) - Solid Orange Line
ax2.plot(x, df_comp["Realized_Utilization"] * 100, color=COLOR_LINE, linewidth=2.5, marker='o', markersize=6, label="Utilization (With C)")

# Line 2: Utilization Without C - Dashed Dark Grey Line to contrast
ax2.plot(x, df_comp["Utilization_Without_C"] * 100, color='#555555', linewidth=2.5, linestyle='--', marker='s', markersize=5, label="Utilization (Without C)")

# Format the secondary axis
ax2.set_ylabel("Utilization (%)", color='grey')
ax2.set_ylim(0, 105) # Set slightly above 100% to give the top data points breathing room
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

# Apply styling to both axes
apply_clean_style(ax1, ax2)
plt.title("Physical Utilization: Average Power Served vs Max Capacity", fontsize=14, pad=20, color='#333333')

# Combine legends from both axes at the bottom
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

plt.tight_layout()
plt.show()