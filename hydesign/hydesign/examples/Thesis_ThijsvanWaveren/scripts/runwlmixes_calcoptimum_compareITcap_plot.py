# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:21:16 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Unified AI Data Center Analysis Tool
Processes multiple HPP capacity sweeps, calculates financials, and plots comparison dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

# =============================================================================
# 1. MASTER USER INPUTS & ASSUMPTIONS
# =============================================================================

# --- FILE PATHS ---
# The folder where all your 'Feasible_3D_Sweep...' CSVs are located
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"

# List the IT capacities you want to process. The script will look for files named:
# "Feasible_3D_Sweep_Results_99.9pct_IT{cap}.csv"
IT_CAPACITIES_MW = [16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0] 

HOURS_PER_YEAR = 8760
PUE = 1.15  # Power Usage Effectiveness for liquid cooling

# --- REVENUE ASSUMPTIONS (€ / MWh) ---
REVENUE_PER_MWH = {
    "A": 3000.0,
    "B1": 2100.0,
    "B2": 1200.0,
    "C": 300.0
}

# --- FINANCIAL OPEX / CAPEX ASSUMPTIONS (AI-Focused) ---
# Variable OPEX (Water cooling, minor consumables) - € per MWh delivered
OTHER_VARIABLE_OPEX_EUR_PER_MWH = 5.50 

# Fixed OPEX (24/7 Staff, Maintenance, Security, Insurance) - € per MW per year
FIXED_OPEX_PER_MW_EUR_PER_YEAR = 1_000_000.0

# IT Hardware (GPUs, InfiniBand Networking)
CAPEX_IT_PER_MW_EUR = 35_000_000.0  
IT_LIFETIME_YEARS = 4.5   

# Facility (Building, Power Delivery, Liquid Cooling systems)
CAPEX_FACILITY_PER_MW_EUR = 15_000_000.0 
FACILITY_LIFETIME_YEARS = 20.0 


# =============================================================================
# 2. DATA PROCESSING ENGINE
# =============================================================================
optimal_results = []

print("Starting data processing...")

for cap in IT_CAPACITIES_MW:
    # Construct exact filename based on capacity
    file_name = f"Feasible_3D_Sweep_Results_99.9pct_IT{cap:.1f}.csv"
    file_path = os.path.join(BASE_FOLDER, file_name)
    
    if not os.path.exists(file_path):
        print(f"  [WARNING] File not found, skipping: {file_name}")
        continue
        
    print(f"  Processing {cap} MW facility...")
    df = pd.read_csv(file_path)

    # Convert Energy to MWh
    for col in ["Energy_A_Annual_GWh", "Energy_B1_Annual_GWh", "Energy_B2_Annual_GWh", 
                "Energy_C_Annual_GWh", "Total_Delivered_Annual_GWh", "Curtailment_Annual_GWh"]:
        df[col.replace("_GWh", "_MWh")] = df[col] * 1000.0

    df["Total_Committed_Delivered_Annual_MWh"] = df["Energy_A_Annual_MWh"] + df["Energy_B1_Annual_MWh"] + df["Energy_B2_Annual_MWh"]

    # Calculate Revenues
    df["Revenue_A_EUR"] = df["Energy_A_Annual_MWh"] * REVENUE_PER_MWH["A"]
    df["Revenue_B1_EUR"] = df["Energy_B1_Annual_MWh"] * REVENUE_PER_MWH["B1"]
    df["Revenue_B2_EUR"] = df["Energy_B2_Annual_MWh"] * REVENUE_PER_MWH["B2"]
    df["Revenue_C_EUR"] = df["Energy_C_Annual_MWh"] * REVENUE_PER_MWH["C"]
    df["Revenue_Total_EUR"] = df["Revenue_A_EUR"] + df["Revenue_B1_EUR"] + df["Revenue_B2_EUR"] + df["Revenue_C_EUR"]

    # Calculate Costs (using physical 'cap' for amortization)
    df["Electricity_Cost_EUR"] = df["LCOE_delivered"] * df["Total_Delivered_Annual_MWh"]
    df["Other_Variable_OPEX_EUR"] = OTHER_VARIABLE_OPEX_EUR_PER_MWH * df["Total_Delivered_Annual_MWh"]
    df["Total_Variable_OPEX_EUR"] = df["Electricity_Cost_EUR"] + df["Other_Variable_OPEX_EUR"]

    df["Annualized_IT_CAPEX_EUR"] = (cap * CAPEX_IT_PER_MW_EUR) / IT_LIFETIME_YEARS
    df["Annualized_Facility_CAPEX_EUR"] = (cap * CAPEX_FACILITY_PER_MW_EUR) / FACILITY_LIFETIME_YEARS
    df["Annual_Fixed_OPEX_EUR"] = cap * FIXED_OPEX_PER_MW_EUR_PER_YEAR

    df["Total_Annual_Cost_EUR"] = df["Total_Variable_OPEX_EUR"] + df["Annual_Fixed_OPEX_EUR"] + df["Annualized_IT_CAPEX_EUR"] + df["Annualized_Facility_CAPEX_EUR"]

    # Calculate Margins & Utilization
    df["Contribution_Margin_EUR"] = df["Revenue_Total_EUR"] - df["Total_Variable_OPEX_EUR"]
    df["Net_Profit_EUR"] = df["Revenue_Total_EUR"] - df["Total_Annual_Cost_EUR"]
    df["Realized_Utilization"] = df["Total_Delivered_Annual_MWh"] / (cap * HOURS_PER_YEAR)
    df["Utilization_Without_C"] = df["Total_Committed_Delivered_Annual_MWh"] / (cap * HOURS_PER_YEAR)

    # Isolate the #1 optimal scenario for this capacity
    optimal_scenario = df.sort_values("Contribution_Margin_EUR", ascending=False).head(1).copy()
    optimal_scenario["Installed_IT_Capacity_MW"] = cap
    optimal_results.append(optimal_scenario)

# Combine all optimal scenarios into one DataFrame
if not optimal_results:
    print("No valid data processed. Exiting.")
    exit()

df_comp = pd.concat(optimal_results, ignore_index=True)
df_comp = df_comp.sort_values("Installed_IT_Capacity_MW").reset_index(drop=True)
print("Data processing complete! Generating dashboards...\n")


# =============================================================================
# 3. PLOTTING & VISUALIZATION
# =============================================================================
x_labels = [f"{cap} MW" for cap in df_comp["Installed_IT_Capacity_MW"]]
x = np.arange(len(df_comp))

# Pre-calculate Energy and Averages for plotting
df_comp["Energy_Served_Total_GWh"] = df_comp["Energy_A_Annual_GWh"] + df_comp["Energy_B1_Annual_GWh"] + df_comp["Energy_B2_Annual_GWh"] + df_comp["Energy_C_Annual_GWh"]
df_comp["Cooling_Energy_GWh"] = df_comp["Energy_Served_Total_GWh"] * (PUE - 1.0)
df_comp["Avg_Committed_MW"] = df_comp["Total_Committed_Delivered_Annual_MWh"] / HOURS_PER_YEAR
df_comp["Avg_C_MW"] = df_comp["Energy_C_Annual_MWh"] / HOURS_PER_YEAR

# Aesthetics Theme
COLOR_BAR = '#19607D'    # Dark Teal
COLOR_BAR_2 = '#5E93A5'  # Lighter Teal 
COLOR_LINE = '#E86624'   # Vibrant Orange

def apply_clean_style(ax1, ax2=None):
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
ax2.set_ylim(bottom=0) 

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

b_fac = df_comp["Annualized_Facility_CAPEX_EUR"] / 1e6
b_it = df_comp["Annualized_IT_CAPEX_EUR"] / 1e6
b_fix_op = df_comp["Annual_Fixed_OPEX_EUR"] / 1e6
b_var_op = df_comp["Total_Variable_OPEX_EUR"] / 1e6

ax1.bar(x - width/2, b_fac, width, color='#8FA8B3', label="Facility CAPEX")
ax1.bar(x - width/2, b_it, width, bottom=b_fac, color=COLOR_BAR_2, label="IT CAPEX")
ax1.bar(x - width/2, b_fix_op, width, bottom=b_fac + b_it, color='#B0C4DE', label="Fixed OPEX")
ax1.bar(x - width/2, b_var_op, width, bottom=b_fac + b_it + b_fix_op, color='#D3D3D3', label="Variable OPEX")

ax1.bar(x + width/2, df_comp["Revenue_Total_EUR"] / 1e6, width, color=COLOR_BAR, label="Total Revenue")

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("IT Capacity (MW)", color='grey', fontweight='bold')
ax1.set_ylabel("Millions € / Year", color='grey')

ax2 = ax1.twinx()
ax2.plot(x, df_comp["LCOE_delivered"], color=COLOR_LINE, linewidth=2.5, marker='o', markersize=6, label="LCOE Delivered (€/MWh)")
ax2.set_ylabel("LCOE Delivered (€/MWh)", color='grey')
ax2.set_ylim(bottom=0) 

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

ax1.bar(x, df_comp["Installed_IT_Capacity_MW"], color='#f4f4f4', edgecolor='darkgrey', linestyle='--', linewidth=1.5, width=0.5, label="Max IT Cap")
ax1.bar(x, df_comp["Avg_Committed_MW"], color=COLOR_BAR, width=0.5, label="Committed (A+B1+B2)")
ax1.bar(x, df_comp["Avg_C_MW"], bottom=df_comp["Avg_Committed_MW"], color='lightgrey', edgecolor='grey', width=0.5, label="Tier C")

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("IT Capacity (MW)", color='grey', fontweight='bold')
ax1.set_ylabel("Average Power (MW)", color='grey')

ax2 = ax1.twinx()
ax2.plot(x, df_comp["Realized_Utilization"] * 100, color=COLOR_LINE, linewidth=2.5, marker='o', markersize=6, label="Utilization (With C)")
ax2.plot(x, df_comp["Utilization_Without_C"] * 100, color='#555555', linewidth=2.5, linestyle='--', marker='s', markersize=5, label="Utilization (Without C)")

ax2.set_ylabel("Utilization (%)", color='grey')
ax2.set_ylim(0, 105) 
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

apply_clean_style(ax1, ax2)
plt.title("Physical Utilization: Average Power Served vs Max Capacity", fontsize=14, pad=20, color='#333333')
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
plt.show()