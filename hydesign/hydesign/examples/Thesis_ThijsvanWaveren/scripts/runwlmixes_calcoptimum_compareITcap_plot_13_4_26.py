# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:40:32 2026

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
from matplotlib.patches import Patch
import os

# =============================================================================
# 1. MASTER USER INPUTS & ASSUMPTIONS
# =============================================================================

BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITIES_MW = [16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0] 

HOURS_PER_YEAR = 8760
PUE = 1.15 

REVENUE_PER_MWH = {
    "A": 3000.0,
    "B1": 2100.0,
    "B2": 1200.0,
    "C": 300.0
}

OTHER_VARIABLE_OPEX_EUR_PER_MWH = 5.50 
FIXED_OPEX_PER_MW_EUR_PER_YEAR = 1_000_000.0
CAPEX_IT_PER_MW_EUR = 35_000_000.0  
IT_LIFETIME_YEARS = 4.5    
CAPEX_FACILITY_PER_MW_EUR = 15_000_000.0 
FACILITY_LIFETIME_YEARS = 20.0 

# =============================================================================
# 2. DATA PROCESSING ENGINE
# =============================================================================
optimal_results = []
print("Starting data processing...")

for cap in IT_CAPACITIES_MW:
    file_name = f"Feasible_3D_Sweep_Results_99.9pct_IT{cap:.1f}.csv"
    file_path = os.path.join(BASE_FOLDER, file_name)
    
    if not os.path.exists(file_path):
        print(f"  [WARNING] File not found, skipping: {file_name}")
        continue
        
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

    # Calculate Costs
    df["Electricity_Cost_EUR"] = df["LCOE_delivered"] * df["Total_Delivered_Annual_MWh"]
    df["Other_Variable_OPEX_EUR"] = OTHER_VARIABLE_OPEX_EUR_PER_MWH * df["Total_Delivered_Annual_MWh"]
    df["Total_Variable_OPEX_EUR"] = df["Electricity_Cost_EUR"] + df["Other_Variable_OPEX_EUR"]

    df["Annualized_IT_CAPEX_EUR"] = (cap * CAPEX_IT_PER_MW_EUR) / IT_LIFETIME_YEARS
    df["Annualized_Facility_CAPEX_EUR"] = (cap * CAPEX_FACILITY_PER_MW_EUR) / FACILITY_LIFETIME_YEARS
    df["Annual_Fixed_OPEX_EUR"] = cap * FIXED_OPEX_PER_MW_EUR_PER_YEAR

    df["Total_Annual_Cost_EUR"] = df["Total_Variable_OPEX_EUR"] + df["Annual_Fixed_OPEX_EUR"] + df["Annualized_IT_CAPEX_EUR"] + df["Annualized_Facility_CAPEX_EUR"]

    df["Contribution_Margin_EUR"] = df["Revenue_Total_EUR"] - df["Total_Variable_OPEX_EUR"]
    df["Net_Profit_EUR"] = df["Revenue_Total_EUR"] - df["Total_Annual_Cost_EUR"]
    df["Realized_Utilization"] = df["Total_Delivered_Annual_MWh"] / (cap * HOURS_PER_YEAR)
    df["Utilization_Without_C"] = df["Total_Committed_Delivered_Annual_MWh"] / (cap * HOURS_PER_YEAR)
    
    # NEW: Calculate Unit Economics
    df["Average_Capture_Price_EUR_per_MWh"] = df["Revenue_Total_EUR"] / df["Total_Delivered_Annual_MWh"]

    # CRITICAL FIX: Sort by Net Profit to capture the massive penalty of overbuilding CAPEX
    optimal_scenario = df.sort_values("Net_Profit_EUR", ascending=False).head(1).copy()
    optimal_scenario["Installed_IT_Capacity_MW"] = cap
    optimal_results.append(optimal_scenario)

if not optimal_results:
    print("No valid data processed. Exiting.")
    exit()

df_comp = pd.concat(optimal_results, ignore_index=True)
df_comp = df_comp.sort_values("Installed_IT_Capacity_MW").reset_index(drop=True)

# Find the absolute best facility size for the Waterfall and Tornado charts
best_scenario_idx = df_comp["Net_Profit_EUR"].idxmax()
best_scenario = df_comp.loc[best_scenario_idx]
print(f"Optimal Facility Size: {best_scenario['Installed_IT_Capacity_MW']} MW")

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

def apply_clean_style(ax):
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')

# --- PLOT 1: The Hero Plot (Utilization vs Net Profit) ---
fig, ax1 = plt.subplots(figsize=(10, 6))
bars1 = ax1.bar(x, df_comp["Realized_Utilization"] * 100, color='#19607D', width=0.4, label="Utilization rate")
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("IT Capacity (MW)", color='grey', fontweight='bold')
ax1.set_ylabel("Utilization rate (%)", color='grey')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_ylim(0, 105) 

ax2 = ax1.twinx()
# UPDATED: Now plotting Net Profit to show the "hill" or drop-off of overbuilding
line1 = ax2.plot(x, df_comp["Net_Profit_EUR"] / 1e6, color='#E86624', linewidth=3, marker='o', label="Net Profit (M€/y)")
ax2.set_ylabel("Net Profit (Millions € / Year)", color='grey')
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

apply_clean_style(ax1)
plt.title("Utilization vs Net Profit", fontsize=14, pad=20, color='#333333')
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
plt.tight_layout()
plt.show()

# --- PLOT 2A: Unit Economics (Dual Axis for Scale Variance) ---
fig, ax1 = plt.subplots(figsize=(10, 5))

color_capture = '#19607D' # Deep Blue
color_lcoe = '#E86624'    # Vibrant Orange

# Left Axis: Capture Price (High Magnitude)
line1 = ax1.plot(x, df_comp["Average_Capture_Price_EUR_per_MWh"], color=color_capture, linewidth=3, marker='s', label="Avg. compute revenue")
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("IT Capacity (MW)", color='grey', fontweight='bold')

# Color code the left Y-axis to match the Capture Price line
ax1.set_ylabel("Capture Price (€ / MWh)", color=color_capture, fontweight='bold', fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_capture)
ax1.set_ylim(bottom=0)

# Right Axis: LCOE (Low Magnitude)
ax2 = ax1.twinx()
line2 = ax2.plot(x, df_comp["LCOE_delivered"], color=color_lcoe, linewidth=3, marker='o', label="LCOE")

# Color code the right Y-axis to match the LCOE line
ax2.set_ylabel("LCOED (€ / MWh)", color=color_lcoe, fontweight='bold', fontsize=11)
ax2.tick_params(axis='y', labelcolor=color_lcoe)
ax2.set_ylim(bottom=0)

# Apply clean style (customized for dual axis)
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)
ax1.xaxis.grid(False)
ax1.set_axisbelow(True) 
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False) # Hide right spine of ax1 so ax2 can use it
ax1.spines['left'].set_color(color_capture)
ax2.spines['right'].set_color(color_lcoe)
ax1.spines['bottom'].set_color('grey')

plt.title("Compute revenue per MWh vs LCOED", fontsize=14, pad=20, color='#333333')

# Combine legends from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
plt.show()

# --- PLOT 2B: Waterfall Chart for the Optimal Capacity ---
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Gross Revenue', 'Variable OPEX', 'Fixed OPEX', 'Facility CAPEX', 'IT CAPEX', 'Net Profit']
values = [
    best_scenario["Revenue_Total_EUR"] / 1e6,
    -best_scenario["Total_Variable_OPEX_EUR"] / 1e6,
    -best_scenario["Annual_Fixed_OPEX_EUR"] / 1e6,
    -best_scenario["Annualized_Facility_CAPEX_EUR"] / 1e6,
    -best_scenario["Annualized_IT_CAPEX_EUR"] / 1e6,
    best_scenario["Net_Profit_EUR"] / 1e6
]

# Calculate bottoms for waterfall effect
bottoms = [0]
for i in range(len(values) - 2):
    bottoms.append(bottoms[-1] + values[i])
bottoms.append(0) # Net profit starts at 0

colors = ['#19607D', '#D3D3D3', '#B0C4DE', '#8FA8B3', '#5E93A5', '#E86624']
for i in range(len(categories)):
    ax.bar(categories[i], values[i], bottom=bottoms[i], color=colors[i], width=0.6)
    # Add text labels
    val_text = f"+{values[i]:.1f}" if values[i] > 0 else f"{values[i]:.1f}"
    y_pos = bottoms[i] + values[i]/2 if i != len(categories)-1 else values[i] + 5
    ax.text(i, y_pos, val_text, ha='center', va='center', color='black' if i!=0 else 'white', fontweight='bold')

apply_clean_style(ax)
plt.title(f"Financial Analysis at Optimal Capacity ({best_scenario['Installed_IT_Capacity_MW']} MW)", fontsize=14, pad=20)
plt.ylabel("Millions € / Year", color='grey')
plt.tight_layout()
plt.show()

# --- PLOT 3: Energy Balance (GWh) Including Cooling ---
# UPDATED: Better colors. Useful = Blues/Greens. Waste/Parasitic = Red/Grey
fig, ax1 = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(df_comp))
plot_colors = ['#1a3c5a', '#29638c', '#4ba4cc', '#f1c40f', '#d3d3d3', '#e74c3c'] 

for col, label, color in zip(
    ["Energy_A_Annual_GWh", "Energy_B1_Annual_GWh", "Energy_B2_Annual_GWh", "Energy_C_Annual_GWh", "Cooling_Energy_GWh", "Curtailment_Annual_GWh"],
    ["Tier A", "Tier B1", "Tier B2", "Tier C", f"Cooling (PUE {PUE})", "Curtailed"],
    plot_colors
):
    bars = ax1.bar(x, df_comp[col], bottom=bottom, label=label, color=color, width=0.5)
    bottom += df_comp[col].values

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("IT Capacity (MW)", color='grey', fontweight='bold')
ax1.set_ylabel("Annual Energy (GWh)", color='grey')

apply_clean_style(ax1)
plt.title("Energy Balance & Curtailment", fontsize=14, pad=20, color='#333333')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
plt.show()
# --- PLOT 4: Absolute Utilization (Stacked Bar with Stranded Assets) ---
fig, ax1 = plt.subplots(figsize=(10, 6))

total_avg_power = df_comp["Avg_Committed_MW"] + df_comp["Avg_C_MW"]
idle_power = df_comp["Installed_IT_Capacity_MW"] - total_avg_power

# Stack 1: Committed Load (Vibrant Blue)
ax1.bar(x, df_comp["Avg_Committed_MW"], color='#19607D', width=0.5, label="Committed Load (A+B1+B2)")

# Stack 2: Tier C Load (Lighter Teal)
ax1.bar(x, df_comp["Avg_C_MW"], bottom=df_comp["Avg_Committed_MW"], color='#5E93A5', width=0.5, label="Tier C Load")

# Stack 3: Idle/Stranded Hardware (White/Grey with Red Hatching)
ax1.bar(x, idle_power, bottom=total_avg_power, facecolor='#f8f9fa', edgecolor='#e74c3c', hatch='//', linewidth=1.5, width=0.5, label="Idle Hardware")

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("IT Capacity (MW)", color='grey', fontweight='bold')
ax1.set_ylabel("Average Power (MW)", color='grey')

apply_clean_style(ax1)
plt.title("Hardware Efficiency", fontsize=14, pad=20, color='#333333')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
plt.show()

# --- PLOT 5: TORNADO CHART (ALL PARAMETERS SENSITIVITY) ---
base_profit = best_scenario["Net_Profit_EUR"]

# Define all parameters and their baseline monetary values in the optimal scenario
# Type "revenue" means +20% increases profit. Type "cost" means +20% decreases profit.
sensitivities = {
    "Tier A Price": {"base": best_scenario["Revenue_A_EUR"], "type": "revenue"},
    "Tier B1 Price": {"base": best_scenario["Revenue_B1_EUR"], "type": "revenue"},
    "Tier B2 Price": {"base": best_scenario["Revenue_B2_EUR"], "type": "revenue"},
    "Tier C Price": {"base": best_scenario["Revenue_C_EUR"], "type": "revenue"},
    "IT Hardware CAPEX": {"base": best_scenario["Annualized_IT_CAPEX_EUR"], "type": "cost"},
    "Facility CAPEX": {"base": best_scenario["Annualized_Facility_CAPEX_EUR"], "type": "cost"},
    "Fixed OPEX": {"base": best_scenario["Annual_Fixed_OPEX_EUR"], "type": "cost"},
    "Variable OPEX": {"base": best_scenario["Other_Variable_OPEX_EUR"], "type": "cost"}
}

swing_data = []

for label, data in sensitivities.items():
    delta = data["base"] * 0.20  # 20% swing
    
    if data["type"] == "revenue":
        profit_low = base_profit - delta   # Revenue drops 20% -> Profit drops
        profit_high = base_profit + delta  # Revenue rises 20% -> Profit rises
    else: # type == "cost"
        profit_low = base_profit + delta   # Cost drops 20% -> Profit rises (Best case)
        profit_high = base_profit - delta  # Cost rises 20% -> Profit drops (Worst case)
        
    impact_magnitude = abs(profit_high - profit_low)
    
    swing_data.append({
        "label": label,
        "low_scenario": profit_low / 1e6,
        "high_scenario": profit_high / 1e6,
        "impact": impact_magnitude
    })

# Sort by impact magnitude so the biggest bars are at the top
swing_data = sorted(swing_data, key=lambda x: x["impact"])

labels = [d["label"] for d in swing_data]
lows = [d["low_scenario"] for d in swing_data]
highs = [d["high_scenario"] for d in swing_data]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(labels))

base_profit_m = base_profit / 1e6
ax.axvline(x=base_profit_m, color='black', linestyle='--', linewidth=1.5, zorder=0)

for i in range(len(labels)):
    # Green if the scenario improves profit, Red if it hurts profit
    color_high_swing = '#2ecc71' if highs[i] > base_profit_m else '#e74c3c'
    color_low_swing = '#e74c3c' if lows[i] < base_profit_m else '#2ecc71'
    
    # Plot the +20% scenario
    ax.barh(y_pos[i], highs[i] - base_profit_m, left=base_profit_m, height=0.5, color=color_high_swing, edgecolor='white', zorder=3)
    # Plot the -20% scenario
    ax.barh(y_pos[i], lows[i] - base_profit_m, left=base_profit_m, height=0.5, color=color_low_swing, edgecolor='white', zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontweight='bold', color='#333333')
ax.set_xlabel("Net Profit (Millions € / Year)", color='grey', fontweight='bold')

# Legend for clarity on Tornado Chart
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', edgecolor='white', label='Favorable Scenario (Profit Increases)'),
                   Patch(facecolor='#e74c3c', edgecolor='white', label='Adverse Scenario (Profit Decreases)')]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

apply_clean_style(ax)
plt.title(f"Sensitivity Analysis: ±20% Parameter Swings (Optimal {best_scenario['Installed_IT_Capacity_MW']} MW Facility)", fontsize=14, pad=20)
plt.tight_layout()
plt.show()