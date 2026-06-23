# -*- coding: utf-8 -*-
"""
Updated Script 1: Evaluate a single CSV, plot, and export optimal scenario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONSTANTS & ASSUMPTIONS
# =========================
HOURS_PER_YEAR = 8760
INSTALLED_IT_CAP_MW = 20.0


# Paths (Adjust these to your local machine)
CSV_PATH = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts\Feasible_3D_Sweep_Results_99.9pct_IT20.0.csv"
EXPORT_PATH = rf"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts\Optimal_Workload_Mix_{INSTALLED_IT_CAP_MW}MW.csv"

# Financials
REVENUE_PER_MWH = {"A": 3000.0, "B1": 2100.0, "B2": 1200.0, "C": 300.0}

# =========================
# FINANCIAL OPEX / CAPEX ASSUMPTIONS (AI-Focused)
# =========================
# Variable OPEX (Water cooling, consumables) - € per MWh delivered
OTHER_VARIABLE_OPEX_EUR_PER_MWH = 5.5

# IT Hardware (GPUs, Networking)
CAPEX_IT_PER_MW_EUR = 35_000_000.0  
IT_LIFETIME_YEARS = 4.5   

# Facility (Building, Power Delivery, Liquid Cooling systems)
CAPEX_FACILITY_PER_MW_EUR = 10_000_000.0 
FACILITY_LIFETIME_YEARS = 25.0 

# Fixed OPEX (24/7 Staffing, Maintenance, Insurance, Telecom, Security) 
# Derived from 30MW NL Calculator (~€30.8M non-energy fixed costs)
FIXED_OPEX_PER_MW_EUR_PER_YEAR = 1_000_000.0


# =========================
# DATA LOADING & MATH
# =========================
df = pd.read_csv(CSV_PATH)

# Convert GWh to MWh
for col in ["Energy_A_Annual_GWh", "Energy_B1_Annual_GWh", "Energy_B2_Annual_GWh", 
            "Energy_C_Annual_GWh", "Total_Delivered_Annual_GWh", "Curtailment_Annual_GWh"]:
   df[col.replace("_GWh", "_MWh")] = df[col] * 1000.0

# Calculate committed (contracted) MWh
df["Total_Committed_Delivered_Annual_MWh"] = df["Energy_A_Annual_MWh"] + df["Energy_B1_Annual_MWh"] + df["Energy_B2_Annual_MWh"]


# =========================
# FINANCIAL CALCULATIONS
# =========================
# 1. Revenues
df["Revenue_A_EUR"] = df["Energy_A_Annual_MWh"] * REVENUE_PER_MWH["A"]
df["Revenue_B1_EUR"] = df["Energy_B1_Annual_MWh"] * REVENUE_PER_MWH["B1"]
df["Revenue_B2_EUR"] = df["Energy_B2_Annual_MWh"] * REVENUE_PER_MWH["B2"]
df["Revenue_C_EUR"] = df["Energy_C_Annual_MWh"] * REVENUE_PER_MWH["C"]
df["Revenue_Total_EUR"] = df["Revenue_A_EUR"] + df["Revenue_B1_EUR"] + df["Revenue_B2_EUR"] + df["Revenue_C_EUR"]

# 2. Variable Costs
df["Electricity_Cost_EUR"] = df["LCOE_delivered"] * df["Total_Delivered_Annual_MWh"]
df["Other_Variable_OPEX_EUR"] = OTHER_VARIABLE_OPEX_EUR_PER_MWH * df["Total_Delivered_Annual_MWh"]
df["Total_Variable_OPEX_EUR"] = df["Electricity_Cost_EUR"] + df["Other_Variable_OPEX_EUR"]

# 3. Fixed Costs & Split Amortization
df["Total_IT_CAPEX_EUR"] = INSTALLED_IT_CAP_MW * CAPEX_IT_PER_MW_EUR
df["Annualized_IT_CAPEX_EUR"] = df["Total_IT_CAPEX_EUR"] / IT_LIFETIME_YEARS

df["Total_Facility_CAPEX_EUR"] = INSTALLED_IT_CAP_MW * CAPEX_FACILITY_PER_MW_EUR
df["Annualized_Facility_CAPEX_EUR"] = df["Total_Facility_CAPEX_EUR"] / FACILITY_LIFETIME_YEARS

df["Annualized_Total_CAPEX_EUR"] = df["Annualized_IT_CAPEX_EUR"] + df["Annualized_Facility_CAPEX_EUR"]
df["Annual_Fixed_OPEX_EUR"] = INSTALLED_IT_CAP_MW * FIXED_OPEX_PER_MW_EUR_PER_YEAR

df["Total_Annual_Cost_EUR"] = df["Total_Variable_OPEX_EUR"] + df["Annual_Fixed_OPEX_EUR"] + df["Annualized_Total_CAPEX_EUR"]

# 4. Margins & Profits
df["Contribution_Margin_EUR"] = df["Revenue_Total_EUR"] - df["Total_Variable_OPEX_EUR"]
df["Net_Profit_EUR"] = df["Revenue_Total_EUR"] - df["Total_Annual_Cost_EUR"]

# 5. Utilization Metrics (Calculated against Physical Installed Capacity)
df["Realized_Utilization"] = df["Total_Delivered_Annual_MWh"] / (INSTALLED_IT_CAP_MW * HOURS_PER_YEAR)
df["Utilization_Without_C"] = df["Total_Committed_Delivered_Annual_MWh"] / (INSTALLED_IT_CAP_MW * HOURS_PER_YEAR)


# =========================
# EXPORT OPTIMAL SCENARIO
# =========================
top10 = df.sort_values("Contribution_Margin_EUR", ascending=False).head(10).reset_index(drop=True)
top10["Scenario"] = [f"S{i+1}" for i in range(len(top10))]

# Isolate the #1 scenario, stamp the capacity, and export
optimal_scenario = df.sort_values("Contribution_Margin_EUR", ascending=False).head(1).copy()
optimal_scenario["Installed_IT_Capacity_MW"] = INSTALLED_IT_CAP_MW  
optimal_scenario.to_csv(EXPORT_PATH, index=False)
print(f"Exported optimal scenario to: {EXPORT_PATH}")


# =========================
# PLOTTING
# =========================

# --- PLOT 1: Workload Mix vs Margin ---
fig, ax1 = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(top10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for col, label, color in zip(["Tier_A_MW", "Tier_B1_MW", "Tier_B2_MW"], ["Tier A", "Tier B1", "Tier B2"], colors):
    bars = ax1.bar(top10["Scenario"], top10[col], bottom=bottom, label=label, color=color)
    ax1.bar_label(bars, label_type='center', fmt=lambda x: f'{x:.1f}' if x > 0 else '')
    bottom += top10[col].values

top10["Avg_C_MW"] = top10["Energy_C_Annual_MWh"] / HOURS_PER_YEAR
bars_c = ax1.bar(top10["Scenario"], top10["Avg_C_MW"], bottom=bottom, label="Tier C (Avg Equivalent)", color='grey', alpha=0.4, edgecolor='black', linestyle='--')
ax1.set_ylabel("Power (MW)")
ax1.set_title(f"Workload mix vs Margin")

ax2 = ax1.twinx()
ax2.plot(top10["Scenario"], top10["Contribution_Margin_EUR"] / 1e6, color='red', marker='D')
ax2.set_ylabel("Margin (M€/yr)", color='red')
plt.show()

# --- PLOT 2: Energy Balance (in GWh) ---
plt.figure(figsize=(12, 6))
bottom = np.zeros(len(top10))

# Defined explicit colors: Blue, Orange, Green, Grey, Red (Curtailment)
plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'grey', 'red']

# Plot all tiers directly in GWh (No negative losses to draw downwards!)
for col, label, color in zip(
    ["Energy_A_Annual_GWh", "Energy_B1_Annual_GWh", "Energy_B2_Annual_GWh", 
     "Energy_C_Annual_GWh", "Curtailment_Annual_GWh"],
    ["Tier A", "Tier B1", "Tier B2", "Tier C", "Curtailed"],
    plot_colors
):
    bars = plt.bar(top10["Scenario"], top10[col], bottom=bottom, label=label, color=color)
    
    # Add data labels, formatted to 1 decimal place. 
    plt.bar_label(bars, label_type='center', fmt=lambda x: f'{x:.1f}' if x > 0.1 else '') 
    
    bottom += top10[col].values

plt.ylabel("Annual Energy (GWh)")
plt.title("HPP Performance – Granular Energy Balance (GWh)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()