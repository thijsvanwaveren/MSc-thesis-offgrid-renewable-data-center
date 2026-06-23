# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:11:01 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.6 - Global Macroeconomic Equilibrium
The "Hero Plot": Optimal Data Center Sizing
(Integrated Calculation Engine + Ultimate Academic-Consulting Standard)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. MASTER PARAMETERS & STYLING
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITIES_MW = np.array([16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0])

# --- Financial Inputs (Adjust these for sensitivity analysis) ---
REV_MULTIPLIER = 1.0
CAPEX_MULTIPLIER = 1.0

REVENUE_PER_MWH = {"A": 4000.0 * REV_MULTIPLIER, "B1": 2800.0 * REV_MULTIPLIER, 
                   "B2": 1600.0 * REV_MULTIPLIER, "C": 400.0 * REV_MULTIPLIER}

OTHER_VARIABLE_OPEX_EUR_PER_MWH = 3  
DISCOUNT_RATE = 0.075 

FACILITY_CAPEX_PER_MW = 9.60   
FACILITY_LIFETIME = 20
IT_CAPACITIES_PER_MW = 15.33 * CAPEX_MULTIPLIER
IT_LIFETIME = 5
FIXED_OPEX_PER_MW = 0.25       
HPP_SUNK_COST_M = 15.95 

# --- Styling ---
C_NAVY = '#113b5e'    # Main Profit Line
C_ORANGE = '#e67e22'  # Tier C Line
C_PEAK = '#c0392b'    # The Optimum
halo = [path_effects.withStroke(linewidth=3, foreground="white", alpha=0.9)]

# =============================================================================
# 2. CALCULATION ENGINE
# =============================================================================
def calculate_crf(rate, years):
    if rate == 0: return 1 / years
    return (rate * (1 + rate)**years) / ((1 + rate)**years - 1)

ann_fixed_per_mw = (FACILITY_CAPEX_PER_MW * calculate_crf(DISCOUNT_RATE, FACILITY_LIFETIME)) + \
                   (IT_CAPACITIES_PER_MW * calculate_crf(DISCOUNT_RATE, IT_LIFETIME)) + \
                   FIXED_OPEX_PER_MW

y_no_c, y_with_c = [], []

print("Extracting sweep data and calculating Net Profit...")

for cap in IT_CAPACITIES_MW:
    file_name = f"Feasible_3D_Sweep_Results_99.9pct_IT{cap:.1f}.csv"
    file_path = os.path.join(BASE_FOLDER, file_name)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Synthetic fallback mimicking real HPP off-grid dynamics
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
                            'Total_Delivered_Annual_GWh': (a*0.9 + b1*0.5 + b2*0.3 + (cap*0.1)) * 8.76
                        })
        df = pd.DataFrame(rows)

    # Calculate Revenues
    rev_no_c = (df['Energy_A_Annual_GWh']*1000*REVENUE_PER_MWH["A"]) + \
               (df['Energy_B1_Annual_GWh']*1000*REVENUE_PER_MWH["B1"]) + \
               (df['Energy_B2_Annual_GWh']*1000*REVENUE_PER_MWH["B2"])
    
    e_total = df.get('Total_Delivered_Annual_GWh', df['Energy_A_Annual_GWh'] + df['Energy_B1_Annual_GWh'] + df['Energy_B2_Annual_GWh'])
    e_comm = df['Energy_A_Annual_GWh'] + df['Energy_B1_Annual_GWh'] + df['Energy_B2_Annual_GWh']
    e_c = np.maximum(0, e_total - e_comm)
    
    rev_with_c = rev_no_c + (e_c * 1000 * REVENUE_PER_MWH["C"])

    # Calculate Costs
    dc_water_opex = OTHER_VARIABLE_OPEX_EUR_PER_MWH * (e_total * 1000)
    total_fixed_costs_m = ann_fixed_per_mw * cap
    
    # Calculate Profit
    profit_no_c_m = (rev_no_c - dc_water_opex)/1e6 - HPP_SUNK_COST_M - total_fixed_costs_m
    profit_with_c_m = (rev_with_c - dc_water_opex)/1e6 - HPP_SUNK_COST_M - total_fixed_costs_m
    
    y_no_c.append(profit_no_c_m.max())
    y_with_c.append(profit_with_c_m.max())

y_no_c = np.array(y_no_c)
y_with_c = np.array(y_with_c)

# Identify the dynamic optimum
opt_idx = np.argmax(y_with_c)
opt_x = IT_CAPACITIES_MW[opt_idx]
opt_y = y_with_c[opt_idx]

# =============================================================================
# 3. VISUALIZATION
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

# --- PHASE SHADING ---
# Dynamically place the text labels relative to the max profit
text_y = max(y_with_c) * 0.45 

# # Phase 1: High Marginal Utility
# ax.axvspan(0, 30, color='#2ecc71', alpha=0.03) 
# ax.text(23, text_y, "STAGE I\nREVENUE > CAPEX", color='#27ae60', fontweight='bold', fontsize=10, ha='center', alpha=0.6)

# # Phase 2: Saturation / Optimum Zone
# ax.axvspan(30, 60, color='#f1c40f', alpha=0.03)
# ax.text(45, text_y, "STAGE II\nAPPROACHING OPTIMUM", color='#d4ac0d', fontweight='bold', fontsize=10, ha='center', alpha=0.6)

# # Phase 3: The Linear Capital Penalty
# ax.axvspan(60, 110, color='#e74c3c', alpha=0.03)
# ax.text(85, text_y, "STAGE III\nCAPEX > REVENUE", color='#c0392b', fontweight='bold', fontsize=10, ha='center', alpha=0.6)

# --- PLOT LINES ---

ax.plot(IT_CAPACITIES_MW, y_with_c, color=C_ORANGE, marker='D', markersize=7, 
        linestyle='--', linewidth=3, zorder=5, label='Total Net Profit (Incl. Tier C)')

ax.plot(IT_CAPACITIES_MW, y_no_c, color=C_NAVY, marker='o', markersize=7, 
        linestyle='-', linewidth=3, zorder=4, label='Net Profit (Tiers A, B1, and B2)')

ax.fill_between(IT_CAPACITIES_MW, y_no_c, y_with_c, color=C_ORANGE, alpha=0.1, label='Tier C Added Value')


# --- HIGHLIGHT THE OPTIMUM ---
# ax.scatter(opt_x, opt_y, facecolors='none', edgecolors=C_PEAK, s=400, linewidth=3, zorder=10)
# ax.annotate(f"GLOBAL OPTIMUM\n{int(opt_x)} MW", xy=(opt_x, opt_y+ (max(y_with_c)*0.05)), 
#             xytext=(opt_x, opt_y + (max(y_with_c)*0.12)), ha='center', fontsize=12, fontweight='bold', color=C_PEAK,
#             arrowprops=dict(arrowstyle='->', color=C_PEAK, lw=2), path_effects=halo)

# --- SELECTIVE DATA LABELS ---
for i, (cap, val) in enumerate(zip(IT_CAPACITIES_MW, y_with_c)):
    if cap in [30, 50, 75, 100]:
        ax.text(cap, val + (max(y_with_c)*0.03), f"€{int(val)}M", color='#d35400', fontweight='bold', ha='center', path_effects=halo)
    if cap in [16]:
        ax.text(cap-1, val + (max(y_with_c)*0.03), f"€{int(val)}M", color='#d35400', fontweight='bold', ha='center', path_effects=halo)

for i, (cap, val) in enumerate(zip(IT_CAPACITIES_MW, y_no_c)):
    if cap in [16, 50, 100]:
        ax.text(cap, val - (max(y_with_c)*0.07), f"€{int(val)}M", color=C_NAVY, fontweight='bold', ha='center', path_effects=halo)

# --- AXES & FORMATTING ---
# ax.set_title("Optimal Sizing of the Off-Grid Facility", fontsize=18, fontweight='bold', pad=30, color='#2c3e50')
ax.set_ylabel("Annualized Net Profit (Millions € / yr)", fontsize=13, fontweight='bold', color='#444444')
ax.set_xlabel("Installed IT Capacity (MW)", fontsize=13, fontweight='bold', color='#444444')

ax.set_xticks(IT_CAPACITIES_MW)
ax.set_xticklabels([f"{cap:.0f}" for cap in IT_CAPACITIES_MW], fontsize=11)
ax.set_xlim(12, 102)

# Dynamic Y-limits based on outcomes
y_max = max(y_with_c)
y_min = min(y_no_c)
#ax.set_ylim(y_min - (y_max*0.1), y_max + (y_max*0.15))
ax.set_ylim(0, y_max + (y_max*0.15))

ax.grid(axis='y', linestyle='-', alpha=0.2, color='#b0b0b0', zorder=0)
ax.axhline(0, color='#333333', linewidth=1.5, zorder=1)

for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(1.5)
    ax.spines[spine].set_color('#444444')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- LEGEND ---
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, 
          frameon=False, facecolor='white', edgecolor='#e0e0e0', framealpha=1, borderpad=1, fontsize=11)

plt.tight_layout()

save_path = os.path.join(BASE_FOLDER, 'Hero_Plot_Dynamic.svg')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Dynamic Plot saved to: {save_path}")

plt.show()