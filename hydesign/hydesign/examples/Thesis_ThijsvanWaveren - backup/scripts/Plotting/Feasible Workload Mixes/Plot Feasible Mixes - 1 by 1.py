# -*- coding: utf-8 -*-
"""
Visualizes workload capacity cannibalization across different flexibility tiers.

Generates a side-by-side comparison of Firm vs. Daily (Tier B1) and Firm vs. 
Weekly (Tier B2) workloads for a bounded IT capacity. Extracts the Pareto front 
from 3D parameter sweep results to demonstrate how flexibility timescales 
impact the maximum feasible workload combinations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. SETUP & DATA EXTRACTION
# =============================================================================

BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITY = 16.0
RELIABILITY_TARGET = 99.9

FILE_NAME = f"Feasible_3D_Sweep_Results_99.9pct_IT{IT_CAPACITY:.1f}.csv"
sweep_file = os.path.join(BASE_FOLDER, FILE_NAME)

def get_pareto_fronts():
    """
    Extracts the absolute maximum B1 (when B2=0) and maximum B2 (when B1=0) 
    for every Tier A value. Uses synthetic data if CSV is not found.
    """
    if os.path.exists(sweep_file):
        df = pd.read_csv(sweep_file)
        # Filter for 99.9% reliability just in case
        if 'Reliability' in df.columns:
            df = df[df['Reliability'] >= 99.9]
            
        # Group to find maximums
        df_b1_only = df[df['Tier_B2_MW'] == 0]
        max_b1_per_a = df_b1_only.groupby('Tier_A_MW')['Tier_B1_MW'].max().reset_index()
        
        df_b2_only = df[df['Tier_B1_MW'] == 0]
        max_b2_per_a = df_b2_only.groupby('Tier_A_MW')['Tier_B2_MW'].max().reset_index()
        
        return max_b1_per_a['Tier_A_MW'].values, max_b1_per_a['Tier_B1_MW'].values, max_b2_per_a['Tier_B2_MW'].values


tier_a, max_b1, max_b2 = get_pareto_fronts()
hardware_limit = IT_CAPACITY - tier_a

# =============================================================================
# 2. GENERATE SIDE-BY-SIDE PLOT
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='white', sharey=True)

# ---------------------------------------------------------
# PANEL 1: Tier A vs Tier B1 (High Cannibalization)
# ---------------------------------------------------------
# Hardware Limit Line
# ax1.plot(tier_a, hardware_limit, color='#d62728', linestyle='--', linewidth=2, label='IT Hardware Limit (16 MW)')

# Feasible Line
ax1.plot(tier_a, max_b1, color='#ff7f0e', linewidth=3, marker='s', markersize=6, label='Max Feasible Tier B1')

# Shading: Feasible Area
ax1.fill_between(tier_a, 0, max_b1, color='#ff7f0e', alpha=0.2)

# Shading: The Battery Constraint Penalty (Gap between hardware and feasible)
# ax1.fill_between(tier_a, max_b1, hardware_limit, facecolor="none", edgecolor='#d62728', hatch='//', alpha=0.5)

# Annotations for Ax1
ax1.text(3, 4, "FEASIBLE REGION", fontsize=11, fontweight='bold', color='#cc5a00', ha='center')
#ax1.text(5, 12, "BATTERY PENALTY\n(Cannibalized Capacity)", fontsize=10, fontweight='bold', color='#d62728', ha='center',
         #bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

#ax1.set_title("A. Firm vs. Daily Workloads", fontsize=14, fontweight='bold', color='#333333', pad=10)
ax1.set_xlabel("Allocated Tier A (Firm Load) [MW]", fontsize=12, fontweight='bold', color='#444444')
ax1.set_ylabel("Max Feasible B1 Load [MW]", fontsize=12, fontweight='bold', color='#444444')

# ---------------------------------------------------------
# PANEL 2: Tier A vs Tier B2 (Zero Cannibalization)
# ---------------------------------------------------------
# Hardware Limit Line
# ax2.plot(tier_a, hardware_limit, color='#d62728', linestyle='--', linewidth=2)

# Feasible Line
ax2.plot(tier_a, max_b2, color='#2ca02c', linewidth=3, marker='^', markersize=6, label='Max Feasible Tier B2')

# Shading: Feasible Area
ax2.fill_between(tier_a, 0, max_b2, color='#2ca02c', alpha=0.2)
# ax2.fill_between(tier_a, max_b2, hardware_limit, facecolor="none", edgecolor='#d62728', hatch='//', alpha=0.5)

# Annotations for Ax2
ax2.text(4, 5, "FEASIBLE REGION", fontsize=11, fontweight='bold', color='#1e7a1e', ha='center')
# ax2.annotate("Perfect Hardware Utilization\n(No Battery Penalty)", 
#              xy=(2, 14), xytext=(4, 10),
#              arrowprops=dict(facecolor='#333333', arrowstyle="->", lw=1.5),
#              fontsize=10, fontweight='bold', color='#333333', ha='center')

#ax2.set_title("B. Firm vs. Weekly Workloads", fontsize=14, fontweight='bold', color='#333333', pad=10)
ax2.set_xlabel("Allocated Tier A (Firm Load) [MW]", fontsize=12, fontweight='bold', color='#444444')
ax2.set_ylabel("Max Feasible B2 Load (MW)",  fontsize=12, fontweight='bold', color='#444444')
# ---------------------------------------------------------
# FORMATTING & CLEANUP
# ---------------------------------------------------------
for ax in [ax1, ax2]:
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 16.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)

# Single comprehensive legend at the bottom
fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12, frameon=False)

plt.tight_layout()

# Save Plot
plot_fn = os.path.join(BASE_FOLDER, 'Cannibalization_Comparison_16MW.svg')
plt.savefig(plot_fn, dpi=300, bbox_inches='tight')

plt.show()