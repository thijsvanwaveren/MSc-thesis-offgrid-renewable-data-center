# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:15:58 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Standalone Plotting Script: Workload Cannibalization (2D Slices)
Generates a side-by-side comparison of Firm vs Daily (High Cannibalization) 
and Firm vs Weekly (Zero Cannibalization) within a 16 MW Data Center.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. SETUP & DATA EXTRACTION
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Replace with your actual sweep file name
sweep_file = os.path.join(current_dir, 'Feasible_3D_Sweep_Results_99.9pct_IT16.0.csv')

IT_CAPACITY = 16.0

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

    else:
        print("CSV not found. Generating synthetic Pareto fronts based on thesis text...")
        # Synthetic data matching your text: A maxes at 8, B1 maxes at 14, B2 maxes at 16.
        # B1 gets cannibalized faster than 1:1. B2 trades exactly 1:1 with hardware.
        tier_a = np.arange(0, 9, 1)
        max_b1 = np.array([14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0, 0.0])
        max_b2 = np.array([16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0])
        return tier_a, max_b1, max_b2

tier_a, max_b1, max_b2 = get_pareto_fronts()

# =============================================================================
# 2. GENERATE SIDE-BY-SIDE PLOT
# =============================================================================
# Removed sharey=True so the second plot renders its own Y-axis labels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')

# ---------------------------------------------------------
# PANEL 1: Tier A vs Tier B1 (High Cannibalization)
# ---------------------------------------------------------

# Feasible Line
ax1.plot(tier_a, max_b1, color='#ff7f0e', linewidth=3, marker='s', markersize=6, label='Max Feasible Tier A / B1 Combination')

# Shading: Feasible Area
ax1.fill_between(tier_a, 0, max_b1, color='#ff7f0e', alpha=0.2)

# Annotations for Ax1
ax1.text(3, 4, "FEASIBLE REGION", fontsize=11, fontweight='bold', color='#cc5a00', ha='center')

ax1.set_xlabel("Tier A Capacity (MW)", fontsize=12, fontweight='bold', color='#444444')
ax1.set_ylabel("Tier B1 Capacity (MW)", fontsize=12, fontweight='bold', color='#444444')

# ---------------------------------------------------------
# PANEL 2: Tier A vs Tier B2 (Zero Cannibalization)
# ---------------------------------------------------------

# Feasible Line
ax2.plot(tier_a, max_b2, color='#2ca02c', linewidth=3, marker='^', markersize=6, label='Max Feasible Tier A / B2 Combination')

# Shading: Feasible Area
ax2.fill_between(tier_a, 0, max_b2, color='#2ca02c', alpha=0.2)

# Annotations for Ax2
ax2.text(4, 5, "FEASIBLE REGION", fontsize=11, fontweight='bold', color='#1e7a1e', ha='center')

ax2.set_xlabel("Tier A Capacity (MW)", fontsize=12, fontweight='bold', color='#444444')
ax2.set_ylabel("Tier B2 Capacity (MW)",  fontsize=12, fontweight='bold', color='#444444')

# ---------------------------------------------------------
# FORMATTING & CLEANUP
# ---------------------------------------------------------
for ax in [ax1, ax2]:
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 16.5)
    ax.set_yticks(np.arange(0, 17, 2)) # Explicitly sets ticks to display 0 through 16 cleanly
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)

# Single comprehensive legend at the bottom
fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12, frameon=False)

plt.tight_layout()

# Save Plot
plot_fn = os.path.join(current_dir, 'Cannibalization_Comparison_16MW.svg')
plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
print(f"✅ Final Plot saved to: {plot_fn}")

plt.show()