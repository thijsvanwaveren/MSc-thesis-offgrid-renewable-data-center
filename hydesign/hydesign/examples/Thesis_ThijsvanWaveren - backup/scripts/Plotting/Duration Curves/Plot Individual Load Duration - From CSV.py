# -*- coding: utf-8 -*-
"""
Section 3.7 - Fast LDC Post-Processing
Reads the previously saved 'Yearly_Operation_XXMW.csv' chronological data
and generates high-contrast Load Duration Curves in seconds.

Updates:
- Legend labels simplified to just the tier names (e.g., "Tier A").
- Legend border removed for a cleaner look.
- Legend dynamically ordered strictly from A -> B1 -> B2 -> C.
- Fixed right-axis text collision specifically for the 50 MW scenario.
"""

import os
import numpy as np
import pandas as pd
#import matplotlib.subplots
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =============================================================================
# 1. SETUP & PATHS
# =============================================================================
# The exact folder where your CSVs are saved
current_dir = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
# os.chdir(current_dir) # Uncomment if running standalone

# The capacities you simulated and saved
IT_CAPACITIES_MW = [16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0]

# High-Contrast Color Palette for easy distinction
C_A = '#08306b'         # Deep Navy (Firm)
C_B1 = '#17becf'        # Teal/Cyan (Daily)
C_B2 = '#d62728'        # Crimson Red (Weekly - Highlights the massive bursts)
C_C = '#7f7f7f'         # Neutral Grey (Opportunistic)
C_GRID = '#e0e0e0'

# =============================================================================
# 2. READ CSV AND PLOT LOOP
# =============================================================================
print("\n" + "=" * 80)
print(" FAST LOAD DURATION CURVE GENERATOR ".center(80))
print("=" * 80)

for cap_mw in IT_CAPACITIES_MW:
    csv_filename = f'Yearly_Operation_{cap_mw:.0f}MW.csv'
    file_path = os.path.join(current_dir, csv_filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Could not find {csv_filename}. Skipping {cap_mw} MW.")
        continue
        
    print(f"📊 Loading and Plotting {cap_mw} MW Facility...")
    
    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # 2. Extract and Sort for Load Duration (Descending)
    ldc_a = np.sort(df['Tier_A_MW'].values)[::-1]
    ldc_b1 = np.sort(df['Tier_B1_MW'].values)[::-1]
    ldc_b2 = np.sort(df['Tier_B2_MW'].values)[::-1]
    ldc_c = np.sort(df['Tier_C_MW'].values)[::-1]

    # Calculate actual realized averages
    avg_a = np.mean(df['Tier_A_MW'])
    avg_b1 = np.mean(df['Tier_B1_MW'])
    avg_b2 = np.mean(df['Tier_B2_MW'])
    avg_c = np.mean(df['Tier_C_MW'])

    x_pct = np.linspace(0, 100, len(df))

    # -------------------------------------------------------------------------
    # 3. PLOTTING THE LDC (Academic-Consulting Style)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='white')

    # Dictionary to collect plot handles for explicit legend ordering
    leg_handles = {}

    # Plot Lines & Ultra-Faint Volume Fills (Z-order ensures small tiers aren't hidden)
    leg_handles['Tier B2'], = ax.plot(x_pct, ldc_b2, color=C_B2, linewidth=2.5, zorder=4, label='Tier B2')
    ax.fill_between(x_pct, 0, ldc_b2, color=C_B2, alpha=0.08, zorder=1)

    leg_handles['Tier C'], = ax.plot(x_pct, ldc_c, color=C_C, linewidth=2, linestyle='-', zorder=2, label='Tier C')
    ax.fill_between(x_pct, 0, ldc_c, color=C_C, alpha=0.05, zorder=1)

    if avg_b1 > 0.1:
        leg_handles['Tier B1'], = ax.plot(x_pct, ldc_b1, color=C_B1, linewidth=2.5, zorder=3, label='Tier B1')
        ax.fill_between(x_pct, 0, ldc_b1, color=C_B1, alpha=0.08, zorder=1)

    leg_handles['Tier A'], = ax.plot(x_pct, ldc_a, color=C_A, linewidth=3.5, zorder=5, label='Tier A')
    ax.fill_between(x_pct, 0, ldc_a, color=C_A, alpha=0.15, zorder=1)

    # Add the Average Horizontal Lines
    ax.axhline(avg_a, color=C_A, linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)
    if avg_b1 > 0.1: 
        ax.axhline(avg_b1, color=C_B1, linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)
    if avg_b2 > 0.1: 
        ax.axhline(avg_b2, color=C_B2, linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)
    if avg_c > 0.1: 
        ax.axhline(avg_c, color=C_C, linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)

    # -------------------------------------------------------------------------
    # Text Annotations (Collision Handling for 50 MW)
    # -------------------------------------------------------------------------
    label_x = 101.5 # Push slightly outside the plot box
    
    y_pos_a = avg_a
    y_pos_c = avg_c
    
    # Specific fix to physically separate the Tier A and Tier C labels in the 50 MW plot
    if cap_mw == 50.0:
        y_pos_c += 1.5
        y_pos_a -= 1.0
        
    ax.text(label_x, y_pos_a, f'{avg_a:.1f} MW', color=C_A, va='center', fontweight='bold', fontsize=10, clip_on=False)
    
    if avg_b1 > 0.1: 
        ax.text(label_x, avg_b1, f'{avg_b1:.1f} MW', color=C_B1, va='center', fontweight='bold', fontsize=10, clip_on=False)
    if avg_b2 > 0.1: 
        ax.text(label_x, avg_b2, f'{avg_b2:.1f} MW', color=C_B2, va='center', fontweight='bold', fontsize=10, clip_on=False)
    if avg_c > 0.1: 
        ax.text(label_x, y_pos_c, f'{avg_c:.1f} MW', color=C_C, va='center', fontweight='bold', fontsize=10, clip_on=False)

    # Chart Formatting
    ax.set_ylabel("Instantaneous Hardware Capacity (MW)", fontsize=12, fontweight='bold', color='#444444')
    ax.set_xlabel("Percentage of the Year (%)", fontsize=12, fontweight='bold', color='#444444')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, cap_mw * 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    # Softer grid and professional spines
    ax.grid(axis='both', linestyle='-', alpha=0.2, color=C_GRID, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('#555555')

    # -------------------------------------------------------------------------
    # Explicit Legend Ordering (A -> B1 -> B2 -> C)
    # -------------------------------------------------------------------------
    ordered_keys = ['Tier A', 'Tier B1', 'Tier B2', 'Tier C']
    final_handles = []
    final_labels = []
    
    for key in ordered_keys:
        if key in leg_handles:
            final_handles.append(leg_handles[key])
            final_labels.append(key)
            
    # Frameon=False removes the box
    ax.legend(handles=final_handles, labels=final_labels, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=len(final_labels), 
              frameon=False, fontsize=11)

    # Adjust layout so the bottom legend doesn't get clipped
    plt.subplots_adjust(bottom=0.25, right=0.92)
    
    # Save & Show in IDE
    svg_filename = os.path.join(current_dir, f'Thesis_LDC_{cap_mw:.0f}MW.svg')
    plt.savefig(svg_filename, dpi=300, bbox_inches='tight')
    # plt.show() 

print("\n" + "=" * 80)
print(" All plots generated successfully! ")
print("=" * 80)