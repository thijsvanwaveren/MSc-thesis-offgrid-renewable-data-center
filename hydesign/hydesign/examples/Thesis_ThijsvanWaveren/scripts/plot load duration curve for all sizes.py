# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:15:20 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.7 - Fast LDC Post-Processing
Reads the previously saved 'Yearly_Operation_XXMW.csv' chronological data
and generates high-contrast Load Duration Curves in seconds.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =============================================================================
# 1. SETUP & PATHS
# =============================================================================
# The exact folder where your CSVs are saved
current_dir = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
os.chdir(current_dir)

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

    # Plot Lines & Ultra-Faint Volume Fills (Z-order ensures small tiers aren't hidden)
    ax.plot(x_pct, ldc_b2, color=C_B2, linewidth=2.5, zorder=4, label=f'Tier B2 (Weekly) - Avg: {avg_b2:.1f} MW')
    ax.fill_between(x_pct, 0, ldc_b2, color=C_B2, alpha=0.08, zorder=1)

    ax.plot(x_pct, ldc_c, color=C_C, linewidth=2, linestyle='-', zorder=2, label=f'Tier C (Opportunistic) - Avg: {avg_c:.1f} MW')
    ax.fill_between(x_pct, 0, ldc_c, color=C_C, alpha=0.05, zorder=1)

    if avg_b1 > 0.1:
        ax.plot(x_pct, ldc_b1, color=C_B1, linewidth=2.5, zorder=3, label=f'Tier B1 (Daily) - Avg: {avg_b1:.1f} MW')
        ax.fill_between(x_pct, 0, ldc_b1, color=C_B1, alpha=0.08, zorder=1)

    ax.plot(x_pct, ldc_a, color=C_A, linewidth=3.5, zorder=5, label=f'Tier A (Firm) - Avg: {avg_a:.1f} MW')
    ax.fill_between(x_pct, 0, ldc_a, color=C_A, alpha=0.15, zorder=1)

    # Add the Average Horizontal Lines
    ax.axhline(avg_a, color=C_A, linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)
    if avg_b1 > 0.1: 
        ax.axhline(avg_b1, color=C_B1, linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)
    if avg_b2 > 0.1: 
        ax.axhline(avg_b2, color=C_B2, linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)
    if avg_c > 0.1: 
        ax.axhline(avg_c, color=C_C, linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)

    # Explicit Right-Axis Annotations (clip_on=False prevents them from being cut off!)
    label_x = 101.5 # Push slightly outside the plot box
    ax.text(label_x, avg_a-1, f'{avg_a:.1f} MW', color=C_A, va='center', fontweight='bold', fontsize=10, clip_on=False)
    if avg_b1 > 0.1: 
        ax.text(label_x, avg_b1, f'{avg_b1:.1f} MW', color=C_B1, va='center', fontweight='bold', fontsize=10, clip_on=False)
    if avg_b2 > 0.1: 
        ax.text(label_x, avg_b2, f'{avg_b2:.1f} MW', color=C_B2, va='center', fontweight='bold', fontsize=10, clip_on=False)
    if avg_c > 0.1: 
        ax.text(label_x, avg_c, f'{avg_c:.1f} MW', color=C_C, va='center', fontweight='bold', fontsize=10, clip_on=False)

    # Chart Formatting
    #ax.set_title(f"Load Duration Curve: {cap_mw:.0f} MW Facility", fontsize=15, fontweight='bold', pad=20, color='#333333')
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

    # Legend (Moved to the bottom, auto-adjusting columns based on active tiers)
    num_active_tiers = sum(1 for avg in [avg_a, avg_b1, avg_b2, avg_c] if avg > 0.1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(2, num_active_tiers), 
              frameon=True, edgecolor='#e0e0e0', fontsize=11, facecolor='white')

    # Adjust layout so the bottom legend doesn't get clipped
    plt.subplots_adjust(bottom=0.25, right=0.92)
    
    # Save & Show in IDE
    svg_filename = os.path.join(current_dir, f'Thesis_LDC_{cap_mw:.0f}MW.svg')
    plt.savefig(svg_filename, dpi=300, bbox_inches='tight')
    plt.show() 

print("\n" + "=" * 80)
print(" All plots generated successfully! ")
print("=" * 80)