# -*- coding: utf-8 -*-
"""
Section 3.7 - Pure Facility Utilization Curves
Reads chronological CSVs, sorts by Total Simultaneous Load, and plots 
a pure monolithic area chart to reveal Total Active Servers vs. Idle Servers.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =============================================================================
# 1. SETUP & PATHS
# =============================================================================
current_dir = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
os.chdir(current_dir)

IT_CAPACITIES_MW = [16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0]

# Clean Consulting Palette
C_ACTIVE = '#2c3e50'    # Dark Slate Blue (Active Servers)
C_IDLE_EDGE = '#c0392b' # Deep Red (Highlights the danger of idle capital)
C_GRID = '#e0e0e0'

print("\n" + "=" * 80)
print(" GENERATING PURE FACILITY UTILIZATION CURVES ".center(80))
print("=" * 80)

# =============================================================================
# 2. MAIN LOOP
# =============================================================================
for cap_mw in IT_CAPACITIES_MW:
    csv_filename = f'Yearly_Operation_{cap_mw:.0f}MW.csv'
    file_path = os.path.join(current_dir, csv_filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping {cap_mw} MW (CSV not found).")
        continue
        
    print(f"📊 Plotting Pure Utilization Curve for {cap_mw} MW Facility...")
    df = pd.read_csv(file_path)
    
    # -------------------------------------------------------------------------
    # THE MATH: Sort ONLY the Total Load descending
    # -------------------------------------------------------------------------
    total_sorted = np.sort(df['Total_Hardware_Used_MW'].values)[::-1]
    x_pct = np.linspace(0, 100, len(df))

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    # 1. Active Hardware (The Monolithic Shape)
    ax.fill_between(x_pct, 0, total_sorted, color=C_ACTIVE, alpha=0.85, 
                    zorder=3, label='Active IT Hardware (All Tiers Combined)')
    ax.plot(x_pct, total_sorted, color='#1a252f', linewidth=2, zorder=4)

    # 2. Idle Hardware (The White Space above the curve)
    ax.fill_between(x_pct, total_sorted, cap_mw, 
                    facecolor='#fef9f8', edgecolor=C_IDLE_EDGE, hatch='///', alpha=0.6, 
                    linewidth=0, zorder=2, label='Idle Servers')

    # 3. Facility Limit Line
    ax.axhline(cap_mw, color='#333333', linestyle='-', linewidth=2.5, zorder=5)
    ax.text(101.5, cap_mw, f'Facility Limit\n({cap_mw} MW)', color='#333333', 
            va='center', fontweight='bold', fontsize=10, clip_on=False)

    # Chart Formatting
    #ax.set_title(f"Combined Load Duration Curve: {cap_mw:.0f} MW Data Center", 
                 #fontsize=15, fontweight='bold', pad=20, color='#333333')
    ax.set_ylabel("Total Hardware Utilized (MW)", fontsize=12, fontweight='bold', color='#444444')
    ax.set_xlabel("Percentage of the Year (%)", fontsize=12, fontweight='bold', color='#444444')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, cap_mw * 1.1) 
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    ax.grid(axis='both', linestyle='-', alpha=0.3, color=C_GRID, zorder=1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('#555555')

    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, 
              frameon=False, edgecolor='#e0e0e0', fontsize=11, facecolor='white')

    plt.subplots_adjust(bottom=0.25, right=0.92)
    
    # Save & Show in IDE
    svg_filename = os.path.join(current_dir, f'Thesis_Pure_Utilization_{cap_mw:.0f}MW.svg')
    plt.savefig(svg_filename, dpi=300, bbox_inches='tight')
    plt.show()

print("\n" + "=" * 80)
print(" All pure duration curves generated successfully! ")
print("=" * 80)