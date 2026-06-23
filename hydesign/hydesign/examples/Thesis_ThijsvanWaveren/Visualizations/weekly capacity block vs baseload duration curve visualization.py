# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:07:43 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Generate a Capacity Duration Curve for Weekly Blocks
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_capacity_duration_curve():
    print("\n--- Generating Capacity Duration Curve ---")
    
    # 1. LOAD THE DATA
    # Using the exact path you provided earlier
    csv_fn = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\Results\Weekly\annual simulation weekly result\Weekly_Capacity_Blocks_yearlysimulation.csv"
    
    try:
        df_blocks = pd.read_csv(csv_fn)
        firm_loads = df_blocks['Total_Firm_Load_MW'].values
        print(f"✅ Successfully loaded data from CSV.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find CSV at {csv_fn}")
        return

# 2. SORT FOR DURATION CURVE
    sorted_loads = np.sort(firm_loads)[::-1]
    baseload = np.min(sorted_loads) 
    
    # FIX: Append the last value to give the 52nd block an endpoint
    loads_plot = np.append(sorted_loads, sorted_loads[-1])
    
    # FIX: X-axis goes from 0 to 52 (represents the borders of the weeks)
    weeks = np.arange(0, 53) 
    
    # 3. PLOT THE CURVE
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Firm Capacity Duration Curve', fontsize=16, fontweight='bold')

    # Area 1: The Traditional Baseload
    ax.fill_between(weeks, 0, baseload, step='post', color='#1f77b4', alpha=0.8, 
                    label=f'Annual Baseload ({baseload:.1f} MW)')
    
    # Area 2: The Rescued Capacity
    ax.fill_between(weeks, baseload, loads_plot, step='post', color='#ff7f0e', alpha=0.8, 
                    label='Weekly Capacity Blocks')
    
    # Draw the boundary line
    ax.axhline(baseload, color='red', linestyle='--', linewidth=2, 
               label='Static SLA')

    # Draw the outer edge
    ax.step(weeks, loads_plot, where='post', color='black', linewidth=1.5)

    # 4. FORMATTING
    ax.set_ylabel('Served Firm IT Capacity (MW)', fontsize=12)
    ax.set_xlabel('Duration (Weeks)', fontsize=12)
    
    # FIX: Set the plot window to show the full 0 to 52 range
    ax.set_xlim(0, 52)
    ax.set_ylim(0, np.max(sorted_loads) + 5)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=11)

    # Save and show
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_fn = os.path.join(current_dir, 'Thesis_Plot_Duration_Curve.svg')
    plt.tight_layout()
    fig.savefig(plot_fn, format='svg', dpi=300)
    print(f"✅ Plot successfully saved to: {plot_fn}")
    plt.show()

if __name__ == "__main__":
    plot_capacity_duration_curve()