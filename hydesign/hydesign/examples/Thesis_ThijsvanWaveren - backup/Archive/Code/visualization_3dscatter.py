# -*- coding: utf-8 -*-
"""
Visualize 3D Feasible Parameter Sweep
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Needed for 3D plotting in matplotlib
from mpl_toolkits.mplot3d import Axes3D 

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
# root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))

def plot_3d_feasible_sweep(csv_filepath):
    # 1. Load the data using the absolute path
    try:
        df = pd.read_csv(csv_filepath)
        print(f"✅ Successfully loaded {len(df)} feasible combinations.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_filepath}.")
        print("Please check that the file name matches exactly and is in the same folder as this script.")
        return

    # 2. Setup the 3D figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    fig.suptitle('Feasible Workload Combinations (SLA $\geq$ 99.9%)', 
                 fontsize=16, fontweight='bold', y=0.95)
    ax.set_title('Colored by Total Objective Value', fontsize=12, pad=10)

    # 3. Create the 3D Scatter Plot
    scatter = ax.scatter(
        df['Tier_A_MW'], 
        df['Tier_B1_MW'], 
        df['Tier_B2_MW'],
        c=df['Objective_Value'], 
        cmap='viridis',       
        s=100,                
        alpha=0.8,            
        edgecolors='black',   
        linewidth=0.5
    )

    # 4. Axes Labels & Formatting
    ax.set_xlabel('Tier A Baseload (MW)', fontsize=11, labelpad=10)
    ax.set_ylabel('Tier B1 Daily (MW)', fontsize=11, labelpad=10)
    ax.set_zlabel('Tier B2 Weekly (MW)', fontsize=11, labelpad=10)
    
    # 5. Colorbar Setup
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Objective Score (Value-Weighted Annual GWh)', rotation=270, labelpad=20, fontsize=12)

    # 6. Save and Show
    plt.tight_layout()
    
    png_path = os.path.join(current_dir, 'Thesis_Plot_3D_Feasible_Sweep.png')
    svg_path = os.path.join(current_dir, 'Thesis_Plot_3D_Feasible_Sweep.svg')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"✅ Plots saved as PNG and SVG in: {current_dir}")
    
    plt.show()

if __name__ == "__main__":
    # Build the absolute path to the CSV file
    file_name = 'Feasible_3D_Sweep_Results_99.9pct.csv' 
    target_csv_path = os.path.join(current_dir, file_name)
    
    plot_3d_feasible_sweep(target_csv_path)