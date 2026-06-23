# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:36:17 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Advanced Visualization for 3D Workload Sweep
Generates Heatmap Facets, Top-10 Ranked Stacked Bar, and Pareto Frontier
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap_atlas(df, current_dir):
    unique_b2 = sorted(df['Tier_B2_MW'].unique())
    
    if len(unique_b2) > 6:
        idx = np.round(np.linspace(0, len(unique_b2) - 1, 6)).astype(int)
        unique_b2 = [unique_b2[i] for i in idx]

    n_plots = len(unique_b2)
    cols = min(3, n_plots)
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
    fig.suptitle('Feasible Workload Atlas: Value by Composition', fontsize=16, fontweight='bold', y=0.98)
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    vmin = df['Objective_Value'].min()
    vmax = df['Objective_Value'].max()

    for i, b2_val in enumerate(unique_b2):
        ax = axes[i]
        slice_df = df[df['Tier_B2_MW'] == b2_val]
        
        pivot_table = slice_df.pivot(index='Tier_B1_MW', columns='Tier_A_MW', values='Objective_Value')
        pivot_table = pivot_table.sort_index(ascending=False)
        
        sns.heatmap(pivot_table, ax=ax, cmap='viridis', vmin=vmin, vmax=vmax,
                    annot=True, fmt=".1f", annot_kws={"size": 8}, 
                    cbar=False, linewidths=.5, linecolor='gray')
        
        ax.set_title(f'Slice: Tier B2 = {b2_val} MW', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tier A (MW)')
        ax.set_ylabel('Tier B1 (MW)')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', aspect=40, pad=0.08, fraction=0.05)
    cbar.set_label('Objective Score (Value-Weighted Annual GWh)', fontsize=12, fontweight='bold')

    plt.savefig(os.path.join(current_dir, 'Thesis_Plot_Heatmap_Atlas.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_top_10_compositions(df, current_dir):
    top_df = df.sort_values(by='Objective_Value', ascending=False).head(10).reset_index(drop=True)
    
    if top_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle('Top 10 Optimal Workload Compositions', fontsize=16, fontweight='bold')

    x = np.arange(len(top_df))
    width = 0.6

    p_a = ax1.bar(x, top_df['Tier_A_MW'], width, label='Tier A (Baseload)', color='#1f77b4', edgecolor='black')
    p_b1 = ax1.bar(x, top_df['Tier_B1_MW'], width, bottom=top_df['Tier_A_MW'], label='Tier B1 (Daily)', color='#ff7f0e', edgecolor='black')
    p_b2 = ax1.bar(x, top_df['Tier_B2_MW'], width, bottom=top_df['Tier_A_MW'] + top_df['Tier_B1_MW'], label='Tier B2 (Weekly)', color='#8c564b', edgecolor='black')

    ax1.set_ylabel('Workload Capacity (MW)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Rank (1 = Highest Objective Value)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Rank {i+1}" for i in x])
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    p_val, = ax2.plot(x, top_df['Objective_Value'], color='#d62728', marker='D', markersize=8, linewidth=2, linestyle='--', label='Objective Value')
    ax2.set_ylabel('Objective Score', fontsize=12, fontweight='bold', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(top_df['Objective_Value'].min() * 0.9, top_df['Objective_Value'].max() * 1.1)

    lines = [p_a, p_b1, p_b2, p_val]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

    plt.savefig(os.path.join(current_dir, 'Thesis_Plot_Top10_Stack.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_pareto_front(df, current_dir):
    # Calculate Total Capacity
    df['Total_Capacity_MW'] = df['Tier_A_MW'] + df['Tier_B1_MW'] + df['Tier_B2_MW']
    
    # Calculate the fraction of the total load that is rigid Tier A (for color mapping)
    df['Tier_A_Ratio'] = df['Tier_A_MW'] / df['Total_Capacity_MW']
    
    # Create a string label for the exact composition
    df['Composition'] = df.apply(lambda r: f"A:{r['Tier_A_MW']:.0f}|B1:{r['Tier_B1_MW']:.0f}|B2:{r['Tier_B2_MW']:.0f}", axis=1)
    
    # Find the maximum Objective Value for each Total Capacity level (The Frontier)
    idx_frontier = df.groupby('Total_Capacity_MW')['Objective_Value'].idxmax()
    frontier_df = df.loc[idx_frontier].sort_values(by='Total_Capacity_MW')

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Pareto Frontier: System Value vs. Total Workload Capacity', fontsize=16, fontweight='bold')

    # Scatter all feasible points, colored by their Tier A Ratio
    # Increase zorder slightly so these are on top of the dashed line
    scatter = ax.scatter(
        df['Total_Capacity_MW'], 
        df['Objective_Value'], 
        c=df['Tier_A_Ratio'], 
        cmap='plasma',       
        alpha=0.6,
        s=60,
        edgecolors='black',
        label='Feasible Combinations',
        zorder=2
    )

    # VISUAL CHANGE: Separate Line and Marker plotting to make the frontier dots use the colormap.
    
    # 1. Plot ONLY the red dashed line (set marker='' to remove original markers)
    # Set zorder lower so dots sit on top of the line.
    ax.plot(
        frontier_df['Total_Capacity_MW'], 
        frontier_df['Objective_Value'], 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        marker='', # REMOVES THE MARKERS FROM THIS LINE
        label='Max Value Frontier',
        zorder=1
    )
    
    # 2. Add specific, annotated scatter points FOR THE FRONTIER.
    # These now draw their color from 'Tier_A_Ratio' and the 'plasma' colormap, NOT red.
    ax.scatter(
        frontier_df['Total_Capacity_MW'], 
        frontier_df['Objective_Value'], 
        c=frontier_df['Tier_A_Ratio'], 
        cmap='plasma', 
        s=60, 
        edgecolors='black',
        linewidth=1.0,
        zorder=3
    )

    # Annotate the exact composition for the Frontier points
    for i, row in frontier_df.iterrows():
        # Prevent overlaps by rotating text callouts, which gets busy on a frontier
        # Instead, try a simple, angled offset
        ax.annotate(
            row['Composition'],
            (row['Total_Capacity_MW'], row['Objective_Value']),
            xytext=(-15, 20),              # angled offset to minimize overlap
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=45,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
        )

    ax.set_xlabel('Total Capacity (Tier A + B1 + B2) [MW]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objective Score (Value-Weighted Annual GWh)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Headroom for annotations
    ax.set_ylim(df['Objective_Value'].min() * 0.95, df['Objective_Value'].max() * 1.25)
    
    ax.legend(loc='lower right', fontsize=11)

    # Colorbar now shows the composition makeup
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Rigid Workload Fraction (Tier A / Total)', rotation=270, labelpad=15)

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'Thesis_Plot_Pareto_Frontier_Annotated_NoRedDots.png')
    plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    
    csv_file = 'Feasible_3D_Sweep_Results_99.9pct_IT16.0.csv' 
    csv_path = os.path.join(current_dir, csv_file)
    
    try:
        df_results = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df_results)} feasible runs.")
        
        print("\nGenerating Heatmap Atlas...")
        plot_heatmap_atlas(df_results, current_dir)
        
        print("\nGenerating Top 10 Decision Chart...")
        plot_top_10_compositions(df_results, current_dir)

        print("\nGenerating Pareto Frontier...")
        plot_pareto_front(df_results, current_dir)
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_file}")