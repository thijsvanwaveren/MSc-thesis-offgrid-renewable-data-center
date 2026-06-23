# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:06:19 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
3x3 Grid: Feasible Flexible Workload Regions (Color-coded by Contribution Margin)
Visualizes the Pareto Frontier and shaded feasible region for specific Tier A slices.
Pinpoints ONLY the global optimum with a professional marker.
Handles 1D edge cases (e.g., Tier A = 7, 8) by mapping the gradient onto the frontier line.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import griddata

warnings.filterwarnings("ignore")

# =============================================================================
# 1. MASTER USER INPUTS & FINANCIAL PARAMETERS
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITY = 16.0  
FILE_NAME = f"Feasible_3D_Sweep_Results_99.9pct_IT{IT_CAPACITY:.1f}.csv"
file_path = os.path.join(BASE_FOLDER, FILE_NAME)

# Financial Assumptions for Contribution Margin
REVENUE_PER_MWH = {
    "A": 4000.0, 
    "B1": 2800.0, 
    "B2": 1600.0
}
OTHER_VARIABLE_OPEX_EUR_PER_MWH = 3

# Plotting Settings
CMAP_STYLE = 'plasma' 

# =============================================================================
# 2. LOAD OR GENERATE DATA
# =============================================================================
if os.path.exists(file_path):
    df_sweep = pd.read_csv(file_path)
    print(f"Successfully loaded empirical data: {FILE_NAME}")
else:
    print(f"[WARNING] File not found: {file_path}. Generating synthetic data for visualization...")
    rows = []
    for a in range(0, int(IT_CAPACITY) + 1):
        for b1 in range(0, int(IT_CAPACITY) + 1):
            for b2 in range(0, int(IT_CAPACITY) + 1):
                # Synthetic feasibility condition
                if (a*1.5 + b1*1.1 + b2*1.3) <= IT_CAPACITY: 
                    rows.append({
                        'Tier_A_MW': a,
                        'Tier_B1_MW': b1,
                        'Tier_B2_MW': b2,
                        'Energy_A_Annual_GWh': a * 8.76 * 0.9,
                        'Energy_B1_Annual_GWh': b1 * 8.76 * 0.5,
                        'Energy_B2_Annual_GWh': b2 * 8.76 * 0.3,
                        'Total_Delivered_Annual_GWh': (a*0.9 + b1*0.5 + b2*0.3) * 8.76,
                        'LCOE_delivered': np.random.uniform(40, 60)
                    })
    df_sweep = pd.DataFrame(rows)

# =============================================================================
# 3. CALCULATE FINANCIALS (CONTRIBUTION MARGIN)
# =============================================================================
for col in ["Energy_A_Annual_GWh", "Energy_B1_Annual_GWh", "Energy_B2_Annual_GWh", "Total_Delivered_Annual_GWh"]:
    if col in df_sweep.columns:
        df_sweep[col.replace("_GWh", "_MWh")] = df_sweep[col] * 1000.0

df_sweep["Energy_Delivered_MWh"] = (df_sweep.get("Energy_A_Annual_MWh", 0) + 
                                    df_sweep.get("Energy_B1_Annual_MWh", 0) + 
                                    df_sweep.get("Energy_B2_Annual_MWh", 0))

df_sweep["Revenue_EUR"] = (df_sweep.get("Energy_A_Annual_MWh", 0) * REVENUE_PER_MWH["A"] + 
                           df_sweep.get("Energy_B1_Annual_MWh", 0) * REVENUE_PER_MWH["B1"] + 
                           df_sweep.get("Energy_B2_Annual_MWh", 0) * REVENUE_PER_MWH["B2"])

energy_col = "Total_Delivered_Annual_MWh" if "Total_Delivered_Annual_MWh" in df_sweep.columns else "Energy_Delivered_MWh"
df_sweep["Total_HPP_Cost_EUR"] = df_sweep["LCOE_delivered"] * df_sweep[energy_col] 
df_sweep["DC_Variable_OPEX_EUR"] = OTHER_VARIABLE_OPEX_EUR_PER_MWH * df_sweep["Energy_Delivered_MWh"]
df_sweep["Total_Variable_OPEX_EUR"] = df_sweep["Total_HPP_Cost_EUR"] + df_sweep["DC_Variable_OPEX_EUR"]

# Contribution Margin (Millions of EUR)
df_sweep["Contribution_Margin_M_EUR"] = (df_sweep["Revenue_EUR"] - df_sweep["Total_Variable_OPEX_EUR"]) / 1e6

# =============================================================================
# 4. PREPARE GLOBAL OPTIMUM & SCALING
# =============================================================================
VMIN = df_sweep["Contribution_Margin_M_EUR"].min()
VMAX = df_sweep["Contribution_Margin_M_EUR"].max()
contour_levels = np.linspace(VMIN, VMAX, 30)

# ---> FIND THE SINGLE GLOBAL OPTIMUM <---
global_max_idx = df_sweep['Contribution_Margin_M_EUR'].idxmax()
opt_A = df_sweep.loc[global_max_idx, 'Tier_A_MW']
opt_B1 = df_sweep.loc[global_max_idx, 'Tier_B1_MW']
opt_B2 = df_sweep.loc[global_max_idx, 'Tier_B2_MW']
opt_Margin = df_sweep.loc[global_max_idx, 'Contribution_Margin_M_EUR']

print(f"Global Optimum Mix -> Tier A: {opt_A}, B1: {opt_B1}, B2: {opt_B2} (Margin: {opt_Margin:.1f} M€)")

# =============================================================================
# 5. CREATE THE 3x3 GRID VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=True, sharey=True, facecolor='white')
axes = axes.flatten()

COLOR_IT_LIMIT = '#d62728'       
COLOR_FRONTIER = '#00ffcc'       

for i, ax in enumerate(axes):
    if i > 8: break 
        
    A_slice = i 
    df_slice = df_sweep[df_sweep['Tier_A_MW'] == A_slice]
    
    x_feas = df_slice['Tier_B1_MW'].values
    y_feas = df_slice['Tier_B2_MW'].values
    c_feas = df_slice['Contribution_Margin_M_EUR'].values
    
    if len(x_feas) > 0: 
        
        # Check if we have enough 2D spread to do a contour fill
        is_2d = len(np.unique(x_feas)) > 1 and len(np.unique(y_feas)) > 1 and len(x_feas) > 3
        
        if is_2d:
            # ---> STANDARD 2D CONTOUR GRADIENT <---
            grid_x, grid_y = np.mgrid[0:IT_CAPACITY:200j, 0:IT_CAPACITY:200j]
            grid_z = griddata((x_feas, y_feas), c_feas, (grid_x, grid_y), method='linear')
            
            ax.contourf(grid_x, grid_y, grid_z, levels=contour_levels, cmap=CMAP_STYLE, 
                        extend='both', alpha=0.9, zorder=2)
            
            # Faint inner contour lines
            ax.contour(grid_x, grid_y, grid_z, levels=contour_levels, colors='black', 
                       linewidths=0.3, alpha=0.4, zorder=3)
        else:
            # ---> 1D FALLBACK FOR NARROW SLICES (e.g. A=7, A=8) <---
            # Scatter points densely to act as a colored line gradient
            ax.scatter(x_feas, y_feas, c=c_feas, cmap=CMAP_STYLE, vmin=VMIN, vmax=VMAX, 
                       s=150, marker='s', alpha=0.9, zorder=2, label='1D Margin Gradient')
            
        # CALCULATE AND PLOT THE PARETO FRONTIER 
        frontier_x = list(np.unique(x_feas))
        frontier_y = [y_feas[x_feas == fx].max() for fx in frontier_x]
        
        if frontier_x[0] > 0:
            frontier_x.insert(0, frontier_x[0])
            frontier_y.insert(0, 0)
        if frontier_y[-1] > 0:
            frontier_x.append(frontier_x[-1])
            frontier_y.append(0)
        
        #ax.plot(frontier_x, frontier_y, color=COLOR_FRONTIER, linewidth=2.5, 
                #zorder=5, label='Pareto Frontier (Local Max)')

        # ---> ADDED: GLOBAL OPTIMUM HIGHLIGHT <---
        # Only plot the optimal marker if it belongs on this specific slice
        if A_slice == opt_A:
            # Professional Diamond Marker
            ax.plot(opt_B1, opt_B2, marker='D', markersize=12, color='white', 
                    markeredgecolor='black', markeredgewidth=2, zorder=10, 
                    label='Global Optimum Mix')
            
            # Sleek Annotation Box pointing to the optimum
            ax.annotate(f'Global Optimum\n({opt_Margin:.1f} M€)', 
                        xy=(opt_B1, opt_B2), 
                        xytext=(opt_B1 + 1.5, opt_B2 + 2.5),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                        fontsize=11, fontweight='bold', zorder=11,
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.5, alpha=0.9))

    # PHYSICAL IT CAPACITY LIMITS 
    remaining_it = IT_CAPACITY - A_slice
    
    ax.plot([0, remaining_it], [remaining_it, 0], 
            color=COLOR_IT_LIMIT, linestyle='--', linewidth=2, alpha=0.7,
            zorder=6, label=f'Remaining IT Cap ({remaining_it} MW)')
    
    x_fill = np.linspace(0, 20, 100)
    y_limit = np.maximum(remaining_it - x_fill, 0)
    ax.fill_between(x_fill, y_limit, 20, facecolor='none', edgecolor=COLOR_IT_LIMIT, 
                    hatch='//', alpha=0.15, zorder=1, label='Infeasible (Exceeds Hardware)')

    # Subplot Formatting
    ax.set_title(f"Tier A = {A_slice} MW", fontsize=12, fontweight='bold', color='#333333')
    ax.set_xlim(0, IT_CAPACITY + 0.5)
    ax.set_ylim(0, IT_CAPACITY + 0.5)
    
    if i >= 6: ax.set_xlabel("Tier B1 Capacity [MW]", fontweight='bold', color='#595959')
    if i % 3 == 0: ax.set_ylabel("Tier B2 Capacity [MW]", fontweight='bold', color='#595959')

# =============================================================================
# 6. MASTER FORMATTING, LEGEND & COLORBAR
# =============================================================================
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles)) 

contour_proxy = Patch(facecolor='#8e44ad', edgecolor='black', alpha=0.8)
by_label['Margin Gradient'] = contour_proxy

# Add the Global Optimum marker to the legend manually so it appears at the top
opt_proxy = Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='black', markersize=9, markeredgewidth=1.5)
by_label['Global Optimum Mix'] = opt_proxy

# Enforce logical legend order
ordered_labels = ['Margin Gradient', 'Global Optimum Mix', 
                  f'Remaining IT Cap ({remaining_it} MW)', 'Infeasible (Exceeds Hardware)']
ordered_handles = [by_label[lbl] for lbl in ordered_labels if lbl in by_label]

fig.legend(ordered_handles, [lbl.split(' (')[0] if 'Remaining' in lbl else lbl for lbl in ordered_labels], 
           loc='upper center', bbox_to_anchor=(0.47, 0.96), 
           ncol=5, framealpha=0.95, fontsize=11)

fig.suptitle(f"Feasible Workload Regions across Tier A Constraints\n(IT Capacity = {IT_CAPACITY} MW | Colored by Exact Contribution Margin)", 
             fontsize=18, fontweight='bold', y=1.02)

# Adjust layout to make strict room for the colorbar
plt.tight_layout(rect=[0, 0, 0.9, 0.93])

# Explicit Colorbar using ScalarMappable
cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.75]) 
sm = cm.ScalarMappable(cmap=CMAP_STYLE, norm=mcolors.Normalize(vmin=VMIN, vmax=VMAX))
sm.set_array([])

cbar_ticks = np.linspace(VMIN, VMAX, 10)
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=cbar_ticks)

cbar.ax.set_yticklabels([f"{val:.1f} M€" for val in cbar_ticks])
cbar.set_label('Contribution Margin (Millions € / Year)', rotation=270, labelpad=25, fontsize=12, fontweight='bold')

# Save and Show
save_path = os.path.join(BASE_FOLDER, 'Thesis_3x3_Workload_Grid_GlobalOptimum.svg')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {save_path}")
plt.show()