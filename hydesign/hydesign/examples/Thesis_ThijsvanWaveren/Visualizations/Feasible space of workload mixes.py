# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:02:15 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Empirical Topographical Heatmap (2.5D Pareto Frontier)
Visualizes the 3D feasible workload space in a single contour plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.tri as tri
import matplotlib.colors as mcolors

# =============================================================================
# 1. LOAD THE EMPIRICAL DATA
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITY = 16
FILE_NAME = f"Feasible_3D_Sweep_Results_99.9pct_IT{IT_CAPACITY:.1f}.csv"

file_path = os.path.join(BASE_FOLDER, FILE_NAME)

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    # Create dummy data for demonstration if file is missing
    # df_sweep = pd.DataFrame({'Tier_B1_MW': [0, 0, 5], 'Tier_B2_MW': [0, 7, 0], 'Tier_A_MW': [8, 8, 4]})
    
df_sweep = pd.read_csv(file_path)

# =============================================================================
# 2. DATA PROCESSING (Extracting the Z-Axis)
# =============================================================================
# We want X = Tier B1, Y = Tier B2, Z = Max Tier A
heatmap_df = df_sweep.groupby(['Tier_B1_MW', 'Tier_B2_MW'])['Tier_A_MW'].max().reset_index()

x = heatmap_df['Tier_B1_MW'].values
y = heatmap_df['Tier_B2_MW'].values
z = heatmap_df['Tier_A_MW'].values

# Create an unstructured triangular grid from the pruned data
triang = tri.Triangulation(x, y)

# =============================================================================
# 3. VISUALIZATION
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

# Define the contour levels (0 MW to 8 MW for Tier A)
#levels = np.arange(0, 10, 1)
levels = np.arange(-0.5, 9.5, 1)

# CUSTOM COLORMAP: Slice the 'Blues' colormap so it starts at 20% intensity.
# This makes "0 MW" light blue, meaning pure white is strictly reserved for "infeasible/empty"
base_cmap = plt.get_cmap('Blues')
custom_blues = mcolors.LinearSegmentedColormap.from_list('trunc_blues', base_cmap(np.linspace(0.25, 1.0, 100)))

# Plot the filled contours
contour_filled = ax.tricontourf(triang, z, levels=levels, cmap=custom_blues, extend='neither', alpha=0.85, zorder=1)

# Add sharp topographical lines
contour_lines = ax.tricontour(triang, z, levels=levels, colors='#1A6587', linewidths=1.0, alpha=0.65, zorder=2)

# ---------------------------------------------------------
# ADDITIONS: Scatter points and IT Capacity Hatching
# ---------------------------------------------------------
# 1. Scatter the actual empirical data points
ax.scatter(x, y, c='black', s=30, alpha=0.75, marker='o', label='Simulated Feasible Combinations', zorder=3, edgecolors='white', linewidths=0.5)
# In your annotation loop, change the f-string:
# for i in range(len(x)):
#     ax.annotate(f"{round(z[i])}",  # <--- Use round() here instead of int()
#                 (x[i], y[i]), 
#                 textcoords="offset points", 
#                 xytext=(5, 5), 
#                 fontsize=8, 
#                 fontweight='bold',
#                 color='#333333',
#                 alpha=0.8,
#                 zorder=4)


# ---> NEW: Label the contour lines directly <---
#ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.0f', colors='#333333')

# 2. Diagonal IT Capacity Line (y = 16 - x)
x_it_line = np.array([0, IT_CAPACITY])
y_it_line = np.array([IT_CAPACITY, 0])
ax.plot(x_it_line, y_it_line, color='#d62728', linestyle='--', linewidth=1.5, 
        label=f'IT Capacity Limit ({IT_CAPACITY} MW)', zorder=4)

# 3. Red Hatching for everything ABOVE the IT Capacity
# Create a polygon fill above the x+y=16 line up to the top right corner
x_fill = np.linspace(0, 20, 100)
y_limit = np.maximum(IT_CAPACITY - x_fill, 0)
ax.fill_between(x_fill, y_limit, 20, facecolor='none', edgecolor='#d62728', hatch='//', 
                alpha=0.3, label='Infeasible (Exceeds IT Capacity)', zorder=4)

# ---------------------------------------------------------
# Styling & Formatting
# ---------------------------------------------------------
ax.set_title(f"Feasible Workload Combinations \n(IT Capacity = {IT_CAPACITY} MW & Reliability = 99.9%)", 
             fontsize=15, fontweight='bold', color='#333333', pad=20)

ax.set_xlabel("Tier B1 Capacity [MW]", color='#595959', fontweight='bold', fontsize=12)
ax.set_ylabel("Tier B2 Capacity [MW]", color='#595959', fontweight='bold', fontsize=12)

# Crop axes to 20 MW max
ax.set_xlim(0, 16)
ax.set_ylim(0, 16)

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

# Add Legend for the Scatter and Hatching
ax.legend(loc='upper right', framealpha=0.95, fontsize=10)

# =============================================================================
# 4. COLORBAR (The Z-Axis Legend)
# =============================================================================
# Change your cbar definition to explicitly state the ticks:
cbar = fig.colorbar(contour_filled, ax=ax, ticks=np.arange(0, 9, 1), pad=0.05, fraction=0.046)
#cbar = fig.colorbar(contour_filled, ax=ax, ticks=levels, pad=0.05, fraction=0.046)
cbar.set_label('Max Feasible Tier A [MW]', rotation=270, labelpad=20, fontweight='bold', color='#333333')
cbar.ax.tick_params(colors='#595959')

plt.tight_layout()
plt.savefig('Thesis_Topographical_Heatmap_Improved.png', dpi=300, bbox_inches='tight')
plt.show()