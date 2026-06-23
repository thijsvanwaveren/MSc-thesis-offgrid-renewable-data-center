# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:24:30 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:55:59 2026

@author: thijs
"""

import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the absolute outer edge (Pareto frontier) of your scatter plot
tier_a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
max_tier_b1 = np.array([24, 23, 22, 21, 19, 18, 17, 15, 14, 13, 12, 9, 9, 7, 6, 5, 4, 3, 2])

# Initialize figure
fig, ax = plt.subplots(figsize=(11, 6.5))

# Define clean, professional colors (using TU Delft House Style)
color_a = '#00549F'   # TU Delft dark blue for firm baseload
color_b1 = '#E07A5F'  # Contrasting orange for flexible load

# Create the stacked bars
bars_a = ax.bar(tier_a, tier_a, label='Tier A (Firm Baseload)', color=color_a, edgecolor='black', zorder=3)
bars_b1 = ax.bar(tier_a, max_tier_b1, bottom=tier_a, label='Tier B1 (Daily Batch)', color=color_b1, edgecolor='black', zorder=3)

# Add text annotations inside the bars for absolute clarity
for i in range(len(tier_a)):
    # Annotate Tier A (Only label if height is >= 2 to prevent vertical clutter in tiny bars)
    if tier_a[i] >= 2:
        ax.text(tier_a[i], tier_a[i]/2, f'{tier_a[i]}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    
    # Annotate Tier B1
    if max_tier_b1[i] >= 2:
        ax.text(tier_a[i], tier_a[i] + max_tier_b1[i]/2, f'{max_tier_b1[i]}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    
    # Annotate Total Capacity on top of the stacked bar
    total = tier_a[i] + max_tier_b1[i]
    ax.text(tier_a[i], total + 0.3, f'{total}', ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)

# Formatting and aesthetics
ax.set_xlabel('Tier A Baseload (MW)', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Capacity (MW)', fontsize=12, fontweight='bold')
ax.set_title('Firm baseload vs. Daily batch trade-off (≥ 99.0% Reliability)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(tier_a)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.legend(loc='upper right', fontsize=11, framealpha=1)
ax.set_ylim(0, 26.5) # Leave breathing room at the top for labels

plt.tight_layout()
plt.savefig('pareto_stacked_bars.png', dpi=300)