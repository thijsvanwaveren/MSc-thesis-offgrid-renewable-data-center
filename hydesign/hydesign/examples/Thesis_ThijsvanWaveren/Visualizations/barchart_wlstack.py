# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:33:45 2026

@author: thijs
"""

import matplotlib.pyplot as plt
import numpy as np

# --- 1. PASTE YOUR GWh/25y ARRAYS HERE ---
# (Using approximate data from your previous charts as placeholders)
labels = ['Stage 1\n(Firm Load Only)', 'Stage 2\n(+Daily Flex)', 'Stage 3\n(+Weekly Flex)', 'Stage 4\n(+Fully Flex)']

tier_a_gwh = np.array([1751, 1751, 1751, 1751])
tier_b1_gwh = np.array([0, 0, 0, 0])
tier_b2_gwh = np.array([0, 0, 6132, 6132])
tier_c_gwh = np.array([0, 0, 0, 2516])
curtailed_gwh = np.array([11967, 11967, 4912, 2027])

# --- 2. CONVERT GWh/25y TO AVERAGE CONTINUOUS MW ---
HOURS_IN_25_YEARS = 25 * 8760

tier_a_mw = (tier_a_gwh * 1000) / HOURS_IN_25_YEARS
tier_b1_mw = (tier_b1_gwh * 1000) / HOURS_IN_25_YEARS
tier_b2_mw = (tier_b2_gwh * 1000) / HOURS_IN_25_YEARS
tier_c_mw = (tier_c_gwh * 1000) / HOURS_IN_25_YEARS
curtailed_mw = (curtailed_gwh * 1000) / HOURS_IN_25_YEARS

# Calculate total delivered computing power for the summary label
total_delivered_mw = tier_a_mw + tier_b1_mw + tier_b2_mw + tier_c_mw

# --- 3. PLOTTING ---
fig, ax = plt.subplots(figsize=(11, 7))

# TU Delft House Style inspired colors
colors = {
    'A': '#00549F',      
    'B1': '#E07A5F',     
    'B2': '#8B5A2B',     
    'C': '#00A6D6',      
    'Curtail': '#E63946' 
}

width = 0.55

# Stack the bars
bars_a = ax.bar(labels, tier_a_mw, width, label='Tier A (Firm Load)', color=colors['A'], edgecolor='black')
bars_b1 = ax.bar(labels, tier_b1_mw, width, bottom=tier_a_mw, label='Tier B1 (Daily Batch)', color=colors['B1'], edgecolor='black')
bars_b2 = ax.bar(labels, tier_b2_mw, width, bottom=tier_a_mw+tier_b1_mw, label='Tier B2 (Weekly Batch)', color=colors['B2'], edgecolor='black')
bars_c = ax.bar(labels, tier_c_mw, width, bottom=tier_a_mw+tier_b1_mw+tier_b2_mw, label='Tier C (Opportunistic)', color=colors['C'], edgecolor='black')

# Curtailment with hatching
bars_curt = ax.bar(labels, curtailed_mw, width, bottom=total_delivered_mw, 
                   label='Curtailed Power (Wasted)', color=colors['Curtail'], edgecolor='black', hatch='//')

# --- 4. FORMATTING AND LABELS ---
ax.set_ylabel('Power Served (MW)', fontsize=12, fontweight='bold')
ax.set_title('Workload Stack (99.9% Reliability)', fontsize=14, fontweight='bold', pad=20)

ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax.set_axisbelow(True)

# Put legend outside
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, framealpha=1, edgecolor='black')

# --- 5. ADD EXACT MW ANNOTATIONS INSIDE BARS ---
for i in range(len(labels)):
    
    # Label Tier A
    if tier_a_mw[i] > 1.0:
        ax.text(i, tier_a_mw[i]/2, f'{tier_a_mw[i]:.1f} MW', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    # Label Tier B1
    if tier_b1_mw[i] > 0.5:
        y_pos = tier_a_mw[i] + (tier_b1_mw[i] / 2)
        ax.text(i, y_pos, f'{tier_b1_mw[i]:.1f} MW', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        
    # Label Tier B2
    if tier_b2_mw[i] > 0.5:
        y_pos = tier_a_mw[i] + tier_b1_mw[i] + (tier_b2_mw[i] / 2)
        ax.text(i, y_pos, f'{tier_b2_mw[i]:.1f} MW', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        
    # Label Tier C
    if tier_c_mw[i] > 0.5:
        y_pos = tier_a_mw[i] + tier_b1_mw[i] + tier_b2_mw[i] + (tier_c_mw[i] / 2)
        ax.text(i, y_pos, f'{tier_c_mw[i]:.1f} MW', ha='center', va='center', color='black', fontweight='bold', fontsize=10)


plt.tight_layout()
plt.savefig('average_mw_stacked_bar.png', dpi=300, bbox_inches='tight')