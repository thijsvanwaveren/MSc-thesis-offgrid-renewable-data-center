# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:02:37 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.3 - Data Center Scaling Effect
Generates:
1. IT Hardware Efficiency Stacked Bar Chart
2. HPP Power Balance Stacked Bar Chart
3. Trade-off Line Graph (Unutilized Hardware vs. Curtailment)
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- HYDESIGN IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

# =============================================================================
# GLOBAL COLOR PALETTE & STYLING
# =============================================================================
# Workload Tiers (Darkest to Lightest)
C_A = '#08306b'      # Deep Navy
C_B1 = '#2879b9'     # Strong Blue
C_B2 = '#73b3d8'     # Soft Blue
C_C = '#c8ddf0'      # Pale Blue

C_COOLING = '#d9d9d9' # Neutral Light Grey

# Differentiated "Waste" Styling (Supervisor Feedback)
# 1. Curtailed Power (Aggressive, active waste)
C_CURTAIL_FACE = 'white'
C_CURTAIL_EDGE = '#e74c3c'  # Sharp Red
HATCH_CURTAIL = '//'

# 2. Unutilized Hardware (Passive empty space/metal)
C_UNUTILIZED_FACE = 'white'
C_UNUTILIZED_EDGE = '#8c564b' # Muted Brown/Grey-Red
HATCH_UNUTILIZED = 'xx'

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_hardware_efficiency_scaling(df_results, save_dir):
    """
    Generates a stacked bar chart showing IT hardware utilization vs capacity.
    """
    df = df_results.sort_values(by='IT_Capacity_MW').reset_index(drop=True)
    conversion_factor = 1000 / 8760.0 / 25
    
    it_capacity = df['IT_Capacity_MW'].values
    labels = [f"{mw:.1f} MW" for mw in it_capacity]
    x = np.arange(len(labels))
    width = 0.5
    
    avg_power_a = df['Tier_A_GWh'].values * conversion_factor
    avg_power_b1 = df['Tier_B1_GWh'].values * conversion_factor
    avg_power_b2 = df['Tier_B2_GWh'].values * conversion_factor
    avg_power_c = df['Tier_C_GWh'].values * conversion_factor
    
    total_avg_power = avg_power_a + avg_power_b1 + avg_power_b2 + avg_power_c
    unutilized_power = np.maximum(0, it_capacity - total_avg_power)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    bottom_tracker = np.zeros(len(labels))
    
    ax.bar(x, avg_power_a, width, label='Tier A', color=C_A, bottom=bottom_tracker)
    bottom_tracker += avg_power_a
    ax.bar(x, avg_power_b1, width, label='Tier B1', color=C_B1, bottom=bottom_tracker)
    bottom_tracker += avg_power_b1
    ax.bar(x, avg_power_b2, width, label='Tier B2', color=C_B2, bottom=bottom_tracker)
    bottom_tracker += avg_power_b2
    ax.bar(x, avg_power_c, width, label='Tier C', color=C_C, bottom=bottom_tracker)
    bottom_tracker += avg_power_c
    
    # Unutilized Hardware (Empty Racks visual)
    ax.bar(x, unutilized_power, width, bottom=bottom_tracker, 
           facecolor=C_UNUTILIZED_FACE, edgecolor=C_UNUTILIZED_EDGE, hatch=HATCH_UNUTILIZED, linewidth=1.5, 
           label='Unutilized IT Capacity')
    
    ax.set_title("Data Center Hardware Efficiency", fontsize=14, pad=15, color='#333333', fontweight='bold')
    ax.set_ylabel("Average Served Power (MW)", color='#444444', fontweight='bold')
    ax.set_xlabel("Installed IT Capacity (MW)", color='#444444', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.9)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'Thesis_Hardware_Efficiency.svg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Hardware Efficiency Plot saved to: {save_path}")


def plot_power_balance_scaling(df_results, save_dir):
    """
    Generates a stacked bar chart showing the HPP power breakdown.
    """
    df = df_results.sort_values(by='IT_Capacity_MW').reset_index(drop=True)
    conversion_factor = 1000 / 8760.0 / 25
    
    labels = [f"{mw:.1f} MW" for mw in df['IT_Capacity_MW']]
    x = np.arange(len(labels))
    width = 0.5
    
    tier_a = df['Tier_A_GWh'].values * conversion_factor
    tier_b1 = df['Tier_B1_GWh'].values * conversion_factor
    tier_b2 = df['Tier_B2_GWh'].values * conversion_factor
    tier_c = df['Tier_C_GWh'].values * conversion_factor
    curtailment = df['Curtailment_GWh'].values * conversion_factor
    
    total_it = tier_a + tier_b1 + tier_b2 + tier_c
    cooling = total_it * 0.15 
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    bottom_tracker = np.zeros(len(labels))
    
    ax.bar(x, tier_a, width, label='Tier A', color=C_A, bottom=bottom_tracker)
    bottom_tracker += tier_a
    ax.bar(x, tier_b1, width, label='Tier B1', color=C_B1, bottom=bottom_tracker)
    bottom_tracker += tier_b1
    ax.bar(x, tier_b2, width, label='Tier B2', color=C_B2, bottom=bottom_tracker)
    bottom_tracker += tier_b2
    ax.bar(x, tier_c, width, label='Tier C', color=C_C, bottom=bottom_tracker)
    bottom_tracker += tier_c
    ax.bar(x, cooling, width, label='Cooling (PUE 1.15)', color=C_COOLING, bottom=bottom_tracker)
    bottom_tracker += cooling
    
    # Curtailed Power (Spilled electrons visual)
    ax.bar(x, curtailment, width, bottom=bottom_tracker, 
           facecolor=C_CURTAIL_FACE, edgecolor=C_CURTAIL_EDGE, hatch=HATCH_CURTAIL, linewidth=1.5,
           label='Curtailed Power')
    
    ax.set_title("Hybrid Power Plant (HPP) Power Balance", fontsize=14, pad=15, color='#333333', fontweight='bold')
    ax.set_ylabel("Average Served Power (MW)", color='#444444', fontweight='bold')
    ax.set_xlabel("Installed IT Capacity (MW)", color='#444444', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.9)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'Thesis_Power_Balance.svg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Power Balance Plot saved to: {save_path}")


def plot_structural_tradeoff(df_results, save_dir):
    """
    Generates a line graph mapping Unutilized Sprint Capacity vs Curtailed Power.
    This clearly visualizes the economic tradeoff intersection.
    """
    df = df_results.sort_values(by='IT_Capacity_MW').reset_index(drop=True)
    conversion_factor = 1000 / 8760.0 / 25
    
    it_capacity = df['IT_Capacity_MW'].values
    
    # Calculate Data
    total_avg_it_power = (df['Tier_A_GWh'] + df['Tier_B1_GWh'] + df['Tier_B2_GWh'] + df['Tier_C_GWh']).values * conversion_factor
    unutilized_power = np.maximum(0, it_capacity - total_avg_it_power)
    curtailed_power = df['Curtailment_GWh'].values * conversion_factor
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # Plot Lines
    ax.plot(it_capacity, curtailed_power, marker='o', markersize=8, linewidth=3, 
            color=C_CURTAIL_EDGE, label='Curtailed Power')
            
    ax.plot(it_capacity, unutilized_power, marker='s', markersize=8, linewidth=3, 
            color=C_UNUTILIZED_EDGE, label='Unutilized IT Capacity')
            
    # Formatting
    ax.set_title("Data Center Sizing Trade-off", fontsize=15, pad=15, fontweight='bold', color='#333333')
    ax.set_ylabel("Average Power (MW)", fontsize=12, fontweight='bold', color='#444444')
    ax.set_xlabel("Installed IT Capacity (MW)", fontsize=12, fontweight='bold', color='#444444')
    
    ax.set_xticks(it_capacity)
    ax.set_xticklabels([f"{mw:.0f}" for mw in it_capacity])
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Find Intersection (roughly) for annotation
    idx = np.argwhere(np.diff(np.sign(unutilized_power - curtailed_power))).flatten()
    # if len(idx) > 0:
    #     ax.axvline(x=it_capacity[idx[0]]+5, color='grey', linestyle=':', linewidth=2, alpha=0.7)
    #     ax.text(it_capacity[idx[0]]+7, max(curtailed_power)*0.8, 
    #             "Trade-off\nCrossover", color='grey', fontweight='bold', ha='left')

    ax.legend(loc='upper center', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'Thesis_Tradeoff_LineGraph.svg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Trade-off Plot saved to: {save_path}")

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Synthetic dataset matching your example
    data = [
     {'IT_Capacity_MW': 16.0, 'Tier_A_GWh': 1751, 'Tier_B1_GWh': 0, 'Tier_B2_GWh': 1533, 'Tier_C_GWh': 189, 'Curtailment_GWh': 9978},
     {'IT_Capacity_MW': 20.0, 'Tier_A_GWh': 1751, 'Tier_B1_GWh': 0, 'Tier_B2_GWh': 2409, 'Tier_C_GWh': 97, 'Curtailment_GWh': 9075},
     {'IT_Capacity_MW': 30.0, 'Tier_A_GWh': 1751, 'Tier_B1_GWh': 0, 'Tier_B2_GWh': 3723, 'Tier_C_GWh': 502, 'Curtailment_GWh': 7089},
     {'IT_Capacity_MW': 40.0, 'Tier_A_GWh': 1751, 'Tier_B1_GWh': 0, 'Tier_B2_GWh': 4380, 'Tier_C_GWh': 1255, 'Curtailment_GWh': 5462},
     {'IT_Capacity_MW': 50.0, 'Tier_A_GWh': 1751, 'Tier_B1_GWh': 0, 'Tier_B2_GWh': 5037, 'Tier_C_GWh': 1797, 'Curtailment_GWh': 4087},
     {'IT_Capacity_MW': 75.0, 'Tier_A_GWh': 1751, 'Tier_B1_GWh': 0, 'Tier_B2_GWh': 6132, 'Tier_C_GWh': 2847, 'Curtailment_GWh': 1654},
     {'IT_Capacity_MW': 100.0, 'Tier_A_GWh': 1751, 'Tier_B1_GWh': 0, 'Tier_B2_GWh': 6351, 'Tier_C_GWh': 3580, 'Curtailment_GWh': 613},
    ]
    df_results = pd.DataFrame(data)
    
    plot_hardware_efficiency_scaling(df_results, current_dir)
    plot_power_balance_scaling(df_results, current_dir)
    plot_structural_tradeoff(df_results, current_dir)