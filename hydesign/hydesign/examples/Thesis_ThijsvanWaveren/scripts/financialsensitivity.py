# -*- coding: utf-8 -*-
"""
Thesis Financial Sensitivity Analysis Plot
Baseline vs. Scenario (+20% Wind, +100% BESS CAPEX/OPEX)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def generate_sensitivity_plot():
    # --- 1. DATA DEFINITION (in Millions €) ---
    # Baseline Costs
    capex_base = {
        'Wind': 90.000,
        'Solar': 41.400,
        'BESS': 13.925,
        'Shared': 41.529
    }
    opex_base = {
        'Wind': 1.260,
        'Solar': 1.368,
        'BESS': 0.000
    }
    lcoe_base = 115.27

    # Shock Scenario Costs (+20% Wind, +100% BESS)
    capex_shock = {
        'Wind': capex_base['Wind'] * 1.20,
        'Solar': capex_base['Solar'] * 1.00,
        'BESS': capex_base['BESS'] * 2.00,
        'Shared': capex_base['Shared'] * 1.00
    }
    opex_shock = {
        'Wind': opex_base['Wind'] * 1.20,
        'Solar': opex_base['Solar'] * 1.00,
        'BESS': opex_base['BESS'] * 2.00
    }
    
    # Estimate LCOE scaling based on CAPEX (~17.1% up) and OPEX (~9.6% up)
    # LCOE Shock approx 133.00 €/MWh based on prior math
    lcoe_shock = 133.00

    # --- 2. PLOT SETUP ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [1.5, 1.5, 1]})
    fig.suptitle('Financial Sensitivity: Baseline vs. +20% Wind & +100% BESS Costs', fontsize=16, fontweight='bold', y=0.98)

    labels = ['Baseline', 'Shock Scenario']
    x = np.arange(len(labels))
    width = 0.5
    
    colors = {
        'Wind': '#1f77b4',
        'Solar': '#ff7f0e',
        'BESS': '#9467bd',
        'Shared': '#7f7f7f'
    }

    # --- PANEL 1: CAPEX ---
    bottom_base = 0
    bottom_shock = 0
    for component in ['Wind', 'Solar', 'BESS', 'Shared']:
        # Baseline bar
        ax1.bar(x[0], capex_base[component], width, bottom=bottom_base, color=colors[component], edgecolor='white', label=component)
        bottom_base += capex_base[component]
        
        # Shock bar
        # Only add labels to the first bar to avoid duplicates in legend
        ax1.bar(x[1], capex_shock[component], width, bottom=bottom_shock, color=colors[component], edgecolor='white')
        
        # Add text annotation for the changed components in the shock bar
        if component == 'Wind':
            ax1.text(x[1], bottom_shock + (capex_shock[component]/2), f"+20%\n(+€18M)", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        if component == 'BESS':
            ax1.text(x[1], bottom_shock + (capex_shock[component]/2), f"+100%\n(+€13.9M)", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            
        bottom_shock += capex_shock[component]

    ax1.set_title('Total CAPEX Composition', fontsize=13)
    ax1.set_ylabel('Capital Expenditure (€ Millions)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    # Add total values on top
    ax1.text(x[0], bottom_base + 3, f"€{bottom_base:.1f}M", ha='center', fontweight='bold')
    ax1.text(x[1], bottom_shock + 3, f"€{bottom_shock:.1f}M", ha='center', fontweight='bold')
    ax1.set_ylim(0, max(bottom_base, bottom_shock) + 20)

    # --- PANEL 2: OPEX ---
    bottom_base_op = 0
    bottom_shock_op = 0
    for component in ['Wind', 'Solar', 'BESS']:
        ax2.bar(x[0], opex_base[component], width, bottom=bottom_base_op, color=colors[component], edgecolor='white')
        bottom_base_op += opex_base[component]
        
        ax2.bar(x[1], opex_shock[component], width, bottom=bottom_shock_op, color=colors[component], edgecolor='white')
        
        if component == 'Wind':
            ax2.text(x[1], bottom_shock_op + (opex_shock[component]/2), f"+20%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
            
        bottom_shock_op += opex_shock[component]

    ax2.set_title('Annual OPEX Composition', fontsize=13)
    ax2.set_ylabel('Annual Expenditure (€ Millions / yr)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.text(x[0], bottom_base_op + 0.05, f"€{bottom_base_op:.2f}M", ha='center', fontweight='bold')
    ax2.text(x[1], bottom_shock_op + 0.05, f"€{bottom_shock_op:.2f}M", ha='center', fontweight='bold')
    ax2.set_ylim(0, max(bottom_base_op, bottom_shock_op) + 0.3)

    # --- PANEL 3: LCOE IMPACT ---
    lcoe_values = [lcoe_base, lcoe_shock]
    bars = ax3.bar(x, lcoe_values, width=0.6, color=['#2ca02c', '#d62728'], edgecolor='black')
    
    ax3.set_title('Impact on LCOE Delivered', fontsize=13)
    ax3.set_ylabel('LCOED (€ / MWh)', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=11)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        if i == 1:
            perc_increase = ((lcoe_shock - lcoe_base) / lcoe_base) * 100
            ax3.text(bar.get_x() + bar.get_width()/2, yval + 2, f"€{yval:.1f}\n(+{perc_increase:.1f}%)", ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2, yval + 2, f"€{yval:.1f}", ha='center', va='bottom', fontweight='bold')
            
    ax3.set_ylim(0, max(lcoe_values) + 30)

    # --- FINAL FORMATTING & SAVE ---
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Leave room for suptitle
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    plot_fn_png = os.path.join(current_dir, 'Thesis_Plot_Financial_Sensitivity.png')
    plot_fn_svg = os.path.join(current_dir, 'Thesis_Plot_Financial_Sensitivity.svg')
    
    plt.savefig(plot_fn_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_fn_svg, format='svg', bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    generate_sensitivity_plot()