# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:16:44 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Evaluate Value of Workload Stacking (Including Tier C)
Generates a Single Stacked Bar Chart for the Fully Combined Workload.
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- HYDESIGN IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.assembly.assembly_rolling_horizon_thijs_10_3_26 import hpp_model_constant_output_offgrid as hpp_model

def configure_parameters(thesis_dir):
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    sim_pars['G_MW'] = 0
    sim_pars['battery_charge_efficiency'] = float(np.sqrt(0.86))
    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_offgrid_sweep_temp.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    return temp_fn

def plot_single_workload_stack(results):
    """Generates a single stacked bar chart for the final combined stage."""
    
    labels = ['Fully Stacked\nWorkload']
    
    # Extract results into single-element arrays for plotting
    tier_a_base_mw  = np.array([results['base']])
    tier_a_block_mw = np.array([results['blocks']])
    tier_b1_mw      = np.array([results['b1']])
    tier_b2_mw      = np.array([results['b2']])
    tier_c_mw       = np.array([results['c']])  # NEW: The Sponge
    curtailed_mw    = np.array([results['curt']])

    fig, ax = plt.subplots(figsize=(6, 8))

    # TU Delft Colors & Specific Hexes
    colors = {
        'Base': '#00549F',    # Annual Baseload (Dark Blue)
        'Blocks': '#73A4FF',  # Weekly Capacity Blocks (Light Blue)
        'B1': '#E07A5F',      # Tier B1 Daily (Coral)
        'B2': '#8B5A2B',      # Tier B2 Weekly (Brown)
        'C': '#00A6D6',       # Tier C Opportunistic (Green)
        'Curtail': '#E63946'  # Wasted / Curtailed (Red)
    }

    width = 0.8

    # Stack the bars sequentially
    b_base = ax.bar(labels, tier_a_base_mw, width, 
                    label='Tier A: Annual Baseload', color=colors['Base'], edgecolor='black')
    
    b_blocks = ax.bar(labels, tier_a_block_mw, width, bottom=tier_a_base_mw, 
                      label='Tier A: Weekly Blocks', color=colors['Blocks'], edgecolor='black')
    
    y_stack = tier_a_base_mw + tier_a_block_mw
    b_b1 = ax.bar(labels, tier_b1_mw, width, bottom=y_stack, 
                  label='Tier B1 (24h SLA)', color=colors['B1'], edgecolor='black')
    
    y_stack += tier_b1_mw
    b_b2 = ax.bar(labels, tier_b2_mw, width, bottom=y_stack, 
                  label='Tier B2 (168h SLA)', color=colors['B2'], edgecolor='black')
    
    # NEW: Add Tier C
    y_stack += tier_b2_mw
    b_c = ax.bar(labels, tier_c_mw, width, bottom=y_stack, 
                 label='Tier C (Opportunistic)', color=colors['C'], edgecolor='black')
                 
    y_stack += tier_c_mw
    b_curt = ax.bar(labels, curtailed_mw, width, bottom=y_stack, 
                    label='Curtailed Power', color=colors['Curtail'], edgecolor='black', hatch='//')

    ax.set_ylabel('Average Continuous Power (MW)', fontsize=12, fontweight='bold')
    ax.set_title('The Ultimate Workload Stack\nMaximizing Renewable Utility', fontsize=14, fontweight='bold', pad=20)

    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, 0.6)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=11, framealpha=1, edgecolor='black')

    # Add exact text annotations inside the bars
    def annotate_bar(value, current_bottom, color='white', text_color='white'):
        if value > 0.9:
            y_pos = current_bottom + (value / 2)
            ax.text(0, y_pos, f'{value:.1f} MW', ha='center', va='center', 
                    color=text_color, fontweight='bold', fontsize=11)

    annotate_bar(tier_a_base_mw[0], 0, text_color='white')
    annotate_bar(tier_a_block_mw[0], tier_a_base_mw[0], text_color='black')
    annotate_bar(tier_b1_mw[0], tier_a_base_mw[0] + tier_a_block_mw[0], text_color='black')
    annotate_bar(tier_b2_mw[0], tier_a_base_mw[0] + tier_a_block_mw[0] + tier_b1_mw[0], text_color='white')
    # Tier C Annotation
    annotate_bar(tier_c_mw[0], tier_a_base_mw[0] + tier_a_block_mw[0] + tier_b1_mw[0] + tier_b2_mw[0], text_color='white')
    
    # Curtailment (Needs special handling to not overlap if it's small)
    if curtailed_mw[0] > 1.0:
        y_pos = sum([tier_a_base_mw[0], tier_a_block_mw[0], tier_b1_mw[0], tier_b2_mw[0], tier_c_mw[0]]) + (curtailed_mw[0] / 2)
        ax.text(0, y_pos, f'{curtailed_mw[0]:.1f} MW', ha='center', va='center', 
                color='black', fontweight='bold', fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'Thesis_Single_Stack_Bar_With_TierC.png')
    plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot successfully saved to: {plot_fn}")
    plt.show()

def run_evaluation():
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    MAX_IT_CAPACITY_MW = 55.0 
    ANNUAL_FIRM_BASE = 7.0 * 0.95
    N_life_1yr = 8760
    N_life_total = 25 * 8760
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    # --- ACTIVATE TIER C SPONGE ---
    # By setting this negative, the solver earns a reward for dispatching leftover energy to C2
    os.environ['REWARD_C2'] = '-0.3'  

    # =========================================================================
    # 1. LOAD THE CSV AND CREATE PROFILES
    # =========================================================================
    csv_fn = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\Results\Weekly\annual simulation weekly result\Weekly_Capacity_Blocks_yearlysimulation.csv"
    
    try:
        df_blocks = pd.read_csv(csv_fn)
        weekly_firm_mw = df_blocks['Total_Firm_Load_MW'].values
    except FileNotFoundError:
        print(f"❌ Error: Could not find CSV at {csv_fn}")
        sys.exit(1)

    # Create the base 1-year profile and apply 0.95 scaling for 99.9% reliability
    dynamic_blocks_1yr = np.zeros(N_life_1yr)
    for w in range(52):
        dynamic_blocks_1yr[w*168 : (w+1)*168] = weekly_firm_mw[w]
    dynamic_blocks_1yr[52*168:] = weekly_firm_mw[-1]
    
    scaled_blocks_1yr = dynamic_blocks_1yr * 0.95
    scaled_blocks_25yr = np.tile(scaled_blocks_1yr, 25)

    fixed_capacity_profile = np.full(N_life_total, MAX_IT_CAPACITY_MW)
    
    chart_data = {}

    # =========================================================================
    # SINGLE RUN: BATCH WORKLOADS (+ 1 MW B1, + 20 MW B2, + Infinite C)
    # =========================================================================
    print(f"\n--- Running Final Evaluation: Fully Stacked Workloads (With Tier C) ---")
    tier_b1_25yr = np.full(N_life_total, 1.0 * 24.0)
    tier_b2_25yr = np.full(N_life_total, 20.0 * 168.0)
    
    hpp_batch = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=scaled_blocks_25yr, tier_b_profile=tier_b1_25yr, tier_b2_profile=tier_b2_25yr, 
        load_profile_ts=fixed_capacity_profile, battery_deg=False,
        run_mode='yearly' 
    )
    hpp_batch.evaluate(*fixed_design)
    
    # Extract arrays
    curt_ts = hpp_batch.prob.get_val('ems.hpp_curt_t')[:8760]
    unserved_ts = hpp_batch.prob.get_val('ems.Unserved_A')[:8760]
    hpp_t_ts = hpp_batch.prob.get_val('ems.hpp_t')[:8760]
    served_c2_ts = hpp_batch.prob.get_val('ems.Served_C2')[:8760] # Extract the Sponge!
    
    served_total_firm = scaled_blocks_1yr - unserved_ts
    
    # Tier B is whatever was served to the IT load, minus the Tier A firm load, minus Tier C
    served_b_total_ts = np.maximum(hpp_t_ts - served_total_firm - served_c2_ts, 0.0)
    
    # Simple average split: B1 gets up to 1 MW, B2 gets the rest
    avg_b_total = np.mean(served_b_total_ts)
    avg_b1 = min(1.0, avg_b_total)
    avg_b2 = max(0.0, avg_b_total - avg_b1)
    
    chart_data = {
        'base': np.mean(np.minimum(served_total_firm, ANNUAL_FIRM_BASE)),
        'blocks': np.mean(np.maximum(served_total_firm - ANNUAL_FIRM_BASE, 0.0)),
        'b1': avg_b1,
        'b2': avg_b2,
        'c': np.mean(served_c2_ts), # Add the sponge to the dictionary
        'curt': np.mean(curt_ts)
    }

    # =========================================================================
    # GENERATE PLOT
    # =========================================================================
    plot_single_workload_stack(chart_data)

if __name__ == "__main__":
    run_evaluation()