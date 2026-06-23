# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:24:41 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Perfect Foresight vs. Rolling Horizon Comparison
Focus: Tier A (Baseload) Reliability and Yield
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
thesis_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

# Import the rolling horizon assembly!
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

def plot_solver_comparison(df):
    """Generates a comparison plot between Perfect Foresight and Rolling Horizon"""
    df_perfect = df[df['Run_Mode'] == 'perfect_foresight']
    df_rolling = df[df['Run_Mode'] == 'rolling']

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Perfect Foresight vs. Rolling Horizon (Tier A Baseload)', fontsize=14, fontweight='bold')

    # Plot Perfect Foresight
    ax.plot(df_perfect['Tier_A_MW'], df_perfect['Reliability_Time_A'], 
            marker='o', linestyle='--', color='gray', linewidth=2, markersize=8, 
            label='Perfect Foresight (Knows weather 1 yr in advance)')
    
    # Plot Rolling Horizon
    ax.plot(df_rolling['Tier_A_MW'], df_rolling['Reliability_Time_A'], 
            marker='s', linestyle='-', color='#1f77b4', linewidth=3, markersize=8, 
            label='Rolling Horizon (7-day forecast, daily control)')

    ax.axhline(99.9, color='red', linestyle=':', alpha=0.5, label='99.9% Target')
    
    ax.set_ylabel('SLA Reliability (%)', fontsize=12)
    ax.set_xlabel('Tier A Load Capacity (MW)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'Thesis_Plot_Solver_Comparison.svg')
    fig.savefig(plot_fn, format='svg', dpi=300)
    print(f"\n✅ Plot successfully saved to: {plot_fn}")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" SOLVER COMPARISON: PERFECT FORESIGHT VS ROLLING HORIZON ".center(80, '='))
    print("="*80 + "\n")

    # 1. FIXED PARAMETERS
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    N_life = 25 * 8760
    MAX_IT_CAPACITY_MW = 100.0
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    # Disable C2 for a clean Tier A test
    os.environ['REWARD_C2'] = '1.0'  
    
    # 2. THE SWEEP (Kept small to save time! Expand later)
    tier_a_sweep = [8, 10, 15, 20]
    run_modes = ['perfect_foresight', 'rolling']
    all_results = []

    for mode in run_modes:
        print(f"\n--- Starting Sweep with Mode: '{mode.upper()}' ---")
        
        for a_mw in tier_a_sweep:
            print(f"  Evaluating Tier A = {a_mw} MW ... ", end="", flush=True)
            
            # Create Profiles
            t_a_ts = np.full(N_life, a_mw)
            t_zero = np.zeros(N_life)
            total_load = t_a_ts.copy()
            total_load[0] = MAX_IT_CAPACITY_MW 
            
            # Run Model
            hpp = hpp_model(
                latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
                num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
                tier_a_profile=t_a_ts, tier_b_profile=t_zero, tier_b2_profile=t_zero, load_profile_ts=total_load, 
                battery_deg=False, run_mode=mode
            )
            hpp.evaluate(*fixed_design)
            
            # Extract Metrics
            prob = hpp.prob
            rel_a = 100.0 * (1.0 - (np.sum(prob.get_val('ems.Unserved_A') > 1e-3) / N_life))
            
            print(f"Reliability: {rel_a:.2f}%")
            
            all_results.append({
                'Run_Mode': mode,
                'Tier_A_MW': a_mw,
                'Reliability_Time_A': rel_a
            })

    # 3. PLOT RESULTS
    df_results = pd.DataFrame(all_results)
    plot_solver_comparison(df_results)