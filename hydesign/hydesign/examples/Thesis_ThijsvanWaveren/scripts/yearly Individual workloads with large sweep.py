# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:08:07 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.1.2 - Workload Flexibility vs Reliability
Simulates and plots the reliability of individual workload tiers.
High-resolution 1 MW sweeps starting just before the drop-off points.
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

from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model

# ==============================================================================
# PLOTTING FUNCTION
# ==============================================================================
def generate_thesis_plots(df1, df2, df2b):
    """
    Generates a single, combined plot showing the reliability drop-off
    for all three tiers, featuring drop-lines for the 99.9% threshold.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    # Plot the 3 curves
    ax.plot(df1['Tier_A_MW'], df1['Reliability_Time_A'], marker='o', markersize=6, color='#1f77b4', linewidth=2.5, label='Tier A (Firm Load)')
    ax.plot(df2['Tier_B_MW'], df2['Reliability_Deadline_B'], marker='s', markersize=6, color='#ff7f0e', linewidth=2.5, label='Tier B1 (Daily Flexible)')
    ax.plot(df2b['Tier_B2_MW'], df2b['Reliability_Deadline_B2'], marker='^', markersize=6, color='#2ca02c', linewidth=2.5, label='Tier B2 (Weekly Flexible)')

    # Add the 99.9% Target Line
    ax.axhline(99.9, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(2, 99.9 + 1, "Strict Reliability Target (99.9%)", color='grey', fontsize=10, fontweight='bold', va='bottom')

    # --- ADD VERTICAL DROP LINES & LABELS ---
    # These coordinates are based on your theoretical break points
    drop_points = [
        (8, df1, '#1f77b4', '8 MW Max\n(Firm)'),
        (15, df2, '#ff7f0e', '15 MW Max\n(Daily)'),
        (35, df2b, '#2ca02c', '35 MW Max\n(Weekly)')
    ]

    for x_val, df, color, text in drop_points:
        ax.vlines(x=x_val, ymin=45, ymax=99.9, color=color, linestyle=':', linewidth=2)
        ax.text(x_val, 48, text, color=color, fontsize=10, fontweight='bold', ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor=color, alpha=0.9, boxstyle='round,pad=0.3'))

    # Formatting
    ax.set_title('Impact of Workload Flexibility on Maximum Reliable IT Capacity', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('SLA Reliability (%)', fontsize=12, fontweight='bold', color='grey')
    ax.set_xlabel('Theoretical Workload Capacity (MW)', fontsize=12, fontweight='bold', color='grey')
    
    # Set limits
    ax.set_ylim(45, 105) 
    ax.set_xlim(0, 52) # Keep X-axis starting at 0 for visual grounding

    # Clean axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save Plot
    plot_fn = os.path.join(current_dir, 'Flexibility_Reliability_Curves_1MW.png')
    plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_fn}")
    plt.show()

# ==============================================================================
# SIMULATION FUNCTION
# ==============================================================================
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

def run_firm_yield_analysis():
    # File Paths updated to '_1MW' to force a fresh run instead of loading old 5MW data
    file_a = os.path.join(current_dir, 'data_tier_a_rel_1MW.csv')
    file_b1 = os.path.join(current_dir, 'data_tier_b1_rel_1MW.csv')
    file_b2 = os.path.join(current_dir, 'data_tier_b2_rel_1MW.csv')

    # IF DATA EXISTS: Skip Simulation and Just Plot
    if os.path.exists(file_a) and os.path.exists(file_b1) and os.path.exists(file_b2):
        print("Data files found! Skipping simulation and generating plot...")
        df1 = pd.read_csv(file_a)
        df2 = pd.read_csv(file_b1)
        df2b = pd.read_csv(file_b2)
        generate_thesis_plots(df1, df2, df2b)
        return

    # IF DATA DOES NOT EXIST: Run Simulation
    print("Data files not found. Starting High-Resolution HPP simulations...")
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    MAX_IT_CAPACITY_MW = 300
    N_life = 25 * 8760
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    # Specific 1 MW Sweeps starting right before the expected drop-offs
    sweep_a = np.arange(7, 51, 1)
    sweep_b1 = np.arange(13, 51, 1)
    sweep_b2 = np.arange(33, 51, 1)

    # --- STAGE 1: TIER A ---
    print(f"\nSimulating Tier A (from {sweep_a[0]} to {sweep_a[-1]} MW)...")
    results_stage1 = []
    for a_mw in sweep_a:
        t_a_ts = np.full(N_life, a_mw)
        t_b_ts = np.zeros(N_life)
        t_b2_ts = np.zeros(N_life)
        total_load = t_a_ts.copy()
        total_load[0] = MAX_IT_CAPACITY_MW 
        
        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b_ts, tier_b2_profile=t_b2_ts, load_profile_ts=total_load, battery_deg=False
        )
        hpp.evaluate(*fixed_design)
        rel_a = 1.0 - (np.sum(hpp.prob.get_val('ems.Unserved_A') > 1e-3) / N_life)
        results_stage1.append({"Tier_A_MW": a_mw, "Reliability_Time_A": rel_a * 100})
        
        # EARLY STOPPING
        if rel_a < 0.40:
            break

    # --- STAGE 2: TIER B1 ---
    print(f"Simulating Tier B1 (from {sweep_b1[0]} to {sweep_b1[-1]} MW)...")
    SAFE_BASE_A = 0.0
    results_stage2 = []
    for b_mw in sweep_b1:
        t_a_ts = np.full(N_life, SAFE_BASE_A)
        t_b_daily_target_ts = np.full(N_life, b_mw * 24.0) 
        t_b2_ts = np.zeros(N_life)
        total_load = t_a_ts + np.full(N_life, b_mw)
        total_load[0] = MAX_IT_CAPACITY_MW

        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_target_ts, tier_b2_profile=t_b2_ts, load_profile_ts=total_load, battery_deg=False
        )
        hpp.evaluate(*fixed_design)
        shortfall_b = hpp.prob.get_val('ems.Shortfall_B')
        rel_b1 = 1.0 - (np.sum(np.sum(shortfall_b.reshape(-1, 24), axis=1) > 1e-3) / (N_life/24))
        results_stage2.append({"Tier_B_MW": b_mw, "Reliability_Deadline_B": rel_b1 * 100})
        
        # EARLY STOPPING
        if rel_b1 < 0.40:
            break

    # --- STAGE 3: TIER B2 ---
    print(f"Simulating Tier B2 (from {sweep_b2[0]} to {sweep_b2[-1]} MW)...")
    SAFE_BASE_B1 = 0.0
    results_stage2b = []
    for b2_mw in sweep_b2:
        t_a_ts = np.full(N_life, SAFE_BASE_A)
        t_b_daily_ts = np.full(N_life, SAFE_BASE_B1 * 24.0)
        t_b2_weekly_target_ts = np.full(N_life, b2_mw * 168.0)
        total_load = t_a_ts + np.full(N_life, SAFE_BASE_B1) + np.full(N_life, b2_mw)
        total_load[0] = MAX_IT_CAPACITY_MW

        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_ts, tier_b2_profile=t_b2_weekly_target_ts, load_profile_ts=total_load, battery_deg=False
        )
        hpp.evaluate(*fixed_design)
        shortfall_b2 = hpp.prob.get_val('ems.Shortfall_B2')
        n_weeks = N_life // 168
        rel_b2 = 1.0 - (np.sum(np.sum(shortfall_b2[:n_weeks*168].reshape(-1, 168), axis=1) > 1e-3) / n_weeks)
        results_stage2b.append({"Tier_B2_MW": b2_mw, "Reliability_Deadline_B2": rel_b2 * 100})
        
        # EARLY STOPPING
        if rel_b2 < 0.40:
            break

    df1 = pd.DataFrame(results_stage1)
    df2 = pd.DataFrame(results_stage2)
    df2b = pd.DataFrame(results_stage2b)

    # Save to CSV
    df1.to_csv(file_a, index=False)
    df2.to_csv(file_b1, index=False)
    df2b.to_csv(file_b2, index=False)
    print("Simulation complete! Data saved to CSV.")

    # Generate Plot
    generate_thesis_plots(df1, df2, df2b)

if __name__ == "__main__":
    run_firm_yield_analysis()