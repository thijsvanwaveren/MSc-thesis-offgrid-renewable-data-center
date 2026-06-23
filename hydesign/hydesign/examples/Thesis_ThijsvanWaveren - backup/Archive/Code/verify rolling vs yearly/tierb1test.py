# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:42:39 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
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
thesis_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
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
    os.environ['REWARD_C2'] = '1.0' # Disable C2)
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    return temp_fn

def plot_queue_diagnostics(prob, title, target_mw=20.0, hours_to_plot=780):
    """3-Panel Plot: Separating Tier B1, Tier C2, and Battery Flows"""
    
    # 1. Extract Data
    gen = (prob.get_val('ems.wind_t_ext') + prob.get_val('ems.solar_t_ext'))[:hours_to_plot]
    p_hpp = prob.get_val('ems.hpp_t')[:hours_to_plot] 
    soc = prob.get_val('ems.b_E_SOC_t')[:hours_to_plot]
    shortfall = prob.get_val('ems.Shortfall_B')[:hours_to_plot]
    
    # Extract Tier C2 to separate the loads!
    p_c2 = prob.get_val('ems.Served_C2')[:hours_to_plot]
    p_b1 = p_hpp - p_c2 # True Tier B1 Dispatch
    
    # Extract Battery Power
    b_t = prob.get_val('ems.b_t')[:hours_to_plot]
    b_discharge = np.maximum(b_t, 0)
    b_charge = np.minimum(b_t, 0)
    
    # 2. Reconstruct the Queue Mathematically (using TRUE B1 Dispatch)
    queue = np.zeros(hours_to_plot)
    for i in range(1, hours_to_plot):
        queue[i] = max(0, queue[i-1] + target_mw - p_b1[i])
        
    # 3. Build the Figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # --- AXIS 1: Power Dispatch & BESS Flow ---
    ax1.axhline(0, color='black', linewidth=1.5, zorder=3)
    ax1.plot(gen, label='Available Generation (Wind+Solar)', color='black', alpha=0.6, linestyle='--', linewidth=2)
    
    # Plot TRUE Tier B1
    ax1.fill_between(range(hours_to_plot), 0, p_b1, label='Tier B1 Dispatch (SLA Load)', color='#1f77b4', alpha=0.6)
    
    # Stack Tier C2 on top!
    #ax1.fill_between(range(hours_to_plot), p_b1, p_hpp, label='Tier C2 (Opportunistic Sponge)', color='#17becf', alpha=0.8)
    
    ax1.fill_between(range(hours_to_plot), 0, b_discharge, label='BESS Discharge', color='#ff7f0e', alpha=0.8)
    ax1.fill_between(range(hours_to_plot), 0, b_charge, label='BESS Charge', color='#2ca02c', alpha=0.8)
    
    ax1.axhline(y=target_mw, color='green', linestyle=':', label='Tier B1 Arrival Target', linewidth=2)
    ax1.axhline(total_load[0], color='darkred', linestyle=':', linewidth=2, label='IT Capacity Limit')

    
    ax1.set_ylabel('Power (MW)')
    ax1.set_ylim(min(np.min(b_charge) * 1.2, -10), max(np.max(gen) * 1.1, np.max(p_hpp) * 1.1))
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- AXIS 2: Battery State ---
    ax2.plot(soc, label='Battery SOC', color='blue', linewidth=2)
    ax2.set_ylabel('Energy (MWh)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # --- AXIS 3: The Queue & Penalties ---
    ax3.fill_between(range(hours_to_plot), 0, queue, label='Queue Backlog (MWh)', color='purple', alpha=0.3)
    ax3.bar(range(hours_to_plot), shortfall, label='SLA Violations (Jobs past 24h deadline)', color='red', alpha=0.8)
    ax3.set_ylabel('Backlog (MWh)')
    ax3.set_xlabel('Hour of the Year')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    for day in range(0, hours_to_plot, 24):
        ax1.axvline(x=day, color='gray', linestyle=':', alpha=0.4)
        ax2.axvline(x=day, color='gray', linestyle=':', alpha=0.4)
        ax3.axvline(x=day, color='gray', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    N_life = 25 * 8760
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)
    zero_ts = np.zeros(N_life)

    # ---------------------------------------------------------
    # TEST 3: Flexible Load Queue Mechanics
    # ---------------------------------------------------------
    TARGET_B1 = 30.0
    print(f"\nRunning Test 3: {TARGET_B1} MW Daily Flexible Load (Tier B1)...")
    
    t_b1_ts = np.full(N_life, TARGET_B1 * 24.0)
    total_load = np.full(N_life, TARGET_B1)
    total_load[0] = 80.0 # IT capacity
    
    hpp_t3 = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=zero_ts, tier_b_profile=t_b1_ts, tier_b2_profile=zero_ts, load_profile_ts=total_load, 
        battery_deg=False, run_mode='rolling'
    )
    hpp_t3.evaluate(*fixed_design)
    
    plot_queue_diagnostics(hpp_t3.prob, f"Tier B1 Queue Dynamics Rolling Horizon ({TARGET_B1} MW)", target_mw=TARGET_B1, hours_to_plot=780) 