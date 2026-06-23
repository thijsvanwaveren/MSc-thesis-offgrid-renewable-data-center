# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:10:18 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:47:03 2026

@author: thijs
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
thesis_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

# --- IMPORT ROLLING HORIZON ASSEMBLY ---
from hydesign.assembly.assembly_rolling_horizon_thijs_10_3_26 import hpp_model_constant_output_offgrid as hpp_model

def configure_parameters(thesis_dir):
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    sim_pars['G_MW'] = 0
    sim_pars['battery_charge_efficiency'] = float(np.sqrt(0.86))
    
    # We explicitly set the rolling horizon parameters here
    sim_pars['H_p_hours'] = 168 # 7-day prediction
    sim_pars['H_c_hours'] = 24  # 1-day control execution
    
    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_offgrid_sweep_temp.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    return temp_fn

def run_scenario(b_P, b_E_h, tier_a_val, tier_b_val, max_it_mw=40.0):
    """Helper to run the assembly with specific loads and battery sizes."""
    design = [35, 300, 5, 10, 7, 112, 39, 180, 1.25, b_P, b_E_h, 10] 
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)
    
    with open(sim_pars_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    life_h = sim_pars['N_life'] * 365 * 24  
    
    t_a_ts = np.full(life_h, float(tier_a_val))
    t_b_daily_target_ts = np.full(life_h, float(tier_b_val * 24.0))
    total_load_for_ems = t_a_ts + np.full(life_h, float(tier_b_val))
    total_load_for_ems[0] = float(max_it_mw) 
    
    os.environ['REWARD_C2'] = '1.0' # Disable C2
    
    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_target_ts, tier_b2_profile=np.zeros(life_h), load_profile_ts=total_load_for_ems, battery_deg=False,
        run_mode='rolling' # Enforce rolling mode
    )
    hpp.evaluate(*design)
    return hpp.prob, sim_pars

def plot_rolling_horizon_operation(prob, sim_pars, hours_to_plot=336, tier_a_mw=15.0):
    """Visualizes the rolling horizon steps."""
    
    H_c = sim_pars['H_c_hours']
    
    # Extract data
    gen = (prob.get_val('ems.wind_t_ext') + prob.get_val('ems.solar_t_ext'))[:hours_to_plot]
    unserved_a = prob.get_val('ems.Unserved_A')[:hours_to_plot]
    b_t = prob.get_val('ems.b_t')[:hours_to_plot]
    soc = prob.get_val('ems.b_E_SOC_t')[:hours_to_plot+1]
    
    served_a = tier_a_mw - unserved_a
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    title = f"Rolling Horizon Operation (Control Window $H_c$ = {H_c}h)"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    hours_arr = np.arange(hours_to_plot)
    
    # --- AXIS 1: Power ---
    ax1.plot(hours_arr, gen, label='Available Generation', color='black', alpha=0.3)
    ax1.axhline(tier_a_mw, color='red', linestyle='--', alpha=0.5, label='Tier A Target')
    ax1.fill_between(hours_arr, 0, served_a, color='green', alpha=0.5, label='Served Tier A')
    
    # Split battery for coloring
    b_discharge = np.maximum(b_t, 0)
    b_charge = np.minimum(b_t, 0)
    ax1.bar(hours_arr, b_discharge, color='mediumorchid', alpha=0.7, label='Battery Discharge')
    ax1.bar(hours_arr, b_charge, color='royalblue', alpha=0.7, label='Battery Charge')
    
    ax1.set_ylabel("Power (MW)")
    ax1.legend(loc='upper right')
    
    # --- AXIS 2: State of Charge ---
    # Convert SOC to percentage for clearer viewing. Assume capacity from first step.
    # To get b_E accurately, we'd need to extract it, but let's approximate max SOC seen as capacity
    b_E_approx = np.max(soc) 
    soc_pct = (soc[:-1] / b_E_approx) * 100 if b_E_approx > 0 else np.zeros(hours_to_plot)
    
    ax2.plot(hours_arr, soc_pct, color='darkgreen', linewidth=2, label='State of Charge (%)')
    ax2.set_ylabel("SOC (%)")
    ax2.set_ylim(-5, 105)
    ax2.set_xlabel("Hours")
    ax2.legend(loc='upper right')
    
    # --- Draw the Rolling Horizon Seams ---
    for t in range(0, hours_to_plot, H_c):
        ax1.axvline(x=t, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
        ax2.axvline(x=t, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
        
        # Add text marker for the control window start
        if t < hours_to_plot - H_c:
             ax2.text(t + (H_c/2), 50, f"Window\nExec", ha='center', va='center', color='red', alpha=0.5, fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" ROLLING HORIZON VISUALIZATION ".center(80, '='))
    print("="*80 + "\n")

    # Run a stressed scenario so the battery has to work hard across windows
    b_P = 25
    b_E_h = 8
    TIER_A_MW = 30.0 
    
    print(f"Running scenario with Tier A = {TIER_A_MW} MW and Battery = {b_P} MW / {b_P*b_E_h} MWh")
    prob, sim_pars = run_scenario(b_P=b_P, b_E_h=b_E_h, tier_a_val=TIER_A_MW, tier_b_val=0)
    
    print("Generating Rolling Horizon Plot...")
    plot_rolling_horizon_operation(prob, sim_pars, hours_to_plot=336, tier_a_mw=TIER_A_MW)