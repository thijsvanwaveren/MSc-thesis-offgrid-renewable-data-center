# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:47:19 2026

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

def run_scenario(b_P, b_E_h, tier_a_val, tier_b1_val, tier_b2_val, allow_c2, max_it_mw=80.0):
    """Helper to run the assembly with specific load combinations."""
    design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, b_P, b_E_h, 10] 
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)
    
    with open(sim_pars_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    life_h = sim_pars['N_life'] * 365 * 24  
    
    # Create the specific load profiles based on the stage
    t_a_ts = np.full(life_h, float(tier_a_val))
    t_b1_ts = np.full(life_h, float(tier_b1_val * 24.0))  # Daily pool
    t_b2_ts = np.full(life_h, float(tier_b2_val * 168.0)) # Weekly pool
    
    total_load_for_ems = np.full(life_h, float(max_it_mw))
    
    # Toggle Tier C2 environmental variable
    if allow_c2:
        os.environ['REWARD_C2'] = '-0.5' 
    else:
        os.environ['REWARD_C2'] = '1.0'
    
    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_a_ts, tier_b_profile=t_b1_ts, tier_b2_profile=t_b2_ts, load_profile_ts=total_load_for_ems, battery_deg=False,
        run_mode='rolling'
    )
    hpp.evaluate(*design)
    return hpp.prob, sim_pars

def plot_stage_diagnostics(prob, sim_pars, title, tier_a_mw, max_it_mw, hours_to_plot=336):
    """Visualizes the stacked load profiles and battery state."""
    H_c = sim_pars['H_c_hours']
    
    # Extract Raw Data
    gen = (prob.get_val('ems.wind_t_ext') + prob.get_val('ems.solar_t_ext'))[:hours_to_plot]
    p_hpp = prob.get_val('ems.hpp_t')[:hours_to_plot]
    b_t = prob.get_val('ems.b_t')[:hours_to_plot]
    soc = prob.get_val('ems.b_E_SOC_t')[:hours_to_plot+1]
    
    # Extract individual loads to build the stack
    unserved_a = prob.get_val('ems.Unserved_A')[:hours_to_plot]
    served_c2 = prob.get_val('ems.Served_C2')[:hours_to_plot]
    
    # Calculate the layers
    served_a = tier_a_mw - unserved_a
    served_b_total = p_hpp - served_a - served_c2 # B1 and B2 combined
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    fig.suptitle(title, fontsize=16, fontweight='bold')
    hours_arr = np.arange(hours_to_plot)
    
    # --- AXIS 1: Stacked Power Dispatch ---
    ax1.axhline(0, color='black', linewidth=1.5, zorder=3)
    
    # Generation Curve
    ax1.plot(hours_arr, gen, label='Available Generation', color='black', linestyle='--', linewidth=2, alpha=0.6)
    
    # The Data Center Stack (Bottom to Top: A -> B -> C)
    ax1.fill_between(hours_arr, 0, served_a, label='Tier A (Baseload)', color='#2ca02c', alpha=0.8)
    ax1.fill_between(hours_arr, served_a, served_a + served_b_total, label='Tier B1/B2 (Flexible Data)', color='#1f77b4', alpha=0.7)
    ax1.fill_between(hours_arr, served_a + served_b_total, p_hpp, label='Tier C2 (Opportunistic Sponge)', color='#17becf', alpha=0.6)
    
    # Limit Lines
    ax1.axhline(max_it_mw, color='darkred', linestyle=':', linewidth=2, label='IT Capacity Limit')
    if tier_a_mw > 0:
        ax1.plot(hours_arr, np.full(hours_to_plot, tier_a_mw), color='white', linestyle=':', linewidth=1.5, alpha=0.8, label='Tier A Target')
    
    # Battery Flow Overlay
    b_discharge = np.maximum(b_t, 0)
    b_charge = np.minimum(b_t, 0)
    ax1.fill_between(hours_arr, 0, b_discharge, label='BESS Discharge (Assisting Grid)', color='#ff7f0e', alpha=0.7)
    ax1.fill_between(hours_arr, 0, b_charge, label='BESS Charge (Soaking Generation)', color='mediumpurple', alpha=0.7)
    
    ax1.set_ylabel("Power (MW)")
    ax1.set_ylim(min(np.min(b_charge)*1.1, -10), max_it_mw * 1.3) # Add headroom above IT limit
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- AXIS 2: Battery State of Charge ---
    ax2.plot(hours_arr, soc[:-1]/2, color='blue', linewidth=2.5, label='Battery SOC')
    ax2.set_ylabel("SoC (%)")
    ax2.axhline(10, color='darkred', linestyle=':', linewidth=2, label='Min SOC')

    ax2.set_xlabel("Hour of the Year")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # --- Visual Markers for Weeks and Days ---
    # for t in range(0, hours_to_plot + 1, 24): # Daily Seams
    #     ax1.axvline(x=t, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    #     ax2.axvline(x=t, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    # for t in range(0, hours_to_plot + 1, 168): # Weekly Planning Windows
    #     ax1.axvline(x=t, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    #     ax2.axvline(x=t, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    #     if t < hours_to_plot:
    #         ax2.text(t + 5, np.max(soc)*0.1, "New Forecast Window", color='red', fontsize=9, alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" MULTI-STAGE ROLLING HORIZON VISUALIZATION ".center(80, '='))
    print("="*80 + "\n")

    # =====================================================================
    # CHOOSE YOUR STAGE HERE (1, 2, 3, or 4)
    # =====================================================================
    STAGE = 4
    
    MAX_IT_MW = 80.0
    B_POWER = 25.0
    B_HOURS = 8.0
    
    if STAGE == 1:
        title = "Stage 1: Inflexible Baseload Only (Tier A)"
        tier_a, tier_b1, tier_b2, allow_c2 = 30.0, 0.0, 0.0, False
    elif STAGE == 2:
        title = "Stage 2: Baseload + Daily Flexible (Tier A + B1)"
        tier_a, tier_b1, tier_b2, allow_c2 = 20.0, 20.0, 0.0, False
    elif STAGE == 3:
        title = "Stage 3: Full SLA Stack (Tier A + B1 + B2)"
        tier_a, tier_b1, tier_b2, allow_c2 = 10.0, 15.0, 15.0, False
    elif STAGE == 4:
        title = "Stage 4: The Ultimate Microgrid (A + B1 + B2 + C2 Opportunistic)"
        tier_a, tier_b1, tier_b2, allow_c2 = 10.0, 15.0, 15.0, True
    else:
        raise ValueError("Invalid Stage selected. Choose 1, 2, 3, or 4.")

    print(f"Running {title}...")
    print(f"  Loads -> Tier A: {tier_a} MW | Tier B1: {tier_b1} MW | Tier B2: {tier_b2} MW | Tier C2: {allow_c2}")
    
    prob, sim_pars = run_scenario(
        b_P=B_POWER, b_E_h=B_HOURS, 
        tier_a_val=tier_a, tier_b1_val=tier_b1, tier_b2_val=tier_b2, 
        allow_c2=allow_c2, max_it_mw=MAX_IT_MW
    )
    
    print("Generating Visualization...")
    plot_stage_diagnostics(prob, sim_pars, title, tier_a_mw=tier_a, max_it_mw=MAX_IT_MW, hours_to_plot=8760)