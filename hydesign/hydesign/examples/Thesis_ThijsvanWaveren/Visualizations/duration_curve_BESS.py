# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:50:44 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Plot: Available RE vs. System Power (with BESS) Duration Curve
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

from hydesign.assembly.hpp_assembly_offgrid_thijs_2_2_26 import hpp_model_constant_output_offgrid as hpp_model

# ==============================================================================
# BESS CHRONOLOGICAL SIMULATION
# ==============================================================================

def apply_chronological_bess(re_series, target_baseload=16.0, p_bess=25.0, e_bess=200.0, eta_rt=0.86):
    """
    Simulates a simple chronological BESS dispatch attempting to maintain a target baseload.
    Returns the chronologically available system power after battery shifting.
    """
    sys_power = np.zeros(len(re_series))
    soc = e_bess * 0.5  # Assume battery starts half full
    eta_ch = np.sqrt(eta_rt)  # One-way charging efficiency
    eta_dis = np.sqrt(eta_rt) # One-way discharging efficiency

    for i, re in enumerate(re_series):
        if re >= target_baseload:
            # SURPLUS: We have more than the baseload. Try to charge the battery.
            surplus = re - target_baseload
            # Charge is limited by surplus, BESS power rating, and available empty space in the battery
            charge = min(surplus, p_bess, (e_bess - soc) / eta_ch)
            soc += charge * eta_ch
            sys_power[i] = re - charge  # Energy leaves the available pool to go into the battery
        else:
            # DEFICIT: We have less than the baseload. Try to discharge the battery.
            deficit = target_baseload - re
            # Discharge is limited by deficit, BESS power rating, and available energy in the battery
            discharge = min(deficit, p_bess, soc * eta_dis)
            soc -= discharge / eta_dis
            sys_power[i] = re + discharge # Energy is added to the available pool from the battery

    return sys_power

# ==============================================================================
# PLOTTING FUNCTION
# ==============================================================================

def plot_re_vs_bess_duration_curve(df):
    """
    Plots a duration curve showing original Wind+Solar vs. System Power after BESS.
    """
    # 1. Calculate combined Renewable Energy
    df['Available_RE'] = df['Wind'] + df['Solar']

    # 2. Apply Chronological BESS
    # Using parameters established for Denmark: 25 MW / 200 MWh (8 hours) targeting a 16 MW baseload
    df['System_Power'] = apply_chronological_bess(df['Available_RE'].values, 
                                                  target_baseload=8.0, 
                                                  p_bess=25.0, 
                                                  e_bess=200.0)

    # 3. Sort independently to create duration curves
    re_sorted = df['Available_RE'].sort_values(ascending=False).reset_index(drop=True)
    sys_sorted = df['System_Power'].sort_values(ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6))

    # --- PLOT THE CURVES ---
    plt.plot(re_sorted.index, re_sorted, color='#2ca02c', linewidth=2, alpha=0.6, label='Wind and Solar generation')
    plt.plot(sys_sorted.index, sys_sorted, color='#1f77b4', linewidth=2.5, label='System Power (Including BESS)')
    
    # --- VISUALIZE THE SHIFT ---
    # Highlight the chopped peak (Charging)
    plt.fill_between(re_sorted.index, sys_sorted, re_sorted, where=(re_sorted > sys_sorted), 
                     color='#ff7f0e', alpha=0.3, label='Energy Charged')
    
    # Highlight the lifted tail (Discharging / Firming the Baseload)
    plt.fill_between(sys_sorted.index, re_sorted, sys_sorted, where=(sys_sorted > re_sorted), 
                     color='#1f77b4', alpha=0.3, label='Energy Discharged')

    # Add a reference line for the 16 MW Baseload
    plt.axhline(y=8.0, color='red', linestyle='--', linewidth=1.5, label='Target Firm Baseload (8 MW)')

    # Formatting
    plt.title("Effect of BESS on Renewable Generation Duration Curve", fontsize=16, fontweight='bold')
    plt.xlabel("Hours of the Year", fontsize=12)
    plt.ylabel("Power (MW)", fontsize=12)
    plt.xlim(0, 8760)
    plt.ylim(0, re_sorted.max() * 1.05)
    
    # Grid and Legend
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', framealpha=0.95, shadow=True, fontsize=10)
    
    plt.tight_layout()

    # Save the plot
    plot_fn = os.path.join(current_dir, 'BESS_Effect_Duration_Curve.png')
    plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
    print(f"✅ Saved BESS Effect duration curve to: {plot_fn}")
    plt.show()

def configure_parameters(thesis_dir):
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    sim_pars['G_MW'] = 0
    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_offgrid_sweep_temp.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    return temp_fn

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def run_generation_extraction():

    # HPP Hardware Setup
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    N_life = 25 * 8760

    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    print(f"--- Running HPP Model to Extract Weather Data ---")

    # Build Empty Load Profiles (Since we only care about generation)
    t_zero_ts = np.zeros(N_life)

    # Initialize Model
    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_zero_ts, tier_b_profile=t_zero_ts, load_profile_ts=t_zero_ts,
        battery_deg=False 
    )

    print("Evaluating design to generate wind/solar profiles...")
    hpp.evaluate(*fixed_design)
    prob = hpp.prob

    # =========================================================================
    # DATA EXTRACTION FOR PLOTTING (First Year Only)
    # =========================================================================
    print("\nExtracting hourly generation data...")
    HOURS_TO_PLOT = 8760

    wind = prob.get_val('ems.wind_t_ext')[:HOURS_TO_PLOT]
    solar = prob.get_val('ems.solar_t_ext')[:HOURS_TO_PLOT]

    # Build DataFrame
    df_plot = pd.DataFrame({
        'Wind': wind,
        'Solar': solar
    })
    
    avg_wind = df_plot['Wind'].mean()
    avg_solar = df_plot['Solar'].mean()
    avg_re = (df_plot['Wind'] + df_plot['Solar']).mean()

    print(f"Average Wind Power:  {avg_wind:.2f} MW")
    print(f"Average Solar Power: {avg_solar:.2f} MW")
    print(f"Average RE Power:    {avg_re:.2f} MW")

    # Plot it
    plot_re_vs_bess_duration_curve(df_plot)

if __name__ == "__main__":
    run_generation_extraction()