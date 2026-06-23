# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:54:34 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Tier B2 (168-hour Queue) Diagnostic & Validation Script
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

def run_b2_validation():
    # Standard 35MW Wind, 300MW Solar, 5MW/20MWh Battery
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    N_life = 25 * 8760
    
    # ---------------------------------------------------------
    # SCENARIO CONFIGURATION: Isolate B2!
    # ---------------------------------------------------------
    MAX_IT_CAPACITY_MW = 16
    TIER_A_MW = 0.0     # Firm Baseload
    TIER_B1_MW = 0.0    # TURNED OFF
    TIER_B2_MW = 16   # A heavy weekly batch load to force queueing
    
    os.environ['REWARD_C2'] = '1.0'  # Kills C2 opportunistic loads
    
    print(f"\n--- Running B2 Diagnostic Validation ---")
    print(f"Tier A (Firm):     {TIER_A_MW} MW")
    print(f"Tier B2 (168h):    {TIER_B2_MW} MW average")
    print(f"Max IT Limit:      {MAX_IT_CAPACITY_MW} MW\n")

    # Build Profiles
    t_a_ts = np.full(N_life, TIER_A_MW)
    t_b1_ts = np.zeros(N_life)  # Zero B1 Target
    t_b2_weekly_target_ts = np.full(N_life, TIER_B2_MW * 168.0) 
    
    total_load_for_ems = t_a_ts + np.full(N_life, TIER_B1_MW) + np.full(N_life, TIER_B2_MW)
    total_load_for_ems[0] = MAX_IT_CAPACITY_MW

    # Site configuration
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    # Initialize and Evaluate Model
    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_a_ts, tier_b_profile=t_b1_ts, tier_b2_profile=t_b2_weekly_target_ts, 
        load_profile_ts=total_load_for_ems, battery_deg=False
    )
    
    out = hpp.evaluate(*fixed_design)
    prob = hpp.prob

    # ---------------------------------------------------------
    # EXTRACT AND DERIVE TIME-SERIES RESULTS
    # ---------------------------------------------------------
    # Calculate Reliabilities
    unserved_a_ts = prob.get_val('ems.Unserved_A')
    rel_time_a = 1.0 - (np.sum(unserved_a_ts > 1e-3) / N_life)
    
    shortfall_b2_ts = prob.get_val('ems.Shortfall_B2')
    n_weeks = N_life // 168
    b2_weeks_matrix = shortfall_b2_ts[:n_weeks * 168].reshape(-1, 168)
    rel_deadline_b2 = 1.0 - (np.sum(np.sum(b2_weeks_matrix, axis=1) > 1e-3) / n_weeks)
    
    print(f"Results:")
    print(f"  Tier A Reliability:  {rel_time_a * 100:.2f}%")
    print(f"  Tier B2 Reliability: {rel_deadline_b2 * 100:.2f}%\n")

    # Extract Physics for the first 3 weeks (504 hours)
    T_PLOT = 168 * 5
    
    hpp_out = prob.get_val('ems_long_term_operation.hpp_t')[:T_PLOT]
    gen_wind = prob.get_val('ems_long_term_operation.wind_t_ext')[:T_PLOT]
    gen_solar = prob.get_val('ems_long_term_operation.solar_t_ext')[:T_PLOT]
    b_soc = prob.get_val('ems_long_term_operation.b_E_SOC_t')[:T_PLOT]
    b_power = prob.get_val('ems_long_term_operation.b_t')[:T_PLOT] # Discharge > 0, Charge < 0
    curt = prob.get_val('ems_long_term_operation.hpp_curt_t')[:T_PLOT]
    
    unserved_a_plot = unserved_a_ts[:T_PLOT]
    shortfall_b2_plot = shortfall_b2_ts[:T_PLOT]
    
    # Deriving Tier B2 Served Power
    # Because B1 and C2 are 0, any power drawn by the data center above Tier A is Tier B2!
    served_a_plot = TIER_A_MW - unserved_a_plot
    served_b2_plot = hpp_out - served_a_plot

    # ---------------------------------------------------------
    # PLOTTING THE VALIDATION
    # ---------------------------------------------------------
    hours = np.arange(T_PLOT)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Tier B2 Diagnostic Validation (First 3 Weeks)', fontsize=16, fontweight='bold')

    # Panel 1: Power Balance & Workload Stacking
    ax1.plot(hours, gen_wind + gen_solar, color='lightgray', label='Available Renewable Gen (Wind+Solar)')
    ax1.fill_between(hours, 0, served_a_plot, color='#1f77b4', label='Tier A Served (Firm)')
    ax1.fill_between(hours, served_a_plot, served_a_plot + served_b2_plot, color='#8c564b', alpha=0.8, label='Tier B2 Served (168h Batch)')
    
    ax1.axhline(TIER_A_MW + TIER_B2_MW, color='black', linestyle='--', label='Average Target Load (A + B2)')
    ax1.axhline(MAX_IT_CAPACITY_MW, color='red', linestyle='--', linewidth=2, label='Max IT Hardware Limit')
    ax1.set_ylabel('Power (MW)')
    ax1.set_title('Workload Dispatch & Generation')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Battery Operations
    ax2.plot(hours, b_soc, color='green', linewidth=2, label='Battery SoC (MWh)')
    ax2.bar(hours, b_power, color='purple', alpha=0.5, label='Battery Dispatch (MW) [+Discharge, -Charge]')
    ax2.set_ylabel('Energy / Power')
    ax2.set_title('Battery State of Charge and Cycling')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: SLA Violations (Sanity Check)
    ax3.plot(hours, shortfall_b2_plot, color='orange', linewidth=2, label='Tier B2 Queue Past 168h Deadline (MWh)')
    ax3.plot(hours, unserved_a_plot, color='red', linewidth=2, label='Tier A Dropped Load (MW)')
    
    # Add vertical lines to show weekly boundaries
    for w in range(1, 4):
        ax3.axvline(w * 168, color='black', linestyle=':', alpha=0.5)
        ax1.axvline(w * 168, color='black', linestyle=':', alpha=0.5)
        
    ax3.set_xlabel('Hours')
    ax3.set_ylabel('Violations')
    ax3.set_title('SLA Compliance & Weekly Boundaries (Dotted Lines = 168h)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_b2_validation()
    