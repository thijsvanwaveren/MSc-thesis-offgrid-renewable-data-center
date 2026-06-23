# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 13:20:01 2026

@author: thijs
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

# FORCE C2 OFF (We strictly want to examine Tier A & B physics)
os.environ['REWARD_C2'] = '1.0'

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

def plot_utilization_tradeoffs(df):
    """Generates a 3-panel plot showing the CapEx vs OpEx vs Reliability tradeoff."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Reliability vs Utilization
    ax1.plot(df['Utilization_%'], df['Reliability_Time_A'], marker='o', color='blue', linewidth=2, label='Tier A Uptime')
    ax1.plot(df['Utilization_%'], df['Reliability_Deadline_B'], marker='s', color='orange', linewidth=2, label='Tier B Deadline Success')
    ax1.axhline(99.0, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel("Reliability (%)", fontsize=12)
    ax1.set_title("Reliability vs Server Utilization", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis() # Invert so 100% (No Headroom) is on the left, 40% (Massive Headroom) is on the right
    
    # 2. Curtailment vs Utilization
    ax2.plot(df['Utilization_%'], df['Curtailment_GWh'], marker='^', color='gray', linewidth=2)
    ax2.set_ylabel("Lifetime Curtailment (GWh)", fontsize=12)
    ax2.set_title("Curtailment vs Server Utilization", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. LCOEd vs Utilization
    ax3.plot(df['Utilization_%'], df['LCOE'], marker='D', color='purple', linewidth=2)
    ax3.set_ylabel("LCOE Delivered (€/MWh)", fontsize=12)
    ax3.set_xlabel("Average Server Utilization (%) -> (Decreasing means MORE idle servers)", fontsize=12)
    ax3.set_title("Cost of Energy vs Server Utilization", fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_utilization_sweep():
    # Setup Paths
    site_name = 'Denmark_good_wind'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)
    
    N_life = 25 * 8760
    
    # --- FIXED LOADS ---
    T_A_MW = 8.0
    T_B_MW = 8.0
    TOTAL_AVG_LOAD = T_A_MW + T_B_MW
    
    # Hardware Design: Ensure you use your exact optimal hardware array here!
    # [clearance, sp, p_rated, Nwt, wind_MW_per_km2, solar_MW, surface_tilt, surface_azimuth, DC_AC_ratio, b_P, b_E_h, cost_batt_fluct]
    fixed_design = [35, 300, 5, 10, 7, 112, 39, 180, 1.25, 40, 14, 10] 
    
    # --- THE SWEEP ---
    # 1.0 = 100% Util (16 MW Max), 0.4 = 40% Util (40 MW Max)
    util_sweep = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4] 
    results = []
    
    print(f"--- Starting Server Utilization Sweep ---")
    print(f"Base Load: {T_A_MW} MW | Flex Load: {T_B_MW} MW | Total Avg: {TOTAL_AVG_LOAD} MW\n")
    print(f"{'Util (%)':<10} | {'Max IT (MW)':<12} | {'Rel(A) %':<10} | {'Rel(B) %':<10} | {'Curt (GWh)':<12} | {'LCOEd (€)':<10}")
    print("-" * 75)
    
    for util in util_sweep:
        max_it_MW = TOTAL_AVG_LOAD / util
        
        # 1. Prepare Arrays
        t_a_ts = np.full(N_life, T_A_MW)
        t_b_hourly_ts = np.full(N_life, T_B_MW)
        t_b_daily_ts = np.full(N_life, T_B_MW * 24.0) # EMS expects daily targets
        
        load_profile = t_a_ts + t_b_hourly_ts
        load_profile[0] = max_it_MW # Pass the IT Limit to the EMS
        
        # 2. Initialize and Run
        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_ts, load_profile_ts=load_profile, battery_deg=False
        )
        out = hpp.evaluate(*fixed_design)
        prob = hpp.prob
        
        # 3. Extract Metrics
        unserved_a_ts = prob.get_val('ems.Unserved_A')
        rel_time_a = 1.0 - (np.sum(unserved_a_ts > 1e-3) / N_life)
        
        shortfall_b_ts = prob.get_val('ems.Shortfall_B')
        rel_deadline_b = 1.0 - (np.sum(np.sum(shortfall_b_ts.reshape(-1, 24), axis=1) > 1e-3) / (N_life/24))
        
        curt_GWh = np.sum(prob.get_val('ems.hpp_curt_t')) / 1000.0
        
        try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_d = out[3]
        
        print(f"{util*100:<10.1f} | {max_it_MW:<12.1f} | {rel_time_a*100:<10.2f} | {rel_deadline_b*100:<10.2f} | {curt_GWh:<12.1f} | {lcoe_d:<10.2f}")
        
        results.append({
            "Utilization_%": util * 100,
            "Max_IT_MW": max_it_MW,
            "Reliability_Time_A": rel_time_a * 100,
            "Reliability_Deadline_B": rel_deadline_b * 100,
            "Curtailment_GWh": curt_GWh,
            "LCOE": lcoe_d
        })
        
    df_res = pd.DataFrame(results)
    plot_utilization_tradeoffs(df_res)

if __name__ == "__main__":
    run_utilization_sweep()