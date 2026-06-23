# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:47:03 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 12:11:35 2026

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
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.assembly.hpp_assembly_offgrid_thijs_2_2_26 import hpp_model_constant_output_offgrid as hpp_model

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

def run_scenario(b_P, b_E_h, tier_a_val, tier_b_val, max_it_mw=40.0):
    """Helper to run the assembly with specific loads and battery sizes."""
    design = [35, 300, 5, 10, 7, 112, 39, 180, 1.25, b_P, b_E_h, 10] 
    
    site_name = 'Denmark_good_wind'
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
        tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_target_ts, load_profile_ts=total_load_for_ems, battery_deg=False
    )
    hpp.evaluate(*design)
    return hpp.prob

def plot_stacked_priority(prob, hours, tier_a_mw, tier_b_mw, title):
    # Extract data
    gen = (prob.get_val('ems.wind_t_ext') + prob.get_val('ems.solar_t_ext'))[:len(hours)]
    unserved_a = prob.get_val('ems.Unserved_A')[:len(hours)]
    served_c2 = prob.get_val('ems.Served_C2')[:len(hours)]
    hpp_t = prob.get_val('ems.hpp_t')[:len(hours)] # Total Served IT Load
    
    # Calculate individual served loads
    served_a = tier_a_mw - unserved_a
    served_b = hpp_t - served_a - served_c2
    
    # Calculate the Tier B Queue
    arrival = np.full(len(hours), tier_b_mw)
    queue = np.zeros(len(hours))
    for t in range(len(hours)):
        if t == 0:
            queue[t] = arrival[t] - served_b[t]
        else:
            queue[t] = queue[t-1] + arrival[t] - served_b[t]
            
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.set_title(title)
    
    # Removed PUE division for clarity
    effective_gen = gen 
    ax1.plot(hours, effective_gen, label='Generation', color='black', linestyle='--', alpha=0.8, zorder=4)
    ax1.axhline(tier_a_mw, color='red', linestyle=':', label='Target Tier A Load', zorder=5)
    
    # Stacked bars for loads
    ax1.bar(hours, served_a, label='Served Tier A (Highest priority)', color='green', alpha=0.7, zorder=3)
    ax1.bar(hours, served_b, bottom=served_a, label='Served Tier B (Lower priority)', color='teal', alpha=0.7, zorder=3)
    
    # Queue on secondary axis
    ax2.fill_between(hours, 0, queue, color='crimson', alpha=0.15, label='Tier B queue (MWh)', zorder=1)
    ax2.plot(hours, queue, color='darkred', linewidth=1.5, zorder=2)
    
    ax1.set_ylabel("Power (MW)")
    ax2.set_ylabel("Tier B Queue Energy (MWh)", color='darkred')
    ax1.set_xlabel("Hours")
    
    # Combine legends cleanly
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_stacked_priority_with_battery(prob, hours, tier_a_mw, tier_b_mw, title, b_E):
    # Extract data
    gen = (prob.get_val('ems.wind_t_ext') + prob.get_val('ems.solar_t_ext'))[:len(hours)]
    unserved_a = prob.get_val('ems.Unserved_A')[:len(hours)]
    served_c2 = prob.get_val('ems.Served_C2')[:len(hours)]
    hpp_t = prob.get_val('ems.hpp_t')[:len(hours)]
    b_t = prob.get_val('ems.b_t')[:len(hours)]
    soc = prob.get_val('ems.b_E_SOC_t')[:len(hours)]
    
    served_a = tier_a_mw - unserved_a
    served_b = hpp_t - served_a - served_c2
    
    # Calculate the Tier B Queue
    arrival = np.full(len(hours), tier_b_mw)
    queue = np.zeros(len(hours))
    for t in range(len(hours)):
        if t == 0:
            queue[t] = arrival[t] - served_b[t]
        else:
            queue[t] = queue[t-1] + arrival[t] - served_b[t]
            
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax2 = ax1.twinx()
    ax1.set_title(title)
    
    effective_gen = gen 
    ax1.plot(hours, effective_gen, label='Generation', color='black', linestyle='--', alpha=0.8, zorder=4)
    ax1.axhline(tier_a_mw, color='red', linestyle=':', label='Target Tier A Load', zorder=5)
    
    # Stacked bars for loads and battery
    ax1.bar(hours, served_a, label='Served Tier A (Highest priority)', color='green', alpha=0.7, zorder=3)
    ax1.bar(hours, served_b, bottom=served_a, label='Served Tier B (Lower priority)', color='teal', alpha=0.7, zorder=3)
    ax1.bar(hours, b_t, color='purple', alpha=0.4, label='Battery (Pos=Discharge, Neg=Charge)', zorder=4)
    
    # Queue on secondary axis
    ax2.fill_between(hours, 0, queue, color='crimson', alpha=0.15, label='Tier B queue (MWh)', zorder=1)
    ax2.plot(hours, queue, color='darkred', linewidth=1.5, zorder=2)
    
    ax1.set_ylabel("Power (MW)")
    ax2.set_ylabel("Tier B Queue Energy (MWh)", color='darkred')
    
    # SOC Plot
    soc_pct = (soc / b_E) * 100 if b_E else np.zeros(len(hours))
    ax3.plot(hours, soc_pct, color='blue', linewidth=2, label='SOC (%)')
    ax3.axhline(10, color='red', linestyle='--', alpha=0.7, label='Min SOC (10%)')
    ax3.set_ylabel("SOC (%)", color='blue')
    ax3.set_ylim(-5, 105)
    ax3.set_xlabel("Hours")
    ax3.legend(loc='upper right')

    # Combine legends cleanly for ax1
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_tier_a_scenario(prob, hours, title, tier_a_mw, has_battery, b_E=None):
    gen = (prob.get_val('ems.wind_t_ext') + prob.get_val('ems.solar_t_ext'))[:len(hours)]
    unserved_a = prob.get_val('ems.Unserved_A')[:len(hours)]
    served_a = tier_a_mw - unserved_a
    
    if has_battery:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(12, 4))
        
    ax1.set_title(title)
    ax1.plot(hours, gen, label='Total Generation', color='black', alpha=0.5)
    ax1.axhline(tier_a_mw, color='red', linestyle=':', label='Target Tier A Load')
    ax1.fill_between(hours, 0, served_a, label='Served Tier A', color='green', alpha=0.6)
    ax1.fill_between(hours, served_a, tier_a_mw, label='Unserved Load (Cancelled)', color='red', alpha=0.6)
    
    if has_battery:
        b_t = prob.get_val('ems.b_t')[:len(hours)]
        soc = prob.get_val('ems.b_E_SOC_t')[:len(hours)]
        
        # Convert SOC to percentage
        soc_pct = (soc / b_E) * 100 if b_E else np.zeros(len(hours))
        
        ax1.bar(hours, b_t, color='purple', alpha=0.4, label='Battery (Pos=Discharge, Neg=Charge)')
        
        # Plot SOC in the lower subplot
        ax2.plot(hours, soc_pct, color='blue', linewidth=2, label='SOC (%)')
        ax2.axhline(10, color='red', linestyle='--', alpha=0.7, label='Min SOC (10%)') # Default DoD is 90%
        ax2.set_ylabel("SOC (%)", color='blue')
        ax2.set_ylim(-5, 105)
        ax2.legend(loc='upper right')
        ax2.set_xlabel("Hours")
    else:
        ax1.set_xlabel("Hours")

    ax1.set_ylabel("Power (MW)")
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
def plot_power_balance_proof(prob, hours, title):
    # Get standard arrays
    gen = (prob.get_val('ems.wind_t_ext') + prob.get_val('ems.solar_t_ext'))[:len(hours)]
    b_t = prob.get_val('ems.b_t')[:len(hours)] # Positive = discharge, Negative = charge
    hpp_t = prob.get_val('ems.hpp_t')[:len(hours)] # Served IT Load
    curt = prob.get_val('ems.hpp_curt_t')[:len(hours)]

    # EMS CPLEX explicitly models PUE - Disabled (set to 1.0) for clarity
    pue = 1.0
    electrical_load = hpp_t * pue

    # Split battery into charge and discharge for stacking
    b_discharge = np.maximum(b_t, 0)
    b_charge = np.minimum(b_t, 0) # Already negative

    # Make demands negative for plotting below the X-axis
    demand_load = -electrical_load
    demand_curt = -curt

    # Calculate net imbalance (Should be an array of exact 0.0s)
    net_imbalance = gen + b_discharge + demand_load + b_charge + demand_curt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)

    # Stack Supply (Above 0)
    ax.bar(hours, gen, label='+ Generation', color='mediumseagreen', alpha=0.8)
    ax.bar(hours, b_discharge, bottom=gen, label='+ Battery Discharge', color='mediumorchid', alpha=0.8)

    # Stack Demand (Below 0)
    ax.bar(hours, demand_load, label='- Load', color='crimson', alpha=0.8)
    ax.bar(hours, b_charge, bottom=demand_load, label='- Battery Charge', color='royalblue', alpha=0.8)
    ax.bar(hours, demand_curt, bottom=demand_load + b_charge, label='- Curtailment', color='orange', alpha=0.8)

    # The Proof Line
    ax.plot(hours, net_imbalance, label='Power balance (0)', color='black', linestyle='--', linewidth=2)

    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel("Power (MW)")
    ax.set_xlabel("Hours")
    
    # Put legend outside to avoid blocking the graph
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()
    
    print(f"Max Absolute Imbalance in Plot 5: {np.max(np.abs(net_imbalance)):.6f} MW")    

def plot_ems_sanity_checks():
    PLOT_HOURS = 8760
    hours = np.arange(PLOT_HOURS)
    
    # 1. Pure Generation
    print("Running Plot 1: Pure Generation...")
    prob1 = run_scenario(b_P=0, b_E_h=0, tier_a_val=0, tier_b_val=0)
    wind = prob1.get_val('ems.wind_t_ext')[:PLOT_HOURS]
    solar = prob1.get_val('ems.solar_t_ext')[:PLOT_HOURS]
    
    plt.figure(figsize=(12, 4))
    plt.title("Plot 1: Generation (No Load, No Battery)")
    plt.plot(hours, wind, label='Wind MW', color='blue')
    plt.plot(hours, solar, label='Solar MW', color='orange')
    plt.plot(hours, wind+solar, label='Total Gen', color='black', linestyle='--')
    plt.ylabel("Power (MW)")
    plt.xlabel("Hours")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Tier A (15 MW) - No Battery
    print("Running Plot 2: Tier A (15 MW) WITHOUT Battery...")
    prob2 = run_scenario(b_P=0, b_E_h=0, tier_a_val=15.0, tier_b_val=0)
    plot_tier_a_scenario(prob2, hours, "Plot 2a: Firm 15 MW load (tier A) WITHOUT Battery", 15.0, False)

    # 2b. Tier A (25 MW) - No Battery
    print("Running Plot 2b: Tier A (25 MW) WITHOUT Battery...")
    prob2b = run_scenario(b_P=0, b_E_h=0, tier_a_val=25.0, tier_b_val=0)
    plot_tier_a_scenario(prob2b, hours, "Plot 2b: Firm 25 MW load (tier A) WITHOUT Battery", 25.0, False)

    # 3. Tier A (15 MW) - With Battery
    print("Running Plot 3: Tier A (15 MW) WITH Battery...")
    b_P = 20
    b_E_h = 8
    b_E = b_P * b_E_h # 80 MWh
    prob3 = run_scenario(b_P=b_P, b_E_h=b_E_h, tier_a_val=15.0, tier_b_val=0) 
    plot_tier_a_scenario(prob3, hours, f"Plot 3a: Firm 15 MW load (tier A) WITH Battery ({b_P}MW/{b_E}MWh)", 15.0, True, b_E)

    # 3b. Tier A (25 MW) - With Battery
    print("Running Plot 3b: Tier A (25 MW) WITH Battery...")
    prob3b = run_scenario(b_P=b_P, b_E_h=b_E_h, tier_a_val=25.0, tier_b_val=0) 
    plot_tier_a_scenario(prob3b, hours, "Plot 3b: Firm 25 MW load (tier A) WITH Battery ({b_P}MW/{b_E}MWh)", 25.0, True, b_E)

    # 4. Tier B Queue Logic (No Battery, Heavy Load)
    print("Running Plot 4: Tier B Schedulable Queue logic...")
    TIER_B_MW = 40 
    prob4 = run_scenario(b_P=0, b_E_h=0, tier_a_val=0, tier_b_val=TIER_B_MW)
    
    gen = (prob4.get_val('ems.wind_t_ext') + prob4.get_val('ems.solar_t_ext'))[:PLOT_HOURS]
    hpp_out = prob4.get_val('ems.hpp_t')[:PLOT_HOURS]
    served_c2 = prob4.get_val('ems.Served_C2')[:PLOT_HOURS]
    shortfall = prob4.get_val('ems.Shortfall_B')[:PLOT_HOURS]
    served_b = hpp_out - served_c2 
    
    arrival = np.full(PLOT_HOURS, TIER_B_MW)
    queue = np.zeros(PLOT_HOURS)
    for t in range(PLOT_HOURS):
        if t == 0:
            queue[t] = arrival[t] - served_b[t]
        else:
            queue[t] = queue[t-1] + arrival[t] - served_b[t]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    
    ax1.set_title(f"Plot 4: Tier B Schedulable Load ({TIER_B_MW} MW) with queue")
    ax1.plot(hours, gen, label='Total Generation', color='black', alpha=0.3, zorder=1)
    ax1.plot(hours, arrival, label='Tier B Hourly Inflow', color='orange', linestyle='--', zorder=2)
    ax1.bar(hours, served_b, color='teal', alpha=0.7, label='Served Tier B (Scheduled)', zorder=3)
    
    # Making the queue highly visible
    ax2.fill_between(hours, 0, queue, color='crimson', alpha=0.3, label='Backlog Queue (MWh)', zorder=4)
    ax2.plot(hours, queue, color='darkred', linewidth=1.5, zorder=5) # Hard outline for the queue
    
    missed_idx = np.where(shortfall > 1e-3)[0]
    if len(missed_idx) > 0:
        ax2.scatter(hours[missed_idx], queue[missed_idx], color='red', edgecolor='black', s=60, zorder=6, label='Penalty: Missed Deadline')

    ax1.set_ylabel("Power (MW)")
    ax2.set_ylabel("Queue Energy (MWh)", color='darkred')
    ax1.set_xlabel("Hours")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    # 5. The Power Balance Proof
    print("Running Plot 5: Power Balance Proof...")
    plot_power_balance_proof(prob3, hours, "Plot 5: Power Balance (Supply vs. Demand)")
    
    # 6. Priority Stacking Proof (Tier A + Tier B, No Battery)
    print("Running Plot 6: Priority Stacking Proof...")
    TIER_A_MIX = 10.0
    TIER_B_MIX = 10.0
    prob6 = run_scenario(b_P=0, b_E_h=0, tier_a_val=TIER_A_MIX, tier_b_val=TIER_B_MIX)
    plot_stacked_priority(prob6, hours, TIER_A_MIX, TIER_B_MIX, 
                          f"Plot 6: Load Priority (Tier A: {TIER_A_MIX}MW, Tier B: {TIER_B_MIX}MW, NO Battery)")

    # 7. Priority Stacking Proof WITH Battery
    print("Running Plot 7: Priority Stacking Proof WITH Battery...")
    b_P = 20
    b_E_h = 4
    b_E = b_P * b_E_h
    prob7 = run_scenario(b_P=b_P, b_E_h=b_E_h, tier_a_val=TIER_A_MIX, tier_b_val=TIER_B_MIX)
    plot_stacked_priority_with_battery(prob7, hours, TIER_A_MIX, TIER_B_MIX, 
                          f"Plot 7: Load Priority (Tier A: {TIER_A_MIX}MW, Tier B: {TIER_B_MIX}MW, WITH Battery)", b_E)

if __name__ == "__main__":
    plot_ems_sanity_checks()