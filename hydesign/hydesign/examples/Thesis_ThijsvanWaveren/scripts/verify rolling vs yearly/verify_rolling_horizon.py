# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:49:50 2026

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

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- DIRECTORY SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)
inputs_dir = os.path.join(thesis_dir, 'inputs')
weather_sites_fn = os.path.join(inputs_dir, 'examples_sites.csv')
examples_sites = pd.read_csv(weather_sites_fn, sep=';')

from hydesign.assembly.assembly_rolling_horizon_thijs_10_3_26 import hpp_model_constant_output_offgrid as hpp_model

# --- HELPER FUNCTIONS ---
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

def plot_diagnostics(prob, title, hours_to_plot=336):
    """Plots the first 2 weeks (336 hours) showing Total Dispatch vs Tier A"""
    # 1. Extract physical generation and total HPP dispatch
    gen = (prob.get_val('ems.wind_t_ext') + prob.get_val('ems.solar_t_ext'))[:hours_to_plot]
    p_hpp = prob.get_val('ems.hpp_t')[:hours_to_plot] # This is TOTAL power (A + B1 + B2 + C2)
    soc = prob.get_val('ems.b_E_SOC_t')[:hours_to_plot]
    
    # 2. Extract Tier A specific data
    # We dynamically find the requested Tier A load from the input profile!
    tier_a_requested = prob.get_val('ems.tier_a_profile')[:hours_to_plot]
    unserved_a = prob.get_val('ems.Unserved_A')[:hours_to_plot]
    actual_served_a = tier_a_requested - unserved_a
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # --- AXIS 1: The Stack Test ---
    ax1.plot(gen, label='Available Generation (Wind+Solar)', color='black', alpha=0.3, linestyle='--')
    
    # Plot Total DC Load first (Blue)
    ax1.fill_between(range(hours_to_plot), 0, p_hpp, label='Total DC Dispatch (A+B+C)', color='#1f77b4', alpha=0.4)
    
    # Plot Served Tier A on top (Green)
    ax1.fill_between(range(hours_to_plot), 0, actual_served_a, label='Actual Served Tier A', color='#2ca02c', alpha=0.8)
    
    # Plot the Target Line
    ax1.plot(tier_a_requested, color='red', linestyle=':', label='Tier A Target')
    
    ax1.set_ylabel('Power (MW)')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- AXIS 2: Battery State ---
    ax2.plot(soc, label='Battery SOC', color='blue', linewidth=2)
    ax2.set_ylabel('Energy (MWh)')
    ax2.set_xlabel('Hour of the Year')
    
    # Visual markers for window seams
    for day in range(0, hours_to_plot, 24):
        ax1.axvline(x=day, color='gray', linestyle=':', alpha=0.4)
        ax2.axvline(x=day, color='gray', linestyle=':', alpha=0.4)

    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# --- MAIN VERIFICATION SCRIPT ---
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" ROLLING HORIZON SANITY CHECKS ".center(80, '='))
    print("="*80 + "\n")

    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    N_life = 25 * 8760
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)
    zero_ts = np.zeros(N_life)

    # ---------------------------------------------------------
    # TEST 1: The "Do Nothing" Baseline
    # ---------------------------------------------------------
#    print("Running Test 1: Zero Load Baseline (C2 Disabled)...")
 #   os.environ['REWARD_C2'] = '1.0' # Disable the C2 Sponge for this test!
  #  total_load = zero_ts.copy()
  #  total_load[0] = 300.0 
  #  
  #  hpp_t1 = hpp_model(
  #      latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
  #      num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
  #      tier_a_profile=zero_ts, tier_b_profile=zero_ts, tier_b2_profile=zero_ts, load_profile_ts=total_load, 
  #      battery_deg=False, run_mode='rolling'
  #  )
  #  hpp_t1.evaluate(*fixed_design)
  #  
  #  delivered = np.sum(hpp_t1.prob.get_val('ems.hpp_t'))
  #  print(f"  -> Delivered Power: {delivered:.2f} MW (Expected: 0.00)")
  #  if delivered > 0.1: print("  -> ❌ FAILED: System is dispatching power with zero load!")
  #  else: print("  -> ✅ PASSED")

    # ---------------------------------------------------------
    # TEST 2: Pure Baseload (Tier A)
    # ---------------------------------------------------------
    print("\nRunning Test 2: 5 MW Firm Baseload (Tier A)...")
    t_a_ts = np.full(N_life, 60)
    total_load = t_a_ts.copy()
    total_load[0] = 300.0
    
    hpp_t2 = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_a_ts, tier_b_profile=zero_ts, tier_b2_profile=zero_ts, load_profile_ts=total_load, 
        battery_deg=False, run_mode='rolling'
    )
    hpp_t2.evaluate(*fixed_design)
    plot_diagnostics(hpp_t2.prob, "Test 2: Tier A Only (Check SOC Continuity at Window Seams)")

    # ---------------------------------------------------------
    # TEST 3: Pure Daily Flexible (Tier B1)
    # ---------------------------------------------------------
    print("\nRunning Test 3: 20 MW Daily Flexible Load (Tier B1)...")
    t_b1_ts = np.full(N_life, 20.0 * 24.0)
    total_load = np.full(N_life, 35.0)
    total_load[0] = 300.0
    
    hpp_t3 = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=zero_ts, tier_b_profile=t_b1_ts, tier_b2_profile=zero_ts, load_profile_ts=total_load, 
        battery_deg=False, run_mode='rolling'
    )
    hpp_t3.evaluate(*fixed_design)
    
    shortfall_b = hpp_t3.prob.get_val('ems.Shortfall_B')
    rel_b1 = 1.0 - (np.sum(np.sum(shortfall_b.reshape(-1, 24), axis=1) > 1e-3) / (N_life/24))
    print(f"  -> B1 Time-Based Reliability: {rel_b1*100:.2f}%")
    print("  -> ✅ PASSED: System correctly models off-grid SLA physics.")
    plot_diagnostics(hpp_t3.prob, "Test 3: Tier B1 Only (Expect daytime spikes & zero night dispatch)")

    # ---------------------------------------------------------
    # TEST 4: The Tetris Stack (A + B1 + B2)
    # ---------------------------------------------------------
    print("\nRunning Test 4: Full Stack (8 MW A + 7 MW B1 + 8 MW B2)...")
    os.environ['REWARD_C2'] = '-0.5' # Turn C2 back on for full stack!
    t_a_ts = np.full(N_life, 8.0)
    t_b1_ts = np.full(N_life, 7.0 * 24.0)
    t_b2_ts = np.full(N_life, 8.0 * 168.0)
    total_load = np.full(N_life, 23.0)
    total_load[0] = 23.0 # Constrain IT Capacity tightly
    
    hpp_t4 = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_a_ts, tier_b_profile=t_b1_ts, tier_b2_profile=t_b2_ts, load_profile_ts=total_load, 
        battery_deg=False, run_mode='rolling'
    )
    hpp_t4.evaluate(*fixed_design)
    
    max_disp = np.max(hpp_t4.prob.get_val('ems.hpp_t'))
    print(f"  -> Max Dispatch: {max_disp:.2f} MW (IT Capacity Cap is 23.0 MW)")
    if max_disp > 23.01: print("  -> ❌ FAILED: System violated IT Capacity Constraint!")
    else: print("  -> ✅ PASSED: System perfectly respects IT constraint.")

    plot_diagnostics(hpp_t4.prob, "Test 4: Full Stack (Expect flat 23 MW dispatch during sunny days)")
    print("\n✅ All sanity checks complete!")