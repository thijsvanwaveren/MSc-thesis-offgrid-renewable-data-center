# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 13:59:42 2026

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

# ==============================================================================
# NEW PLOTTING FUNCTION (SMOOTHED)
# ==============================================================================

def plot_storytelling_duration_curve(df, tier_a_target, max_it_capacity, title):
    """
    Plots a multi-layered duration curve that tells the story of the EMS dispatch.
    Sorted by Available RE to show how demand is met as weather resources decline.
    """
    # 1. Calculate combined metrics
    df['Available_RE'] = df['Wind'] + df['Solar']
    df['Total_Served'] = df['Load_Fixed_Served'] + df['Load_Flexible_Served'] + df['Served_C2']

    # 2. Sort by 'Available_RE' descending. This creates a smooth supply envelope 
    # from the windiest/sunniest hour on the left, to the darkest/stillest hour on the right.
    df_sorted = df.sort_values(by='Available_RE', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(14, 8))

    # --- THE DEMAND STACK ---
    # We stack the loads from most critical (bottom) to most opportunistic (top)
    plt.stackplot(df_sorted.index, 
                  df_sorted['Load_Fixed_Served'], 
                  df_sorted['Load_Flexible_Served'], 
                  df_sorted['Served_C2'],
                  labels=['Tier A (Delivered Baseload)', 'Tier B (Delivered Flexible)', 'Tier C2 (Delivered Opportunistic)'],
                  colors=['#7f7f7f', '#ff7f0e', '#2ca02c'], alpha=0.85)

    # --- THE FAILURE ZONE ---
    # Highlight exactly where Tier A was dropped (Unserved Energy) in bright red
    plt.fill_between(df_sorted.index, 
                     df_sorted['Load_Fixed_Served'], 
                     tier_a_target, 
                     where=(df_sorted['Load_Fixed_Served'] < tier_a_target),
                     color='red', alpha=0.8, label='Dropped Load (Unserved Tier A)')

    # --- THE RENEWABLE SUPPLY LINE ---
    # Plot the smooth curve of what the wind and sun are naturally providing
    plt.plot(df_sorted.index, df_sorted['Available_RE'], color='#9b59b6', linewidth=2.5, label='Available RE (Wind + Solar)')

    # --- THE BATTERY DISCHARGE BRIDGE ---
    # If the Total Served Stack is taller than the Available RE line, the battery is discharging.
    # We use a hatched pattern to show the battery "bridging the gap".
    plt.fill_between(df_sorted.index, 
                     df_sorted['Available_RE'], 
                     df_sorted['Total_Served'], 
                     where=(df_sorted['Total_Served'] > df_sorted['Available_RE']), 
                     facecolor='none', hatch='///', edgecolor='#8c564b', linewidth=0, 
                     label='Battery Discharge (Bridging the Supply Gap)')

    # --- THE EXCESS / RECHARGE ZONE ---
    # If the Available RE line is higher than the Total Served Stack, we have excess.
    # This goes into the battery or gets curtailed.
    plt.fill_between(df_sorted.index, 
                     df_sorted['Total_Served'], 
                     df_sorted['Available_RE'], 
                     where=(df_sorted['Available_RE'] > df_sorted['Total_Served']), 
                     color='#aec7e8', alpha=0.4, label='Excess RE (Charging Battery or Curtailed)')

    # --- CONSTRAINTS ---
    plt.axhline(y=tier_a_target, color='black', linestyle='--', linewidth=2, label=f'Tier A Target ({tier_a_target} MW)')
    plt.axhline(y=max_it_capacity, color='red', linestyle='-.', linewidth=2, label=f'Max IT Capacity ({max_it_capacity} MW)')

    # Formatting
    plt.title(f"EMS Dispatch Story: How Storage and Flexibility Meet Demand\n{title}", fontsize=16, fontweight='bold')
    plt.xlabel("Hours of the Year (Sorted from Highest to Lowest Renewable Generation)", fontsize=12)
    plt.ylabel("Power (MW)", fontsize=12)
    plt.xlim(0, 8760)
    plt.ylim(0, max(df_sorted['Available_RE'].max(), max_it_capacity) * 1.05)
    
    # Grid and Legend
    plt.grid(True, linestyle='--', alpha=0.4)
    # Reverse legend order to roughly match the visual vertical stack
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper right', framealpha=0.95, shadow=True)
    
    plt.tight_layout()

    # Save the plot
    plot_fn = os.path.join(current_dir, 'Storytelling_Duration_Curve.png')
    plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
    print(f"Saved storytelling duration curve to: {plot_fn}")
    plt.show()


def configure_parameters(thesis_dir):
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    sim_pars['G_MW'] = 0
    # Note: Battery efficiency override removed to match predecessor parameters
    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_offgrid_sweep_temp.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    return temp_fn

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def run_single_scenario_duration_curve():

    # --- EASY CONFIGURATION ZONE ---
    TIER_A_MW = 8.0               # Fixed baseload target
    TIER_B_MW = 8.0               # Flexible schedulable load
    MAX_IT_CAPACITY_MW = 24.0     # Total IT capacity

    # HPP Hardware Setup (Matches C2 from Table 5.9)
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    N_life = 25 * 8760
    # -------------------------------

    # Enable the C2 Sponge
    os.environ['REWARD_C2'] = '-0.5'

    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    print(f"--- Running Single Scenario ---")
    print(f"Tier A: {TIER_A_MW} MW | Tier B: {TIER_B_MW} MW | Max IT: {MAX_IT_CAPACITY_MW} MW")

    # Build Load Profiles
    t_a_ts = np.full(N_life, TIER_A_MW)
    t_b_hourly_ts = np.full(N_life, TIER_B_MW)
    t_b_daily_target_ts = np.full(N_life, TIER_B_MW * 24.0)
    total_load_for_ems = t_a_ts + t_b_hourly_ts
    total_load_for_ems[0] = MAX_IT_CAPACITY_MW

    # Initialize Model
    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_target_ts, load_profile_ts=total_load_for_ems,
        battery_deg=False # Kept False for simple Year 1 visualization
    )

    print("Evaluating design in CPLEX...")
    hpp.evaluate(*fixed_design)
    prob = hpp.prob

    # =========================================================================
    # DATA EXTRACTION FOR PLOTTING (First Year Only)
    # =========================================================================
    print("\nExtracting hourly data for Duration Curve...")
    HOURS_TO_PLOT = 8760

    wind = prob.get_val('ems.wind_t_ext')[:HOURS_TO_PLOT]
    solar = prob.get_val('ems.solar_t_ext')[:HOURS_TO_PLOT]

    # Served Loads
    unserved_a = prob.get_val('ems.Unserved_A')[:HOURS_TO_PLOT]
    served_a = TIER_A_MW - unserved_a
    served_c2 = prob.get_val('ems.Served_C2')[:HOURS_TO_PLOT]
    hpp_t = prob.get_val('ems.hpp_t')[:HOURS_TO_PLOT]
    served_b = hpp_t - served_a - served_c2

    # Build DataFrame
    df_plot = pd.DataFrame({
        'Wind': wind,
        'Solar': solar,
        'Load_Fixed_Served': served_a,
        'Load_Flexible_Served': served_b,
        'Served_C2': served_c2
    })

    # Plot it using the new smoothed function!
    plot_title = f"{TIER_A_MW}MW Tier A + {TIER_B_MW}MW Tier B (Max IT: {MAX_IT_CAPACITY_MW}MW)"
    plot_storytelling_duration_curve(df_plot, TIER_A_MW, MAX_IT_CAPACITY_MW, plot_title)

if __name__ == "__main__":
    run_single_scenario_duration_curve()