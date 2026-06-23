# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:23:07 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Plot: Available RE (Wind + Solar) Duration Curve
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
# SIMPLIFIED PLOTTING FUNCTION
# ==============================================================================

def plot_re_duration_curve(df):
    """
    Plots a duration curve showing only available Wind and Solar generation.
    Sorted from highest availability to lowest.
    """
    # 1. Calculate combined Renewable Energy
    df['Available_RE'] = df['Wind'] + df['Solar']

    # 2. Sort by 'Available_RE' descending to create the duration curve
    df_sorted = df.sort_values(by='Available_RE', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6))

    # --- THE RENEWABLE SUPPLY LINE ---
    plt.plot(df_sorted.index, df_sorted['Available_RE'], color='#2ca02c', linewidth=2.5, label='Available RE (Wind + Solar)')
    
    # Fill under the curve for better visual weight
    plt.fill_between(df_sorted.index, 0, df_sorted['Available_RE'], color='#2ca02c', alpha=0.2)

    # Formatting
    plt.title("Available Renewable Generation Duration Curve", fontsize=16, fontweight='bold')
    plt.xlabel("Hours of the Year", fontsize=12)
    plt.ylabel("Available Generation (MW)", fontsize=12)
    plt.xlim(0, 8760)
    plt.ylim(0, df_sorted['Available_RE'].max() * 1.05)
    
    # Grid and Legend
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', framealpha=0.95, shadow=True, fontsize=11)
    
    plt.tight_layout()

    # Save the plot
    plot_fn = os.path.join(current_dir, 'Available_RE_Duration_Curve.png')
    plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Available RE duration curve to: {plot_fn}")
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
    plot_re_duration_curve(df_plot)

if __name__ == "__main__":
    run_generation_extraction()