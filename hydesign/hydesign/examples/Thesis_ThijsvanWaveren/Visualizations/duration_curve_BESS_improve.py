# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:52:02 2026

@author: thijs
"""
# -*- coding: utf-8 -*-
"""
Plot: Available RE vs. System Power (with BESS) Duration Curve
(Upgraded with Academic-Consulting Formatting)
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- DIRECTORY SETUP & HYDESIGN IMPORTS ---
# 1. Setup local directories so the script can find your weather and YAML files
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))

# 2. Force Python to prioritize the Downloads folder above all else for the package
ROOT_DIR = r"C:\Users\thijs\Downloads\hydesign"
if sys.path[0] != ROOT_DIR:
    sys.path.insert(0, ROOT_DIR)

# 3. Import your specific custom assembly
from hydesign.assembly.hpp_assembly_offgrid_thijs_2_2_26 import hpp_model_constant_output_offgrid as hpp_model

# ==============================================================================
# BESS CHRONOLOGICAL SIMULATION
# ==============================================================================
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

# ==============================================================================
# UPDATED PLOTTING FUNCTION
# ==============================================================================

def plot_re_vs_bess_duration_curve(df):
    """
    Plots a duration curve showing original Wind+Solar vs. System Power after BESS,
    with explicit shaded regions and annotations to highlight the bottleneck.
    """
    # 1. Calculate combined Renewable Energy
    df['Available_RE'] = df['Wind'] + df['Solar']

    # 2. Apply Chronological BESS
    df['System_Power'] = apply_chronological_bess(df['Available_RE'].values, 
                                                  target_baseload=8.0, 
                                                  p_bess=25.0, 
                                                  e_bess=200.0)

    # 3. Sort independently to create duration curves
    re_sorted = df['Available_RE'].sort_values(ascending=False).reset_index(drop=True)
    sys_sorted = df['System_Power'].sort_values(ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

    # --- SHADE UNDER THE CURVE ---
    
    # 1. Matched Demand (Firm 8 MW Baseload) - Solid Green
    ax.fill_between(sys_sorted.index, 0, np.minimum(sys_sorted, 8.0), 
                    color='#2ca02c', alpha=0.8, zorder=2)

    # 2. Excess Generation (Stranded Energy) - Solid Blue
    ax.fill_between(sys_sorted.index, 8.0, sys_sorted, where=(sys_sorted > 8.0), 
                    color='#1f77b4', alpha=0.7, zorder=2)

    # 3. BESS Charging (Absorbing the Peak) - Bright Orange
    # Where Raw Generation is higher than System Power
    ax.fill_between(re_sorted.index, sys_sorted, re_sorted, where=(re_sorted > sys_sorted), 
                    color='#f39c12', alpha=1.0, zorder=3)

    # 4. BESS Discharging (Securing the Baseload) - Bright Red
    # Where System Power is higher than Raw Generation
    ax.fill_between(re_sorted.index, re_sorted, sys_sorted, where=(sys_sorted > re_sorted), 
                    color='#e74c3c', alpha=1.0, zorder=4)

    # --- PLOT THE CURVE OUTLINES ---
    ax.plot(sys_sorted.index, sys_sorted, color='#08306b', linewidth=1.5, zorder=5)
    ax.plot(re_sorted.index, re_sorted, color='grey', linewidth=1.0, alpha=0.5, zorder=1)

    # Target Line
    ax.axhline(y=8.0, color='red', linestyle='--', linewidth=1.5, zorder=6)

    # --- ADD ON-CHART TEXT ANNOTATIONS ---
    
    # Excess Generation
    ax.text(2500, 80, "EXCESS GENERATION\n(Curtailed without Flexible Workloads)", 
            fontsize=12, fontweight='bold', color='white', ha='center', va='center',
            bbox=dict(facecolor='#08306b', edgecolor='none', alpha=0.8, pad=4), zorder=7)

    # Firm Baseload
    ax.text(4000, 20, "MATCHED DEMAND\n(Firm 8 MW Baseload)", 
            fontsize=11, fontweight='bold', color='#155d15', ha='center', va='bottom', zorder=7)
    ax.annotate("", xy=(4000, 8.5), xytext=(4000, 19),
                arrowprops=dict(facecolor='#155d15', arrowstyle="->", lw=2), zorder=7)

    # BESS Charging Label (Moved to the middle/right of the curve)
    ax.annotate("BESS CHARGING\n(Absorbing Excess)",
                xy=(5000, 45), xycoords='data',    # Pointing to the curve around hour 5000
                xytext=(6000, 120), textcoords='data', # Text sits nicely in the empty white space
                arrowprops=dict(facecolor='#f39c12', arrowstyle="wedge,tail_width=0.7", alpha=0.8),
                fontsize=10, fontweight='bold', color='#c27d0e', ha='center', zorder=7)

    # BESS Discharging Label
    ax.annotate("BESS DISCHARGING\n(Securing 99.9% Reliability)",
                xy=(8500, 4), xycoords='data',
                xytext=(7200, 50), textcoords='data',
                arrowprops=dict(facecolor='#e74c3c', arrowstyle="wedge,tail_width=0.7", alpha=0.8),
                fontsize=10, fontweight='bold', color='#b83c30', ha='center', zorder=7)

    # --- FORMATTING ---
    # Updated Title based on supervisor feedback
    ax.set_title("Hybrid Power Plant Duration Curve: Powering a 99.9% Reliable Baseload", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Hours of the Year", fontsize=12, fontweight='bold', color='grey')
    ax.set_ylabel("Power (MW)", fontsize=12, fontweight='bold', color='grey')
    
    ax.set_xlim(0, 8760)
    ax.set_ylim(0, re_sorted.max() * 1.05)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Custom Legend to keep it perfectly clean
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Matched Firm Demand (8 MW)'),
        Patch(facecolor='#1f77b4', label='Excess Generation (Stranded)'),
        Patch(facecolor='#f39c12', label='BESS Charging'),
        Patch(facecolor='#e74c3c', label='BESS Discharging')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plot_fn = os.path.join(current_dir, 'BESS_Effect_Duration_Curve_Annotated.png')
    plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Annotated BESS Effect duration curve to: {plot_fn}")
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