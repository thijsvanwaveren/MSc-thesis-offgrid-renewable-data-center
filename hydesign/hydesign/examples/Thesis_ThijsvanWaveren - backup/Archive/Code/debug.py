# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 14:26:40 2026

@author: thijs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hydesign.assembly.hpp_assembly_offgrid_thijs_2_2_26 import hpp_model_constant_output_offgrid as hpp_model
from hydesign.examples.Thesis_ThijsvanWaveren.models.datacenter import DataCenterModel 
from hydesign.ems.ems import expand_to_lifetime
import os
import sys

# --- 1. DYNAMIC PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

# --- 2. CONFIGURE SITE & PARAMETERS ---
site_name = 'Denmark_good_wind'
examples_dir = os.path.abspath(os.path.join(thesis_dir, '..'))
sites_csv_path = os.path.join(examples_dir, 'examples_sites.csv')
examples_sites = pd.read_csv(sites_csv_path, sep=';')
ex_site = examples_sites.loc[examples_sites.name == site_name]
weather_fn = os.path.join(examples_dir, ex_site['input_ts_fn'].values[0])
sim_pars_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')

# Mock load (Tier A=0, Tier B=20MW flat for simplicity in debugging)
# We use a simple load to ensure the battery MUST work
N_life = 25
life_h = N_life * 365 * 24
tier_a = np.zeros(life_h) 
tier_b = np.full(life_h, 20.0) # 20 MW demand every hour
total_load = tier_a + tier_b

# --- FUNCTION TO RUN A SINGLE DESIGN ---
def get_battery_trace(b_hours):
    print(f"\n--- Running Debug Case: {b_hours} Hour Battery ---")
    
    hpp = hpp_model(
        latitude=55.0, longitude=8.0, altitude=50.0, num_batteries=2,
        work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=tier_a, tier_b_profile=tier_b, load_profile_ts=total_load,
        battery_deg=False
    )
    
    # [clearance, sp, p_rated, Nwt, wind_MW_km2, solar_MW, tilt, azimuth, DCAC, b_P, b_E_h, cost_deg]
    # NOTE: High Wind (20 Turbines) and High Battery Power (50MW) to force usage
    x = [35, 300, 6, 20, 7, 50, 25, 180, 1.25, 50, b_hours, 10]
    
    hpp.evaluate(*x)
    
    # Extract SOC (first year only for plotting)
    soc = hpp.prob.get_val('ems.b_E_SOC_t')[:8760]
    return soc

# --- EXECUTE ---
try:
    soc_4h = get_battery_trace(4)
    soc_48h = get_battery_trace(48)

    # --- PLOT RESULTS ---
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Absolute Energy (MWh)
    plt.subplot(1, 2, 1)
    plt.plot(soc_4h, label='4h Battery (Small)', alpha=0.7)
    plt.plot(soc_48h, label='48h Battery (Large)', alpha=0.7)
    plt.title("Absolute Energy Stored (MWh)")
    plt.ylabel("Energy (MWh)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: State of Charge (%)
    # We normalize by their respective capacities
    plt.subplot(1, 2, 2)
    plt.plot(soc_4h / (50*4) * 100, label='4h SOC %', alpha=0.7)
    plt.plot(soc_48h / (50*48) * 100, label='48h SOC %', alpha=0.7)
    plt.title("State of Charge (%)")
    plt.ylabel("SOC %")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("\nDEBUG DIAGNOSIS:")
    print(f"4h  Max Energy: {np.max(soc_4h):.1f} MWh")
    print(f"48h Max Energy: {np.max(soc_48h):.1f} MWh")
    
    if np.max(soc_48h) > 200:
        print("✅ Solver SEES the larger capacity.")
    else:
        print("❌ Solver DOES NOT SEE the capacity (48h maxed out near 200).")

except Exception as e:
    print(f"CRITICAL FAIL: {e}")