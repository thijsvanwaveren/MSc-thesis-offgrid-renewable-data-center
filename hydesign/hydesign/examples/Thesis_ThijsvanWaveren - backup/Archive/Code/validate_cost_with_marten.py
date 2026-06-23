# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:07:05 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Evaluate 16 MW Static Baseload for Denmark_good_solar
Outputs Temporal Reliability and LCOED
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- HYDESIGN IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

# Ensure this points to your latest valid assembly!
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

def run_16mw_baseload():
    print("\n" + "="*60)
    print(" EVALUATING 16 MW STATIC BASELOAD ".center(60))
    print("="*60)
    
    # Standard design [Clearance, Sp, P_rated, Nwt, wind_MW_per_km2, solar_MW, surface_tilt, azimuth, dc_ac, b_P, b_E_h, cost_ratio]
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    # Simulation constants
    N_life_total = 25 * 8760
    TIER_A_MW = 16.0
    
    # Create the rigid 16 MW profile for the entire 25 years
    t_a_ts = np.full(N_life_total, TIER_A_MW)
    t_zero_ts = np.zeros(N_life_total)
    
    # We bypass the IT capacity cap logic by setting the first element high
    total_load_for_ems = t_a_ts.copy()
    total_load_for_ems[0] = 500.0 

    os.environ['REWARD_C2'] = '1.0'  # Disable opportunistic load

    print(f"Running simulation for {site_name} (Yearly Solver)...")
    
    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_a_ts, tier_b_profile=t_zero_ts, tier_b2_profile=t_zero_ts, 
        load_profile_ts=total_load_for_ems, battery_deg=False,
        run_mode='yearly' # Using perfect foresight for Stage 1 static sizing
    )
    
    # Run the evaluation
    out = hpp.evaluate(*fixed_design)
    prob = hpp.prob
    
    # --- METRICS CALCULATION ---
    unserved_a_ts = prob.get_val('ems.Unserved_A')
    
    # 1. Temporal Reliability (Hours completely served)
    hours_failed = np.sum(unserved_a_ts > 1e-3)
    temporal_rel = 100.0 * (1.0 - (hours_failed / N_life_total))
    
    # 2. Energy Reliability (MWh completely served)
    total_demanded_mwh = np.sum(t_a_ts)
    total_unserved_mwh = np.sum(unserved_a_ts)
    energy_rel = 100.0 * (1.0 - (total_unserved_mwh / total_demanded_mwh))
    
    # 3. LCOED Calculation
    try:
        lcoed = prob.get_val('finance.LCOE_delivered')[0]
    except Exception:
        # Fallback if variable is named differently in your specific assembly
        lcoed = out[3]

    print("\n" + "-"*60)
    print(" RESULTS ".center(60))
    print("-"*60)
    print(f"Target Baseload:       {TIER_A_MW} MW")
    print(f"Total Demanded Energy: {total_demanded_mwh / 1000.0:,.2f} GWh")
    print(f"Total Unserved Energy: {total_unserved_mwh / 1000.0:,.2f} GWh")
    print(f"Hours Failed:          {hours_failed} hours (out of {N_life_total})")
    print("-"*60)
    print(f"TEMPORAL RELIABILITY:  {temporal_rel:.4f} %")
    print(f"ENERGY RELIABILITY:    {energy_rel:.4f} %")
    print(f"LCOE Delivered:        € {lcoed:.2f} / MWh")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_16mw_baseload()