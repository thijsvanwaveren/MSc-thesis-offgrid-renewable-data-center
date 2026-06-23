# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:16:02 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Evaluate Solar PV Tilt: 37.4 degrees vs 39.0 degrees
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

def run_tilt_scenario(tilt_degrees):
    """Runs the assembly with zero load/battery to isolate solar generation."""
    # design array: index 6 is surface_tilt
    # [clearance, sp, p_rated, Nwt, wind_MW_per_km2, solar_MW, surface_tilt, azimuth, dc_ac, b_P, b_E_h, cost_ratio]
    design = [35, 300, 5, 10, 7, 112, tilt_degrees, 180, 1.25, 0, 0, 10] 
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)
    
    with open(sim_pars_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    life_h = sim_pars['N_life'] * 365 * 24  
    
    # Dummy profiles for isolated generation test
    t_zero = np.zeros(life_h)
    
    os.environ['REWARD_C2'] = '1.0' 
    
    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_zero, tier_b_profile=t_zero, load_profile_ts=t_zero, battery_deg=False
    )
    hpp.evaluate(*design)
    return hpp.prob

def compare_tilts():
    for tilt in np.arange(40,53,0.25):
        prob = run_tilt_scenario(tilt)
        solar_ts = prob.get_val('ems.solar_t_ext')
        gen_gwh = np.sum(solar_ts)/1000
        print(f"Generation @ {tilt}° : {gen_gwh:,.2f} GWh")
        
if __name__ == "__main__":
    compare_tilts()