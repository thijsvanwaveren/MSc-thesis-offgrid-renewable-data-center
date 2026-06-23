import os
import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# --- 1. PATHS & SITE SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.Parallel_EGO_adjusted_3_2_26 import EfficientGlobalOptimizationDriver
from hydesign.assembly.hpp_assembly_offgrid_thijs_2_2_26 import hpp_model_constant_output_offgrid as hpp_model

site_name = 'Denmark_good_wind'
sites_csv_path = os.path.join(thesis_dir, '..', 'examples_sites.csv')
examples_sites = pd.read_csv(sites_csv_path, sep=';')
ex_site = examples_sites.loc[examples_sites.name == site_name]

# These need to be accessible both to the Model and the Inputs dictionary
longitude = ex_site['longitude'].values[0]
latitude = ex_site['latitude'].values[0]
altitude = ex_site['altitude'].values[0]
input_ts_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
sim_pars_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')

# --- 2. LOAD DEFINITION ---
it_capacity = 16 
pue = 1
total_load_mw = it_capacity * pue # 23 MW
life_years = 25
hours_lifetime = life_years * 8760

tier_a_profile = np.full(hours_lifetime, total_load_mw)
tier_b_profile = np.zeros(hours_lifetime)
load_profile = tier_a_profile + tier_b_profile
batt_eff_one_way = np.sqrt(0.86)

# --- 3. CUSTOM WRAPPER WITH SOFT PENALTY ---
class HPP_Optimization_Wrapper(hpp_model): 
    def __init__(self, **kwargs):
        kwargs['G_MW'] = 0 
        super().__init__(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            num_batteries=5,
            work_dir=current_dir,
            input_ts_fn=input_ts_fn,
            sim_pars_fn=sim_pars_fn,
            tier_a_profile=tier_a_profile,
            tier_b_profile=tier_b_profile,
            load_profile_ts=load_profile,
            battery_deg=False, 
            battery_charge_efficiency=batt_eff_one_way,
        )

    def evaluate(self, *args, **kwargs):
        # 1. Run standard physics
        outs = super().evaluate(*args, **kwargs)
        
        try:
            # 2. Extract results for Reliability calculation
            unserved_a = self.prob.get_val('ems.Unserved_A')
            shortfall_b = self.prob.get_val('ems.Shortfall_B')
            total_unmet = np.sum(unserved_a) + np.sum(shortfall_b)
            total_load_demand = np.sum(self.load_profile_ts)
            
            reliability = 1.0 - (total_unmet / total_load_demand)
            real_lcoe = outs[3]
            
            # 3. Apply Soft Penalty logic
            target_reliability = 0.934
            if reliability < target_reliability:
                # Penalty factor: for every 1% missing reliability, add 10 Euro to LCOE
                # This creates a 'gradient' the optimizer can follow
                penalty = (target_reliability - reliability) * 1000 
                constrained_lcoe = real_lcoe + penalty
                status = "❌ FAIL"
            else:
                constrained_lcoe = real_lcoe
                status = "✅ PASS"

            # 4. MONITORING: Print reliability and sizing for every evaluation
            # Extracting Nwt (args[0]) and solar (args[1]) as examples
            print(f"{status} | Rel: {reliability:.2%} | Real LCOE: {real_lcoe:.2f} | Result: {constrained_lcoe:.2f}")

            # 5. Overwrite LCOE in the output array for the optimizer
            outs[3] = constrained_lcoe
            
        except Exception as e:
            print(f"Wrapper Error: {e}")
            outs[3] = 1e6 # Fallback penalty
            
        return outs

# --- 4. OPTIMIZATION CONFIG ---
inputs = {
    # FIX: Explicitly passing site data into inputs dictionary to avoid KeyError
    'longitude': longitude,
    'latitude': latitude,
    'altitude': altitude,
    
    'name': site_name,
    'n_procs': int(os.cpu_count()) - 2,
    'num_batteries': 5,   
    'n_doe': 35,            # Slightly higher for the wider bounds
    'n_clusters': 5,       
    'n_seed': 42,
    'max_iter': 15,    
    'final_design_fn': f'opt_results_90_soft_{site_name}.csv',
    'npred': 1e4,
    'tol': 1e-6,
    'min_conv_iter': 3,
    'work_dir': current_dir,
    'hpp_model': HPP_Optimization_Wrapper, 
    'opt_var': "LCOE [Euro/MWh]", 
    
    'variables': {
        'clearance [m]': {'var_type': 'fixed', 'value': 35},
        'sp [W/m2]': {'var_type': 'fixed', 'value': 300},
        'p_rated [MW]': {'var_type': 'fixed', 'value': 5},
        'wind_MW_per_km2 [MW/km2]': {'var_type': 'fixed', 'value': 7},
        'surface_tilt [deg]': {'var_type': 'fixed', 'value': 39},
        'surface_azimuth [deg]': {'var_type': 'fixed', 'value': 180},
        'DC_AC_ratio': {'var_type': 'fixed', 'value': 1.25},
        'cost_of_battery_P_fluct_in_peak_price_ratio': {'var_type': 'fixed', 'value': 10},

        'Nwt': {'var_type': 'design', 'limits': [2, 20], 'types': 'int'},
        'solar_MW [MW]': {'var_type': 'design', 'limits': [20, 200], 'types': 'int'},
        'b_P [MW]': {'var_type': 'design', 'limits': [10, 50], 'types': 'int'}, 
        'b_E_h [h]': {'var_type': 'design', 'limits': [4, 24], 'types': 'int'},
    }
}

if __name__ == '__main__':
    print(f"🚀 Starting Optimization: Soft Penalty for <90% Reliability")
    opt_driver = EfficientGlobalOptimizationDriver(**inputs)
    opt_driver.run()
    print("\n✅ Optimization Complete.")