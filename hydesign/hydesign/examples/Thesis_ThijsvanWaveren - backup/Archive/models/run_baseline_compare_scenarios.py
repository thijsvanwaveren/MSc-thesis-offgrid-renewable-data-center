# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:16:45 2026
Updated: Fixes curtailment variable access and scenario names.

@author: thijs
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hydesign.ems.ems import expand_to_lifetime
from hydesign.assembly.hpp_assembly_offgrid_thijs_2_2_26 import hpp_model_constant_output_offgrid as hpp_model
from hydesign.examples.Thesis_ThijsvanWaveren.models.datacenter import DataCenterModel 

# --- 1. PATHS & SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

site_name = 'Denmark_good_wind'
sites_csv_path = os.path.join(thesis_dir, '..', 'examples_sites.csv')
examples_sites = pd.read_csv(sites_csv_path, sep=';')
ex_site = examples_sites.loc[examples_sites.name == site_name]
weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
sim_pars_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')

# --- 2. CONFIGURATION ---
# UPDATED: Using valid keys from your error message
scenarios = [
    "Not_Flexible", 
    "Batch_Traditional", 
    "Batch_Focused", 
    "Interactive_Traditional",  # Was "High_Flexibility" (invalid)
    "Infinitely_Flexible"
]

# Fixed Design: [clearance, sp, p_rated, Nwt, wind_MW_km2, solar_MW, tilt, azimuth, DCAC, b_P, b_E_h, cost_deg]
# 15 Turbines (6MW), 50MW Solar, 50MW / 12h Battery
fixed_design = [35, 300, 5, 11, 7, 102, 39, 180, 1.25, 30, 16, 10]

print(f"🚀 Starting Comparison Study: Fixed Hardware Design")
print(f"   Design: {fixed_design[3]}x{fixed_design[2]}MW Wind, {fixed_design[5]}MW Solar, {fixed_design[9]}MW/{fixed_design[10]}h Battery")
print("-" * 80)

# --- 3. EXECUTION FUNCTION ---
def run_scenario_evaluation(scenario_name, design_vector):
    
    # A. Generate Load Profiles (Cap = 20MW)
    dc_model = DataCenterModel(total_it_capacity=20, pue=1.15)
    tier_a_1yr, tier_b_1yr_daily = dc_model.generate_profile(scenario_name)
    
    # B. Expand to Lifetime
    tier_a_25y = expand_to_lifetime(tier_a_1yr, life=25*8760)
    tier_b_input = np.tile(np.repeat(tier_b_1yr_daily, 24), 25)
    tier_b_hourly_avg = np.tile(np.repeat(tier_b_1yr_daily / 24.0, 24), 25)
    total_load_25y = tier_a_25y + tier_b_hourly_avg
    
    # C. Initialize HPP
    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0],
        longitude=ex_site['longitude'].values[0],
        altitude=ex_site['altitude'].values[0],
        num_batteries=2,
        work_dir=current_dir,
        input_ts_fn=weather_fn,
        sim_pars_fn=sim_pars_fn,
        tier_a_profile=tier_a_25y,
        tier_b_profile=tier_b_input,
        load_profile_ts=total_load_25y,
        battery_deg=False,
    )
    
    # D. Evaluate
    res = hpp.evaluate(*design_vector)
    prob = hpp.prob
    
    # E. Extract Metrics
    
    # 1. Reliability
    unserved_a = prob.get_val('ems.Unserved_A') 
    shortfall_b = prob.get_val('ems.Shortfall_B') 
    total_unmet_MWh = np.sum(unserved_a) + np.sum(shortfall_b)
    total_demand_MWh = np.sum(total_load_25y)
    reliability_pct = 100 * (1 - (total_unmet_MWh / total_demand_MWh))
    
    # 2. Financials
    lcoe = float(res[3])
    capex = float(prob.get_val('finance.CAPEX')[0])
    
    # 3. Curtailment (FIXED: Sum the timeseries manually)
    # The variable 'ems.total_curtailment' does not exist in the simplified component.
    # We grab the hourly series 'ems.hpp_curt_t' (MW) and sum it up -> MWh.
    curtailment_ts = prob.get_val('ems.hpp_curt_t')
    curtailment_MWh = np.sum(curtailment_ts)
    
    return reliability_pct, lcoe, capex, curtailment_MWh

# --- 4. MAIN LOOP ---
results_data = []

print(f"{'Scenario':<25} | {'Rel %':<10} {'LCOE':<10} | {'Curt.(GWh)':<12}")
print("-" * 65)

for scen in scenarios:
    try:
        rel, lcoe, capex, curt = run_scenario_evaluation(scen, fixed_design)
        
        # Print Row
        print(f"{scen:<25} | {rel:<10.3f} {lcoe:<10.2f} | {curt/1e3:<12.2f}")
        
        results_data.append({
            'Scenario': scen,
            'Reliability': rel,
            'LCOE': lcoe,
            'Curtailment': curt
        })
        
    except Exception as e:
        print(f"❌ Error running {scen}: {e}")

# --- 5. PLOTTING ---
if not results_data:
    print("\n❌ All scenarios failed. No plot generated.")
else:
    df_res = pd.DataFrame(results_data)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot LCOE (Bar)
    color = 'tab:blue'
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('LCOE (EUR/MWh)', color=color)
    bars = ax1.bar(df_res['Scenario'], df_res['LCOE'], color=color, alpha=0.6, label='LCOE')
    ax1.tick_params(axis='y', labelcolor=color)
    if df_res['LCOE'].max() > 0:
        ax1.set_ylim(0, df_res['LCOE'].max() * 1.2)

    # Plot Reliability (Line)
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Reliability (%)', color=color)
    line = ax2.plot(df_res['Scenario'], df_res['Reliability'], color=color, marker='o', linewidth=3, label='Reliability')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Dynamic scaling for Reliability axis
    min_rel = df_res['Reliability'].min()
    ax2.set_ylim(min(80, min_rel - 2), 100.5)

    plt.title(f"Impact of Flexibility on Performance (Fixed Design)")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n✅ Comparison Complete.")