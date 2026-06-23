# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:23:17 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Phase 0 Standalone: Calculate, Save, and Plot Weekly Capacity Blocks
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  

warnings.filterwarnings("ignore", category=RuntimeWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.assembly.assembly_rolling_horizon_thijs_10_3_26 import hpp_model_constant_output_offgrid as hpp_model

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

def extract_and_plot_weekly_blocks():
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    print("\n" + "="*80)
    print(" RECALCULATING PHASE 0: WEEKLY FIRM CAPACITY BLOCKS ".center(80))
    print("="*80)
    
    N_life_1yr = 8760
    N_life_total = 25 * 8760 
    weekly_firm_mw = np.zeros(12) #monthly
    
    # Finer sweep for more accurate capacity blocks
    #test_loads = [15]
    test_loads = np.arange(6.5,16,0.5)
    
    for w in range(12): #monthly
        print(f"Evaluating Month {w+1}/12...", end=" ")
        best_mw = 0.0
        
        for mw in test_loads:
            t_a_1yr = np.zeros(N_life_1yr)
            start_idx = w * 730 #monthly
            end_idx = (w + 1) * 730 #monthly
            t_a_1yr[start_idx:end_idx] = mw
            
            t_a_ts = np.tile(t_a_1yr, 25)
            t_zero_ts = np.zeros(N_life_total)
            
            hpp = hpp_model(
                latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
                num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
                tier_a_profile=t_a_ts, tier_b_profile=t_zero_ts, tier_b2_profile=t_zero_ts, 
                load_profile_ts=t_a_ts, battery_deg=False,
                run_mode='yearly' #set runmode to either rolling or something else to do yearly
            )
            hpp.evaluate(*fixed_design)
            
            unserved_week = np.sum(hpp.prob.get_val('ems.Unserved_A')[start_idx:end_idx])
            
            if unserved_week < 1e-3:
                best_mw = mw 
            else:
                break 
                
        weekly_firm_mw[w] = best_mw
        print(f"Max Firm: {best_mw} MW")

    annual_baseload = np.min(weekly_firm_mw)
    capacity_blocks = weekly_firm_mw - annual_baseload
    
    # --- 1. SAVE TO CSV ---
    df_blocks = pd.DataFrame({
        'Week': np.arange(1, 13), #monthly
        'Total_Firm_Load_MW': weekly_firm_mw,
        'Annual_Baseload_MW': [annual_baseload] * 12, #monthly
        'Weekly_Capacity_Block_MW': capacity_blocks
    })
    csv_fn = os.path.join(current_dir, 'Monthly_Capacity_Blocks_yearlysimulation.csv') #monthly
    df_blocks.to_csv(csv_fn, index=False)
    print(f"\n✅ Data safely saved to {csv_fn}")

    # --- 2. PLOT THE RESULTS ---
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Monthly simulation: Baseload vs. Capacity Blocks', fontsize=14, fontweight='bold')

    weeks = np.arange(1, 13)#monthly
    
    # Plot the fixed baseload as the bottom stack
    ax.bar(weeks, [annual_baseload]*12, color='#1f77b4', width=1.0, edgecolor='white', label=f'Annual baseload ({annual_baseload} MW)') #monthly
    
    # Plot the flexible capacity blocks on top
    ax.bar(weeks, capacity_blocks, bottom=[annual_baseload]*12, color='#ff7f0e', width=1.0, edgecolor='white', label='Monthly Capacity Blocks') #monthly

    ax.set_ylabel('Monthly firm capacity (MW)') #monthly
    ax.set_xlabel('Month of the Year')
    ax.set_xlim(0.5, 12.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'Thesis_Plot_Monthly_Capacity_Blocks.svg')
    fig.savefig(plot_fn, format='svg', dpi=300)
    print(f"✅ Plot successfully saved to: {plot_fn}")
    plt.show()

if __name__ == "__main__":
    extract_and_plot_weekly_blocks()