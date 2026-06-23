# -*- coding: utf-8 -*-
"""
Thesis Parametric Sweep: IT Capacity Impact on Workload Stacking
Workload: A=8.0 MW, B1=7.0 MW, B2=8.0 MW (Total Average = 23.0 MW)
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

def plot_it_capacity_sweep(df):
    """Generates a 3-panel plot visualizing the value of IT Headroom and Tier C"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('Impact of IT Capacity Limit on Performance of 18 MW (A), 2 MW (B1), 5 MW (B2) Workload Stack', fontsize=16, fontweight='bold')

    x = np.arange(len(df))
    width = 0.35
    
    # Panel 1: Reliability
    ax1.plot(x, df['Rel_A'], marker='o', color='#1f77b4', linewidth=2, markersize=8, label='Tier A (Baseload)')
    ax1.plot(x, df['Rel_B1'], marker='^', color='#ff7f0e', linewidth=2, markersize=8, label='Tier B1 (24h SLA)')
    ax1.plot(x, df['Rel_B2'], marker='s', color='#8c564b', linewidth=2, markersize=8, label='Tier B2 (168h SLA)')
    ax1.set_ylabel('Reliability (%)', fontsize=12)
    ax1.set_title('Workload Reliability', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Panel 2: Curtailment Comparison
    ax2.bar(x - width/2, df['Curt_No_C'], width, label='Curtailment (No Tier C)', color='#d62728', edgecolor='black')
    ax2.bar(x + width/2, df['Curt_With_C'], width, label='Curtailment (With Tier C)', color='#2ca02c', edgecolor='black')
    ax2.set_ylabel('Curtailed Energy (GWh/25y)', fontsize=12)
    ax2.set_title('Curtailed Energy over 25 Years', fontsize=12)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Panel 3: LCOE Comparison
    bars_no_c = ax3.bar(x - width/2, df['LCOE_No_C'], width, label='LCOE (No Tier C)', color='lightgray', edgecolor='black')
    bars_with_c = ax3.bar(x + width/2, df['LCOE_With_C'], width, label='LCOE (With Tier C)', color='#9467bd', edgecolor='black')
    ax3.set_ylabel('LCOED (€/MWh)', fontsize=12)
    ax3.set_xlabel('IT Capacity Limit (MW)', fontsize=12)
    ax3.set_title('Levelized Cost of Electricity Delivered', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{cap:.1f}" for cap in df['IT_Capacity_MW']])
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add exact value labels on top of the LCOE bars
    for bar in bars_no_c:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 1, f'€{yval:.0f}', ha='center', va='bottom', fontsize=9, rotation=45)
    for bar in bars_with_c:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 1, f'€{yval:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_fn = os.path.join(current_dir, 'Thesis_Plot_IT_Capacity_Sweep.svg')
    plt.savefig(plot_fn, format='svg', dpi=300)
    print(f"\n✅ Plot successfully saved to: {plot_fn}")
    plt.show()

if __name__ == "__main__":
    
    # 1. FIXED PARAMETERS
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    N_life = 25 * 8760
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)
    
    # Fixed Data Center Workloads
    WORKLOAD_A_MW = 18.0
    WORKLOAD_B1_MW = 2.0
    WORKLOAD_B2_MW = 5.0
    
    t_a_ts = np.full(N_life, WORKLOAD_A_MW)
    t_b1_ts = np.full(N_life, WORKLOAD_B1_MW * 24.0)
    t_b2_ts = np.full(N_life, WORKLOAD_B2_MW * 168.0) 
    
    base_load_profile_ts = np.full(N_life, WORKLOAD_A_MW + WORKLOAD_B1_MW + WORKLOAD_B2_MW)

    # 2. PARAMETER SWEEP LIST
    it_capacity_list = [25.0,  27.5, 30.0, 32.5, 35.0, 40.0, 60.0, 100.0, 300.0]
    all_results = []
    
    print(f"\n--- Starting IT Capacity Sweep (Target Load = {WORKLOAD_A_MW+WORKLOAD_B1_MW+WORKLOAD_B2_MW} MW) ---")

    for it_cap in it_capacity_list:
        print(f"Evaluating IT Capacity: {it_cap} MW...")
        current_load_for_ems = base_load_profile_ts.copy()
        current_load_for_ems[0] = it_cap 

        # --- SCENARIO 1: NO TIER C ---
        os.environ['REWARD_C2'] = '1.0'  
        hpp_no_c = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b1_ts, tier_b2_profile=t_b2_ts, load_profile_ts=current_load_for_ems, battery_deg=False
        )
        out_noc = hpp_no_c.evaluate(*fixed_design)
        p_noc = hpp_no_c.prob
        
        rel_a = 100.0 * (1.0 - (np.sum(p_noc.get_val('ems.Unserved_A') > 1e-3) / N_life))
        rel_b1 = 100.0 * (1.0 - (np.sum(np.sum(p_noc.get_val('ems.Shortfall_B').reshape(-1, 24), axis=1) > 1e-3) / (N_life/24)))
        n_w = N_life // 168
        rel_b2 = 100.0 * (1.0 - (np.sum(np.sum(p_noc.get_val('ems.Shortfall_B2')[:n_w * 168].reshape(-1, 168), axis=1) > 1e-3) / n_w))
        
        curt_noc = np.sum(p_noc.get_val('ems.hpp_curt_t')) / 1000.0
        
        # FIX: Extract LCOE securely, falling back directly to out_noc[3] (which is prob['LCOE'])
        try: lcoe_noc = p_noc.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_noc = out_noc[3]

        # --- SCENARIO 2: WITH TIER C ---
        os.environ['REWARD_C2'] = '-0.5' 
        hpp_with_c = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b1_ts, tier_b2_profile=t_b2_ts, load_profile_ts=current_load_for_ems, battery_deg=False
        )
        out_c = hpp_with_c.evaluate(*fixed_design)
        p_c = hpp_with_c.prob
        
        curt_c = np.sum(p_c.get_val('ems.hpp_curt_t')) / 1000.0
        
        # FIX: Extract LCOE securely
        try: lcoe_c = p_c.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_c = out_c[3]

        all_results.append({
            'IT_Capacity_MW': it_cap,
            'Rel_A': rel_a, 'Rel_B1': rel_b1, 'Rel_B2': rel_b2,
            'Curt_No_C': curt_noc, 'Curt_With_C': curt_c,
            'LCOE_No_C': lcoe_noc, 'LCOE_With_C': lcoe_c
        })

    # Generate Plot
    df_results = pd.DataFrame(all_results)
    plot_it_capacity_sweep(df_results)