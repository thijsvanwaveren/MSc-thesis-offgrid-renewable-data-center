# -*- coding: utf-8 -*-
"""
Independent Workload Capacity Sizing (Yearly Optimization)
Focus: Apples-to-apples comparison of how flexibility increases absolute capacity limits.
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

def generate_thesis_plots(df1, df2, df2b, df3, base_a, base_b1, base_b2):
    """
    Generates a single, unified plot comparing the absolute capacity limits 
    of Tier A, Tier B1, and Tier B2 independently.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle('Reliability levels for workload tiers', fontsize=16, fontweight='bold')

    # --- PLOT THE THREE INDEPENDENT CURVES ---
    
    # 1. Baseload (Tier A)
    ax.plot(df1['Tier_A_MW'], df1['Reliability_Time_A'], 
            marker='o', color='#1f77b4', linewidth=3, markersize=8, 
            label='Tier A (Firm Load)')
    
    # 2. Daily Batch (Tier B1)
    ax.plot(df2['Tier_B_MW'], df2['Reliability_Deadline_B'], 
            marker='s', color='#ff7f0e', linewidth=3, markersize=8, 
            label='Tier B1 (Daily Flexible)')
            
    # 3. Weekly Batch (Tier B2)
    ax.plot(df2b['Tier_B2_MW'], df2b['Reliability_Deadline_B2'], 
            marker='^', color='#2ca02c', linewidth=3, markersize=8, 
            label='Tier B2 (Weekly Flexible)')

    # --- FORMATTING ---
    # ax.axhline(99.9, color='red', linestyle=':', alpha=0.8, linewidth=2, label='99.9% Target SLA')
    # ax.axhline(95.0, color='gray', linestyle=':', alpha=0.6, linewidth=1.5, label='95.0% Target SLA')

    ax.set_ylabel('SLA Reliability (%)', fontsize=12)
    ax.set_xlabel('Workload Capacity (MW)', fontsize=12)
    
    # Add a slight drop shadow to the grid for readability
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=11, loc='lower left')

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'Thesis_Independent_Reliability_Comparison.svg')
    fig.savefig(plot_fn, format='svg', dpi=300)
    print(f"\n✅ Plot successfully saved to: {plot_fn}")
    plt.show()

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

def print_stacked_services_summary(df_res, stage_name):
    print(f"\n{'='*125}")
    print(f"{stage_name:^125}")
    print(f"{'='*125}")
    print(f"{'Tier A (MW)':<11} | {'Tier B1 (MW)':<12} | {'Tier B2 (MW)':<12} | {'Max IT (MW)':<11} | {'LCOEd (€)':<9} | {'Rel(A)':<11} | {'Rel(B1)':<11} | {'Rel(B2)':<11} | {'Curt(GWh)':<9} | {'C2(GWh)':<9}")
    print("-" * 125)
    for _, row in df_res.iterrows():
        print(f"{row.get('Tier_A_MW',0):<11.1f} | {row.get('Tier_B_MW',0):<12.1f} | {row.get('Tier_B2_MW',0):<12.1f} | {row.get('Max_IT_MW',0):<11.1f} | "
              f"{row.get('LCOE',0):<9.2f} | {row.get('Reliability_Time_A',0):<10.2f}% | "
              f"{row.get('Reliability_Deadline_B',100):<10.2f}% | "
              f"{row.get('Reliability_Deadline_B2',100):<10.2f}% | "
              f"{row.get('Curtailment_GWh',0):<9.2f} | {row.get('C2_GWh',0):<9.2f}")
    print("="*125 + "\n")

def run_firm_yield_analysis():
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    MAX_IT_CAPACITY_MW = 500.0 # Bumped up slightly to ensure high B2 sweeps don't hit an artificial ceiling
    N_life = 25 * 8760
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    print(f"--- Starting Independent Workload Analysis (Max IT: {MAX_IT_CAPACITY_MW} MW) ---")

    # =========================================================================
    # DISABLE TIER C2 SPONGE FOR ALL PRIMARY SWEEPS
    # =========================================================================
    os.environ['REWARD_C2'] = '1.0'  
    
    # --- SWEEP 1: TIER A (BASELOAD) ONLY ---
    tier_a_sweep = [0,5,10,15,20,25,30,35,40,45, 50]
    results_stage1 = []
    
    for a_mw in tier_a_sweep:
        t_a_ts = np.full(N_life, a_mw)
        t_b_daily_target_ts = np.zeros(N_life)
        t_b2_ts_empty = np.zeros(N_life)
        total_load_for_ems = t_a_ts.copy()
        total_load_for_ems[0] = MAX_IT_CAPACITY_MW 
        
        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_target_ts, tier_b2_profile=t_b2_ts_empty, load_profile_ts=total_load_for_ems, battery_deg=False
        )
        out = hpp.evaluate(*fixed_design)
        prob = hpp.prob
        
        rel_time_a = 1.0 - (np.sum(prob.get_val('ems.Unserved_A') > 1e-3) / N_life)
        curt = np.sum(prob.get_val('ems.hpp_curt_t')) / 1000.0
        c2 = np.sum(prob.get_val('ems.Served_C2')) / 1000.0
        try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_d = out[3]

        results_stage1.append({"Tier_A_MW": a_mw, "Tier_B_MW": 0.0, "Tier_B2_MW": 0.0, "Max_IT_MW": MAX_IT_CAPACITY_MW, "LCOE": lcoe_d, "Reliability_Time_A": rel_time_a * 100, "Reliability_Deadline_B": 100.0, "Reliability_Deadline_B2": 100.0, "Curtailment_GWh": curt, "C2_GWh": c2})

    
    # --- SWEEP 2: TIER B1 (DAILY BATCH) ONLY ---
    SAFE_BASELOAD_MW = 0.0  # ENSURING ZERO BASELOAD
    tier_b_sweep = tier_a_sweep
    results_stage2 = []

    for b_mw in tier_b_sweep:
        t_a_ts = np.full(N_life, SAFE_BASELOAD_MW)
        t_b_hourly_ts = np.full(N_life, b_mw)
        t_b_daily_target_ts = np.full(N_life, b_mw * 24.0) 
        t_b2_ts_empty = np.zeros(N_life)
        total_load_for_ems = t_a_ts + t_b_hourly_ts
        total_load_for_ems[0] = MAX_IT_CAPACITY_MW
        
        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_target_ts, tier_b2_profile=t_b2_ts_empty, load_profile_ts=total_load_for_ems, battery_deg=False
        )
        out = hpp.evaluate(*fixed_design)
        prob = hpp.prob
        
        rel_time_a = 1.0 - (np.sum(prob.get_val('ems.Unserved_A') > 1e-3) / N_life)
        shortfall_b_ts = prob.get_val('ems.Shortfall_B')
        rel_deadline_b = 1.0 - (np.sum(np.sum(shortfall_b_ts.reshape(-1, 24), axis=1) > 1e-3) / (N_life/24))
        curt = np.sum(prob.get_val('ems.hpp_curt_t')) / 1000.0
        c2 = np.sum(prob.get_val('ems.Served_C2')) / 1000.0
        try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_d = out[3]

        results_stage2.append({"Tier_A_MW": SAFE_BASELOAD_MW, "Tier_B_MW": b_mw, "Tier_B2_MW": 0.0, "Max_IT_MW": MAX_IT_CAPACITY_MW, "LCOE": lcoe_d, "Reliability_Time_A": rel_time_a * 100, "Reliability_Deadline_B": rel_deadline_b * 100, "Reliability_Deadline_B2": 100.0, "Curtailment_GWh": curt, "C2_GWh": c2})

    # --- SWEEP 3: TIER B2 (WEEKLY BATCH) ONLY ---
    SAFE_B1_MW = 0.0  # ENSURING ZERO B1 LOAD
    tier_b2_sweep = tier_a_sweep
    results_stage2b = []

    print(f"\n--- SWEEP 3: Tier B2 Only (Base A: {SAFE_BASELOAD_MW} MW, Base B1: {SAFE_B1_MW} MW) ---")

    for b2_mw in tier_b2_sweep:
        t_a_ts = np.full(N_life, SAFE_BASELOAD_MW)
        t_b_hourly_ts = np.full(N_life, SAFE_B1_MW)
        t_b_daily_target_ts = np.full(N_life, SAFE_B1_MW * 24.0)
        
        t_b2_hourly_ts = np.full(N_life, b2_mw)
        t_b2_weekly_target_ts = np.full(N_life, b2_mw * 168.0) 
        
        total_load_for_ems = t_a_ts + t_b_hourly_ts + t_b2_hourly_ts
        total_load_for_ems[0] = MAX_IT_CAPACITY_MW
        
        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_target_ts, 
            tier_b2_profile=t_b2_weekly_target_ts, 
            load_profile_ts=total_load_for_ems, battery_deg=False
        )
        out = hpp.evaluate(*fixed_design)
        prob = hpp.prob
        
        rel_time_a = 1.0 - (np.sum(prob.get_val('ems.Unserved_A') > 1e-3) / N_life)
        shortfall_b_ts = prob.get_val('ems.Shortfall_B')
        rel_deadline_b = 1.0 - (np.sum(np.sum(shortfall_b_ts.reshape(-1, 24), axis=1) > 1e-3) / (N_life/24))
        shortfall_b2_ts = prob.get_val('ems.Shortfall_B2')
        n_weeks = N_life // 168
        b2_weeks_matrix = shortfall_b2_ts[:n_weeks * 168].reshape(-1, 168)
        rel_deadline_b2 = 1.0 - (np.sum(np.sum(b2_weeks_matrix, axis=1) > 1e-3) / n_weeks)
        
        curt = np.sum(prob.get_val('ems.hpp_curt_t')) / 1000.0
        c2 = np.sum(prob.get_val('ems.Served_C2')) / 1000.0
        try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_d = out[3]

        results_stage2b.append({
            "Tier_A_MW": SAFE_BASELOAD_MW, "Tier_B_MW": SAFE_B1_MW, "Tier_B2_MW": b2_mw, 
            "Max_IT_MW": MAX_IT_CAPACITY_MW, "LCOE": lcoe_d, "Reliability_Time_A": rel_time_a * 100, 
            "Reliability_Deadline_B": rel_deadline_b * 100, "Reliability_Deadline_B2": rel_deadline_b2 * 100, 
            "Curtailment_GWh": curt, "C2_GWh": c2
        })
        
    df_stage1 = pd.DataFrame(results_stage1)
    df_stage2 = pd.DataFrame(results_stage2)
    df_stage2b = pd.DataFrame(results_stage2b)

    print_stacked_services_summary(df_stage1, "EVALUATION 1: TIER A (BASELOAD) ONLY")
    print_stacked_services_summary(df_stage2, "EVALUATION 2: TIER B1 (24h FLEX) ONLY")
    print_stacked_services_summary(df_stage2b, "EVALUATION 3: TIER B2 (168h FLEX) ONLY")

    print("\n--- Generating Independent Reliability Plot (.svg format) ---")
    # df3 (Opportunistic) is irrelevant here since C2 doesn't have an SLA limit.
    generate_thesis_plots(df_stage1, df_stage2, df_stage2b, None, 0, 0, 0)

if __name__ == "__main__":
    run_firm_yield_analysis()