# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:55:25 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Rolling Horizon Capacity Block Stacking
Phase 0: Dynamic Weekly Sizing
Phase 1-4: Stacking Flexible Loads on dynamic blocks
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- HYDESIGN IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model

def generate_thesis_plots(df1, df2, df2b, df3, base_a, base_b1, base_b2):
    """
    Generates SLA/LCOE sweep plots for each stage, plus a Master Progression Bar Chart 
    showing the incremental value of flexibility. Saves to SVG.
    """
    LIFETIME_HOURS = 25 * 8760
    
    # =========================================================
    # PLOT 1: STAGE 1 SWEEP (Reliability & LCOEd only)
    # =========================================================
    fig1, (ax1_1, ax1_2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig1.suptitle('Stage 1: Baseload Sizing (Tier A)', fontsize=14, fontweight='bold')

    ax1_1.plot(df1['Tier_A_MW'], df1['Reliability_Time_A'], marker='o', color='#1f77b4', linewidth=2)
    ax1_1.axhline(99.9, color='red', linestyle='--', alpha=0.5, label='99.9% Target')
    ax1_1.set_ylabel('Reliability - Full Load Factor (%)')
    ax1_1.set_title('Temporal Reliability')
    ax1_1.grid(True, linestyle='--', alpha=0.7)
    ax1_1.legend()

    ax1_2.plot(df1['Tier_A_MW'], df1['LCOE'], marker='d', color='#9467bd', linewidth=2)
    ax1_2.set_ylabel('LCOED (€/MWh)')
    ax1_2.set_xlabel('Tier A Baseload (MW)')
    ax1_2.set_title('Levelized Cost of Energy Delivered')
    ax1_2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig1.savefig(os.path.join(current_dir, 'Thesis_Plot_1_Stage1.svg'), format='svg', dpi=300)

    # =========================================================
    # PLOT 2: STAGE 2 SWEEP (+ Tier B1)
    # =========================================================
    fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig2.suptitle(f'Stage 2: Stacking Tier B1 on {base_a} MW Baseload', fontsize=14, fontweight='bold')

    ax2_1.plot(df2['Tier_B_MW'], df2['Reliability_Time_A'], marker='o', linestyle='--', color='#1f77b4', label='Tier A (Baseload)')
    ax2_1.plot(df2['Tier_B_MW'], df2['Reliability_Deadline_B'], marker='s', color='#ff7f0e', linewidth=2, label='Tier B1 (24h SLA)')
    ax2_1.axhline(99.9, color='red', linestyle='--', alpha=0.5, label='99.9% Target')
    ax2_1.set_ylabel('Reliability (%)')
    ax2_1.set_title('System Reliability by Tier')
    ax2_1.legend()
    ax2_1.grid(True, linestyle='--', alpha=0.7)

    ax2_2.plot(df2['Tier_B_MW'], df2['LCOE'], marker='d', color='#9467bd', linewidth=2)
    ax2_2.set_ylabel('LCOED (€/MWh)')
    ax2_2.set_xlabel('Tier B1 Daily Flexible Load (MW)')
    ax2_2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig2.savefig(os.path.join(current_dir, 'Thesis_Plot_2_Stage2.svg'), format='svg', dpi=300)

    # =========================================================
    # PLOT 3: STAGE 2.5 SWEEP (+ Tier B2)
    # =========================================================
    fig2b, (ax2b_1, ax2b_2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig2b.suptitle(f'Stage 3: Stacking Tier B2 on {base_a} MW (A) and {base_b1} MW (B1)', fontsize=14, fontweight='bold')

    ax2b_1.plot(df2b['Tier_B2_MW'], df2b['Reliability_Time_A'], marker='o', linestyle='--', color='#1f77b4', label='Tier A (Baseload)')
    #ax2b_1.plot(df2b['Tier_B2_MW'], df2b['Reliability_Deadline_B'], marker='^', linestyle='--', color='#ff7f0e', label='Tier B1 (24h SLA)')
    ax2b_1.plot(df2b['Tier_B2_MW'], df2b['Reliability_Deadline_B2'], marker='s', color='#8c564b', linewidth=2, label='Tier B2 (168h SLA)')
    ax2b_1.axhline(99.9, color='red', linestyle='--', alpha=0.5, label='99.9% Target')
    ax2b_1.set_ylabel('Reliability (%)')
    ax2b_1.set_title('System Reliability by Tier')
    ax2b_1.legend()
    ax2b_1.grid(True, linestyle='--', alpha=0.7)

    ax2b_2.plot(df2b['Tier_B2_MW'], df2b['LCOE'], marker='d', color='#9467bd', linewidth=2)
    ax2b_2.set_ylabel('LCOED (€/MWh)')
    ax2b_2.set_xlabel('Tier B2 Weekly Flexible Load (MW)')
    ax2b_2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig2b.savefig(os.path.join(current_dir, 'Thesis_Plot_3_Stage2_5.svg'), format='svg', dpi=300)

    # =========================================================
    # PLOT 4: THE MASTER PROGRESSION BAR CHART
    # =========================================================
    # Extract the optimal point from each stage for the comparison
    try:
        p1 = df1[df1['Tier_A_MW'] == base_a].iloc[0]
        p2 = df2[df2['Tier_B_MW'] == base_b1].iloc[0]
        p2b = df2b[df2b['Tier_B2_MW'] == base_b2].iloc[0]
        p3 = df3[df3['Tier_B2_MW'] == base_b2].iloc[0]
    except IndexError:
        print("Warning: Progression plot could not find exact base MW match in DataFrames.")
        return

    stages = ['Stage 1\n(Baseload Only)', 'Stage 2\n(+24h Flex)', 'Stage 3\n(+168h Flex)', 'Stage 4\n(+Opportunistic)']
    
    # Calculate delivered energy for the specific points
    A_deliv = [(p.Tier_A_MW * LIFETIME_HOURS / 1000) * (p.Reliability_Time_A / 100) for p in [p1, p2, p2b, p3]]
    B1_deliv = [0, (p2.Tier_B_MW * LIFETIME_HOURS / 1000) * (p2.Reliability_Deadline_B / 100), 
                (p2b.Tier_B_MW * LIFETIME_HOURS / 1000) * (p2b.Reliability_Deadline_B / 100), 
                (p3.Tier_B_MW * LIFETIME_HOURS / 1000) * (p3.Reliability_Deadline_B / 100)]
    B2_deliv = [0, 0, (p2b.Tier_B2_MW * LIFETIME_HOURS / 1000) * (p2b.Reliability_Deadline_B2 / 100), 
                (p3.Tier_B2_MW * LIFETIME_HOURS / 1000) * (p3.Reliability_Deadline_B2 / 100)]
    C_deliv = [0, 0, 0, p3.C2_GWh]
    Curtailment = [p1.Curtailment_GWh, p2.Curtailment_GWh, p2b.Curtailment_GWh, p3.Curtailment_GWh]
    LCOE = [p1.LCOE, p2.LCOE, p2b.LCOE, p3.LCOE]

    fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle('The Value of Workload Stacking', fontsize=16, fontweight='bold')

    x = np.arange(len(stages))
    width = 0.35

    # Panel A: Stacked Delivered Energy vs Curtailment
    # Stack the delivered energy
    b_A = ax4_1.bar(x - width/2, A_deliv, width, label='Tier A Delivered', color='#1f77b4', edgecolor='black')
    b_B1 = ax4_1.bar(x - width/2, B1_deliv, width, bottom=A_deliv, label='Tier B1 Delivered', color='#ff7f0e', edgecolor='black')
    bottom_B2 = np.add(A_deliv, B1_deliv)
    b_B2 = ax4_1.bar(x - width/2, B2_deliv, width, bottom=bottom_B2, label='Tier B2 Delivered', color='#8c564b', edgecolor='black')
    bottom_C = np.add(bottom_B2, B2_deliv)
    b_C = ax4_1.bar(x - width/2, C_deliv, width, bottom=bottom_C, label='Tier C Delivered', color='#17becf', edgecolor='black')

    # Plot curtailment next to it
    b_Curt = ax4_1.bar(x + width/2, Curtailment, width, label='Curtailed Energy', color='#d62728', hatch='//', edgecolor='black')

    ax4_1.set_ylabel('Total Energy (GWh) over 25 Years')
    ax4_1.set_title('Energy Recovery')
    ax4_1.set_xticks(x)
    ax4_1.set_xticklabels(stages)
    ax4_1.legend()
    ax4_1.grid(axis='y', linestyle='--', alpha=0.7)

    # Panel B: LCOE Reduction
    bars_lcoe = ax4_2.bar(x, LCOE, width=0.5, color='#9467bd', edgecolor='black')
    ax4_2.set_ylabel('LCOED (€/MWh)')
    ax4_2.set_title('Levelized Cost of Electricity Delivered')
    ax4_2.set_xticks(x)
    ax4_2.set_xticklabels(stages)
    ax4_2.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of LCOE bars
    for bar in bars_lcoe:
        yval = bar.get_height()
        ax4_2.text(bar.get_x() + bar.get_width()/2, yval + 2, f'€{yval:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig4.savefig(os.path.join(current_dir, 'Thesis_Plot_4_MasterProgression.svg'), format='svg', dpi=300)
    plt.show()
    
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

def calculate_weekly_firm_capacities(ex_site, weather_fn, sim_pars_fn, fixed_design):
    """
    PHASE 0: Loads the pre-calculated weekly firm capacities from the CSV.
    Returns a 219000-hour array (25 years) with stepped weekly targets.
    """
    print("\n" + "="*80)
    print(" PHASE 0: LOADING PRE-CALCULATED WEEKLY FIRM CAPACITY BLOCKS ".center(80))
    print("="*80)
    
    N_life_1yr = 8760
    
    # 1. Navigate from "scripts" up one level, then down into your specific Results folder
    csv_fn = os.path.abspath(os.path.join(
        current_dir, '..', 'Results', 'Weekly', 'annual simulation weekly result', 'Weekly_Capacity_Blocks_yearlysimulation.csv'
    ))
    
    try:
        df_blocks = pd.read_csv(csv_fn)
        weekly_firm_mw = df_blocks['Total_Firm_Load_MW'].values
        print(f"✅ Successfully loaded {len(weekly_firm_mw)} weeks of data from:\n   {csv_fn}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find the CSV file at:\n   {csv_fn}")
        sys.exit(1)

    annual_baseload = np.min(weekly_firm_mw)
    print(f"=> The Firm Annual Baseload (The Bottleneck) is: {annual_baseload} MW")
    
    # 2. Construct the 1-year stepped array
    dynamic_tier_a_1yr = np.zeros(N_life_1yr)
    for w in range(52):
        dynamic_tier_a_1yr[w*168 : (w+1)*168] = weekly_firm_mw[w]
    
    # Fill the remaining 24 hours of the year with the final week's value
    dynamic_tier_a_1yr[52*168:] = weekly_firm_mw[-1]
    
    # 3. Expand to 25 years
    dynamic_tier_a_ts = np.tile(dynamic_tier_a_1yr, 25)
    
    return dynamic_tier_a_ts, annual_baseload
def run_firm_yield_analysis():
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    MAX_IT_CAPACITY_MW = 500.0 # Unconstrained to isolate weather
    N_life = 25 * 8760
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    os.environ['REWARD_C2'] = '1.0'  # Disable C2

    # =========================================================================
    # PHASE 0: BUILD THE CAPACITY BLOCKS
    # =========================================================================
    dynamic_tier_a_profile, SAFE_BASELOAD_MW = calculate_weekly_firm_capacities(
        ex_site, weather_fn, sim_pars_fn, fixed_design
    )

    # =========================================================================
    # STAGE 2: SWEEP TIER B1 ON TOP OF THE DYNAMIC CAPACITY BLOCKS
    # =========================================================================
    print(f"\n--- STAGE 2: Sweeping Tier B1 on top of Dynamic Capacity Blocks ---")
    tier_b_sweep = [0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    results_stage2 = []

    for b_mw in tier_b_sweep:
        t_b_hourly_ts = np.full(N_life, b_mw)
        t_b_daily_target_ts = np.full(N_life, b_mw * 24.0) 
        
        total_load_for_ems = dynamic_tier_a_profile + t_b_hourly_ts
        total_load_for_ems[0] = MAX_IT_CAPACITY_MW
        
        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=dynamic_tier_a_profile, # <--- Passing the dynamic array!
            tier_b_profile=t_b_daily_target_ts, 
            tier_b2_profile=np.zeros(N_life), 
            load_profile_ts=total_load_for_ems, battery_deg=False,
            run_mode='rolling' # <--- Enforce Rolling Horizon!
        )
        out = hpp.evaluate(*fixed_design)
        prob = hpp.prob
        
        rel_time_a = 1.0 - (np.sum(prob.get_val('ems.Unserved_A') > 1e-3) / N_life)
        shortfall_b_ts = prob.get_val('ems.Shortfall_B')
        rel_deadline_b = 1.0 - (np.sum(np.sum(shortfall_b_ts.reshape(-1, 24), axis=1) > 1e-3) / (N_life/24))
        curt = np.sum(prob.get_val('ems.hpp_curt_t')) / 1000.0
        
        try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_d = out[3]

        results_stage2.append({"Tier_A_MW": SAFE_BASELOAD_MW, "Tier_B_MW": b_mw, "Tier_B2_MW": 0.0, "Max_IT_MW": MAX_IT_CAPACITY_MW, "LCOE": lcoe_d, "Reliability_Time_A": rel_time_a * 100, "Reliability_Deadline_B": rel_deadline_b * 100, "Reliability_Deadline_B2": 100.0, "Curtailment_GWh": curt})

    # =========================================================================
    # STAGE 3: SWEEP TIER B2 ON TOP OF BLOCKS AND B1
    # =========================================================================
    SAFE_B1_MW = 6.0 # Look at your output from Stage 2 and set this!
    
    tier_b2_sweep = [0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    results_stage2b = []

    print(f"\n--- STAGE 3: Sweeping Tier B2 (Base A: Dynamic, Base B1: {SAFE_B1_MW} MW) ---")

    for b2_mw in tier_b2_sweep:
        t_b_hourly_ts = np.full(N_life, SAFE_B1_MW)
        t_b_daily_target_ts = np.full(N_life, SAFE_B1_MW * 24.0)
        
        t_b2_hourly_ts = np.full(N_life, b2_mw)
        t_b2_weekly_target_ts = np.full(N_life, b2_mw * 168.0) 
        
        total_load_for_ems = dynamic_tier_a_profile + t_b_hourly_ts + t_b2_hourly_ts
        total_load_for_ems[0] = MAX_IT_CAPACITY_MW
        
        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=dynamic_tier_a_profile, # <--- Passing the dynamic array!
            tier_b_profile=t_b_daily_target_ts, 
            tier_b2_profile=t_b2_weekly_target_ts, 
            load_profile_ts=total_load_for_ems, battery_deg=False,
            run_mode='rolling' # <--- Enforce Rolling Horizon!
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

        try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_d = out[3]

        results_stage2b.append({
            "Tier_A_MW": SAFE_BASELOAD_MW, "Tier_B_MW": SAFE_B1_MW, "Tier_B2_MW": b2_mw, 
            "Max_IT_MW": MAX_IT_CAPACITY_MW, "LCOE": lcoe_d, "Reliability_Time_A": rel_time_a * 100, 
            "Reliability_Deadline_B": rel_deadline_b * 100, "Reliability_Deadline_B2": rel_deadline_b2 * 100, 
            "Curtailment_GWh": curt
        })
        
    df_stage2 = pd.DataFrame(results_stage2)
    df_stage2b = pd.DataFrame(results_stage2b)
    
    # We can use your existing print and plot functions here!
    print_stacked_services_summary(df_stage2, f"STAGE 2: LAYERED FLEXIBLE LOAD (Base A: Dynamic)")
    print_stacked_services_summary(df_stage2b, f"STAGE 3: WEEKLY FLEXIBLE BATCH (Base A: Dynamic | Base B1: {SAFE_B1_MW} MW)")

if __name__ == "__main__":
    run_firm_yield_analysis()