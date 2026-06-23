# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:04:28 2026

@author: thijs
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

from hydesign.assembly.hpp_assembly_offgrid_thijs_2_2_26 import hpp_model_constant_output_offgrid as hpp_model

def generate_thesis_plots(df1, df2, df3, base_mw):
    """
    Generates three high-quality, 3-panel plots for thesis visualization.
    Calculates delivered energy dynamically based on reliability scores.
    """
    # Helper constants
    LIFETIME_HOURS = 25 * 8760
    
    
    
    # ---------------------------------------------------------
    # PLOT 1: STAGE 1 - BASELOAD CEILING (TIER A SWEEP)
    # ---------------------------------------------------------
    fig1, (ax1_1, ax1_2, ax1_3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig1.suptitle('Stage 1: Effect of Tier A Baseload on System Performance', fontsize=16, fontweight='bold', y=0.95)

    # Panel 1: Reliability
    ax1_1.plot(df1['Tier_A_MW'], df1['Reliability_Time_A'], marker='o', color='#1f77b4', linewidth=2, markersize=8)
    ax1_1.set_ylabel('Reliability (%)', fontsize=12)
    ax1_1.set_title('Tier A Reliability (Time-based)', fontsize=12)
    ax1_1.grid(True, linestyle='--', alpha=0.7)

    # Panel 2: Energy Balance
    # Calculate Delivered Energy = Demand * Reliability
    demand_A_GWh = (df1['Tier_A_MW'] * LIFETIME_HOURS) / 1000
    delivered_A_GWh = demand_A_GWh * (df1['Reliability_Time_A'] / 100)
    
    ax1_2.plot(df1['Tier_A_MW'], delivered_A_GWh, marker='o', color='#2ca02c', linewidth=2, label='Delivered Energy (Tier A)')
    ax1_2.plot(df1['Tier_A_MW'], df1['Curtailment_GWh'], marker='s', color='#d62728', linewidth=2, label='Curtailed Energy')
    ax1_2.set_ylabel('Energy (GWh)', fontsize=12)
    ax1_2.set_title('25-Year Energy Balance', fontsize=12)
    ax1_2.legend()
    ax1_2.grid(True, linestyle='--', alpha=0.7)

    # Panel 3: LCOE
    ax1_3.plot(df1['Tier_A_MW'], df1['LCOE'], marker='d', color='#9467bd', linewidth=2, markersize=8)
    ax1_3.set_ylabel('LCOE (€/MWh)', fontsize=12)
    ax1_3.set_xlabel('Tier A Baseload (MW)', fontsize=12)
    ax1_3.set_title('Levelized Cost of Energy Delivered', fontsize=12)
    ax1_3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(current_dir, 'Thesis_Plot_1_Stage1.png'), dpi=300)

    # ---------------------------------------------------------
    # PLOT 2: STAGE 2 - LAYERING TIER B (FIXED TIER A)
    # ---------------------------------------------------------
    fig2, (ax2_1, ax2_2, ax2_3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig2.suptitle(f'Stage 2: Layering Tier B on {base_mw} MW Baseload', fontsize=16, fontweight='bold', y=0.95)

    # Panel 1: Reliability
    ax2_1.plot(df2['Tier_B_MW'], df2['Reliability_Time_A'], marker='o', linestyle='--', color='#1f77b4', label='Tier A (Time)')
    ax2_1.plot(df2['Tier_B_MW'], df2['Reliability_Deadline_B'], marker='s', color='#ff7f0e', linewidth=2, markersize=8, label='Tier B (24h Deadline)')
    ax2_1.set_ylabel('Reliability (%)', fontsize=12)
    ax2_1.set_title('System Reliability by Tier', fontsize=12)
    ax2_1.legend()
    ax2_1.grid(True, linestyle='--', alpha=0.7)

    # Panel 2: Energy Balance
    demand_B_GWh_2 = (df2['Tier_B_MW'] * LIFETIME_HOURS) / 1000
    delivered_B_GWh_2 = demand_B_GWh_2 * (df2['Reliability_Deadline_B'] / 100)
    delivered_A_GWh_2 = ((df2['Tier_A_MW'] * LIFETIME_HOURS) / 1000) * (df2['Reliability_Time_A'] / 100)
    total_delivered_2 = delivered_A_GWh_2 + delivered_B_GWh_2

    ax2_2.plot(df2['Tier_B_MW'], total_delivered_2, marker='o', color='#2ca02c', linewidth=2, label='Total Delivered (A + B)')
    ax2_2.plot(df2['Tier_B_MW'], df2['Curtailment_GWh'], marker='s', color='#d62728', linewidth=2, label='Curtailed Energy')
    ax2_2.set_ylabel('Energy (GWh)', fontsize=12)
    ax2_2.set_title('25-Year Energy Balance', fontsize=12)
    ax2_2.legend()
    ax2_2.grid(True, linestyle='--', alpha=0.7)

    # Panel 3: LCOE
    ax2_3.plot(df2['Tier_B_MW'], df2['LCOE'], marker='d', color='#9467bd', linewidth=2, markersize=8)
    ax2_3.set_ylabel('LCOE (€/MWh)', fontsize=12)
    ax2_3.set_xlabel('Tier B Flexible Load (MW)', fontsize=12)
    ax2_3.set_title('Levelized Cost of Energy Delivered', fontsize=12)
    ax2_3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(os.path.join(current_dir, 'Thesis_Plot_2_Stage2.png'), dpi=300)

    # ---------------------------------------------------------
    # PLOT 3: STAGE 3 - THE C2 SPONGE EFFECT (COMPARISON)
    # ---------------------------------------------------------
    fig3, (ax3_1, ax3_2, ax3_3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig3.suptitle(f'Stage 3: Impact of Tier C2 Sponge (vs. Stage 2)', fontsize=16, fontweight='bold', y=0.95)

    # Panel 1: Reliability (Usually unchanged, showing to prove C2 doesn't harm A or B)
    ax3_1.plot(df3['Tier_B_MW'], df3['Reliability_Deadline_B'], marker='s', color='#ff7f0e', linewidth=2, label='Tier B (With C2)')
    ax3_1.plot(df2['Tier_B_MW'], df2['Reliability_Deadline_B'], marker='x', linestyle=':', color='black', alpha=0.6, label='Tier B (No C2 Baseline)')
    ax3_1.set_ylabel('Reliability (%)', fontsize=12)
    ax3_1.set_title('Tier B Deadline Reliability (Proving C2 is harmless)', fontsize=12)
    ax3_1.legend()
    ax3_1.grid(True, linestyle='--', alpha=0.7)

    # Panel 2: Energy Balance (The real magic)
    demand_B_GWh_3 = (df3['Tier_B_MW'] * LIFETIME_HOURS) / 1000
    delivered_B_GWh_3 = demand_B_GWh_3 * (df3['Reliability_Deadline_B'] / 100)
    delivered_A_GWh_3 = ((df3['Tier_A_MW'] * LIFETIME_HOURS) / 1000) * (df3['Reliability_Time_A'] / 100)
    
    total_delivered_3 = delivered_A_GWh_3 + delivered_B_GWh_3 + df3['C2_GWh']

    ax3_2.plot(df3['Tier_B_MW'], total_delivered_2, marker='o', linestyle=':', color='#2ca02c', alpha=0.5, label='Delivered (No C2)')
    ax3_2.plot(df3['Tier_B_MW'], total_delivered_3, marker='o', color='#2ca02c', linewidth=2, label='Delivered (With C2)')
    
    ax3_2.plot(df3['Tier_B_MW'], df2['Curtailment_GWh'], marker='s', linestyle=':', color='#d62728', alpha=0.5, label='Curtailed (No C2)')
    ax3_2.plot(df3['Tier_B_MW'], df3['Curtailment_GWh'], marker='s', color='#d62728', linewidth=2, label='Curtailed (With C2)')
    
    # Add a subtle bar chart in the background to show the exact volume C2 absorbed
    ax3_2.bar(df3['Tier_B_MW'], df3['C2_GWh'], color='#17becf', alpha=0.3, width=0.6, label='Volume Absorbed by C2')

    ax3_2.set_ylabel('Energy (GWh)', fontsize=12)
    ax3_2.set_title('How C2 Absorbs Curtailment into Useful Delivery', fontsize=12)
    ax3_2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3_2.grid(True, linestyle='--', alpha=0.7)

    # Panel 3: LCOE Comparison
    ax3_3.plot(df2['Tier_B_MW'], df2['LCOE'], marker='d', linestyle=':', color='black', alpha=0.6, label='LCOE (No C2)')
    ax3_3.plot(df3['Tier_B_MW'], df3['LCOE'], marker='d', color='#9467bd', linewidth=2, markersize=8, label='LCOE (With C2 Profit)')
    ax3_3.set_ylabel('LCOE (€/MWh)', fontsize=12)
    ax3_3.set_xlabel('Tier B Flexible Load (MW)', fontsize=12)
    ax3_3.set_title('LCOE Reduction due to C2 Sponge', fontsize=12)
    ax3_3.legend()
    ax3_3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig3.savefig(os.path.join(current_dir, 'Thesis_Plot_3_Stage3.png'), bbox_inches='tight', dpi=300)

    plt.show()

def plot_true_weekly_dynamic_capacity(stage1_unserved_dict, max_it_mw):
    """
    Evaluates the unserved energy from each Stage 1 sweep run to find the 
    true maximum 100% reliable baseload for each week.
    """
    HOURS_PER_WEEK = 168
    weeks = np.arange(1, 53)
    
    # Sort the tested capacities from lowest to highest
    tested_capacities = sorted(list(stage1_unserved_dict.keys()))
    
    weekly_firm_a = []
    
    for w in range(52):
        start = w * HOURS_PER_WEEK
        end = start + HOURS_PER_WEEK
        
        max_survived_capacity = 0.0
        
        # Test each capacity to see if it survived this specific week
        for cap in tested_capacities:
            unserved_this_week = stage1_unserved_dict[cap][start:end]
            
            # If the max unserved energy this week is effectively zero, it survived!
            if np.max(unserved_this_week) < 1e-3:
                max_survived_capacity = cap
            else:
                # As soon as a capacity fails, we know higher ones will fail too
                break
                
        weekly_firm_a.append(max_survived_capacity)

    static_annual_baseload = min(weekly_firm_a)
    
    # Calculate the "Lost Opportunity" area
    dynamic_bonus_mwh = sum([(w - static_annual_baseload)*168 for w in weekly_firm_a])

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    bars = ax1.bar(weeks, weekly_firm_a, color='#1f77b4', alpha=0.8, edgecolor='black', 
                   label='True Dynamic Weekly Guarantee (100% Reliable)')
    
    ax1.axhline(y=static_annual_baseload, color='#d62728', linestyle='--', linewidth=3, 
                label=f'Static Annual Constraint ({static_annual_baseload:.1f} MW)')
    
    ax1.axhline(y=static_annual_baseload+dynamic_bonus_mwh/8760, color='#d62728', linestyle='--', linewidth=3, 
                label=f'Average Weekly Constant Load {static_annual_baseload+dynamic_bonus_mwh/8760:.1f} MW)')
    
    ax1.fill_between(weeks, static_annual_baseload, weekly_firm_a, step='mid', 
                     color='#2ca02c', alpha=0.2, hatch='//', 
                     label=f'Unlocked Dynamic Value ({dynamic_bonus_mwh/1000:.1f} GWh/yr)')
    
    ax1.set_title('The Cost of Perfect Predictability: Dynamic vs. Static SLAs', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Week of the Year', fontsize=12)
    ax1.set_ylabel('Guaranteed 100% Reliable Power (MW)', fontsize=12)
    ax1.set_xlim(0, 53)
    ax1.set_ylim(0, max(tested_capacities) + 2)
    
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', fontsize=11, framealpha=1.0)
    
    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'True_Weekly_Dynamic_Capacity.png')
    plt.savefig(plot_fn, dpi=300)
    print(f"\nSaved true weekly dynamic capacity plot to: {plot_fn}")
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
    print(f"\n{'='*110}")
    print(f"{stage_name:^110}")
    print(f"{'='*110}")
    print(f"{'Tier A (MW)':<12} | {'Tier B (MW)':<12} | {'Max IT (MW)':<12} | {'LCOEd (€)':<10} | {'Rel(A-Time)':<12} | {'Rel(B-Dead)':<12} | {'Curt(GWh)':<10} | {'C2(GWh)':<10}")
    print("-" * 110)
    for _, row in df_res.iterrows():
        print(f"{row['Tier_A_MW']:<12.1f} | {row['Tier_B_MW']:<12.1f} | {row['Max_IT_MW']:<12.1f} | "
              f"{row['LCOE']:<10.2f} | {row['Reliability_Time_A']:<11.2f}% | "
              f"{row['Reliability_Deadline_B']:<11.2f}% | "
              f"{row['Curtailment_GWh']:<9.2f} | {row['C2_GWh']:<9.2f}")
    print("="*110 + "\n")

def run_firm_yield_analysis():
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25,8, 10] 
    MAX_IT_CAPACITY_MW = 30.0  # Plentiful headroom for catching up!
    N_life = 25 * 8760
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    print(f"--- Starting Full Stack Analysis (Max IT: {MAX_IT_CAPACITY_MW} MW) ---")

    
    # =========================================================================
    # STAGES 1 & 2: NO C2 SPONGE
    # =========================================================================
    os.environ['REWARD_C2'] = '1.0'  # Injects penalty into EMS to KILL Tier C2
    
    # --- STAGE 1 ---
    tier_a_sweep = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22, 24, 26, 28, 30 ,32, 34, 36, 38,40]
    results_stage1 = []
    
    # NEW: Dictionary to store the hourly unserved profile for each tested capacity
    stage1_unserved_dict = {} 
    
    for a_mw in tier_a_sweep:
        t_a_ts = np.full(N_life, a_mw)
        t_b_daily_target_ts = np.zeros(N_life)
        total_load_for_ems = t_a_ts.copy()
        total_load_for_ems[0] = MAX_IT_CAPACITY_MW 
        
        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_target_ts, load_profile_ts=total_load_for_ems, battery_deg=False
        )
        out = hpp.evaluate(*fixed_design)
        prob = hpp.prob
        
        # Grab the hourly unserved array and save it
        unserved_hourly = prob.get_val('ems.Unserved_A')[:8736] # slice to 52 weeks
        stage1_unserved_dict[a_mw] = unserved_hourly
        
        rel_time_a = 1.0 - (np.sum(unserved_hourly > 1e-3) / len(unserved_hourly))
        curt = np.sum(prob.get_val('ems.hpp_curt_t')) / 1000.0
        c2 = np.sum(prob.get_val('ems.Served_C2')) / 1000.0
        try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_d = out[3]

        results_stage1.append({"Tier_A_MW": a_mw, "Tier_B_MW": 0.0, "Max_IT_MW": MAX_IT_CAPACITY_MW, "LCOE": lcoe_d, "Reliability_Time_A": rel_time_a * 100, "Reliability_Deadline_B": 100.0, "Curtailment_GWh": curt, "C2_GWh": c2})

    # --- PLOT THE TRUE WEEKLY DYNAMIC CAPACITY ---
    print("\n--- Generating True Dynamic Weekly Capacity Plot ---")
    plot_true_weekly_dynamic_capacity(stage1_unserved_dict, MAX_IT_CAPACITY_MW)
    
# =========================================================================
    # BUILD THE DYNAMIC CAPACITY BLOCK PROFILE
    # =========================================================================
    print("\n--- Generating Dynamic Weekly Capacity Blocks ---")
    
    dynamic_tier_a_ts = np.zeros(N_life)
    HOURS_PER_WEEK = 168
    tested_capacities = sorted(list(stage1_unserved_dict.keys()))
    
    # Find the max surviving capacity for each week and build the staircase profile
    for w in range(52):
        start = w * HOURS_PER_WEEK
        end = start + HOURS_PER_WEEK
        
        max_survived_capacity = 0.0
        for cap in tested_capacities:
            unserved_this_week = stage1_unserved_dict[cap][start:end]
            if np.max(unserved_this_week) < 1e-3:
                max_survived_capacity = cap
            else:
                break
                
        # Fill this specific week with the maximum reliable Capacity Block
        dynamic_tier_a_ts[start:end] = max_survived_capacity
        
    # Catch the remaining 24 hours of the 8760 year
    dynamic_tier_a_ts[52 * HOURS_PER_WEEK:] = max_survived_capacity


    # --- STAGE 2 ---
    SAFE_BASELOAD_MW = 8.0 
    tier_b_sweep = [0,1,2.0, 3, 4, 5.0,6, 7,  8.0, 11.0, 14.0, 17.0]
    results_stage2 = []

# Inside your Stage 2 / Stage 3 loops...
    for b_mw in tier_b_sweep:
        # USE THE DYNAMIC STAIRCASE INSTEAD OF A FLAT LINE!
        t_a_ts = dynamic_tier_a_ts.copy() 
        
        t_b_hourly_ts = np.full(N_life, b_mw)
        t_b_daily_target_ts = np.full(N_life, b_mw * 24.0) 
        
        # Max IT Capacity must be a ceiling. We don't want A + B to accidentally exceed IT hardware.
        total_load_for_ems = t_a_ts + t_b_hourly_ts
        total_load_for_ems = np.minimum(total_load_for_ems, MAX_IT_CAPACITY_MW)
        total_load_for_ems[0] = MAX_IT_CAPACITY_MW
        
        hpp = hpp_model(
            # ... identical hpp setup ...
            tier_a_profile=t_a_ts, # Passes the dynamic weekly blocks to the EMS!
            tier_b_profile=t_b_daily_target_ts, 
            load_profile_ts=total_load_for_ems, 
            battery_deg=False
        )
        # ... rest of your evaluation logic ...
        out = hpp.evaluate(*fixed_design)
        prob = hpp.prob
        
        rel_time_a = 1.0 - (np.sum(prob.get_val('ems.Unserved_A') > 1e-3) / N_life)
        shortfall_b_ts = prob.get_val('ems.Shortfall_B')
        rel_deadline_b = 1.0 - (np.sum(np.sum(shortfall_b_ts.reshape(-1, 24), axis=1) > 1e-3) / (N_life/24))
        curt = np.sum(prob.get_val('ems.hpp_curt_t')) / 1000.0
        c2 = np.sum(prob.get_val('ems.Served_C2')) / 1000.0
        try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_d = out[3]

        results_stage2.append({"Tier_A_MW": SAFE_BASELOAD_MW, "Tier_B_MW": b_mw, "Max_IT_MW": MAX_IT_CAPACITY_MW, "LCOE": lcoe_d, "Reliability_Time_A": rel_time_a * 100, "Reliability_Deadline_B": rel_deadline_b * 100, "Curtailment_GWh": curt, "C2_GWh": c2})

   

    # =========================================================================
    # STAGE 3: THE OPPORTUNISTIC SPONGE
    # =========================================================================
    os.environ['REWARD_C2'] = '-0.5'  # Turn C2 Reward back on!
    results_stage3 = []

# Inside your Stage 2 / Stage 3 loops...
    for b_mw in tier_b_sweep:
        # USE THE DYNAMIC STAIRCASE INSTEAD OF A FLAT LINE!
        t_a_ts = dynamic_tier_a_ts.copy() 
        
        t_b_hourly_ts = np.full(N_life, b_mw)
        t_b_daily_target_ts = np.full(N_life, b_mw * 24.0) 
        
        # Max IT Capacity must be a ceiling. We don't want A + B to accidentally exceed IT hardware.
        total_load_for_ems = t_a_ts + t_b_hourly_ts
        total_load_for_ems = np.minimum(total_load_for_ems, MAX_IT_CAPACITY_MW)
        total_load_for_ems[0] = MAX_IT_CAPACITY_MW
        
        hpp = hpp_model(
            # ... identical hpp setup ...
            tier_a_profile=t_a_ts, # Passes the dynamic weekly blocks to the EMS!
            tier_b_profile=t_b_daily_target_ts, 
            load_profile_ts=total_load_for_ems, 
            battery_deg=False
        )
        # ... rest of your evaluation logic ...
        out = hpp.evaluate(*fixed_design)
        prob = hpp.prob
        
        # --- Sanity checks: BATTERY CYCLING & EFFICIENCY LOSS CHECK ---

        # Continue...

        rel_time_a = 1.0 - (np.sum(prob.get_val('ems.Unserved_A') > 1e-3) / N_life)
        shortfall_b_ts = prob.get_val('ems.Shortfall_B')
        rel_deadline_b = 1.0 - (np.sum(np.sum(shortfall_b_ts.reshape(-1, 24), axis=1) > 1e-3) / (N_life/24))
        curt = np.sum(prob.get_val('ems.hpp_curt_t')) / 1000.0
        c2 = np.sum(prob.get_val('ems.Served_C2')) / 1000.0
        try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
        except Exception: lcoe_d = out[3]

        results_stage3.append({"Tier_A_MW": SAFE_BASELOAD_MW, "Tier_B_MW": b_mw, "Max_IT_MW": MAX_IT_CAPACITY_MW, "LCOE": lcoe_d, "Reliability_Time_A": rel_time_a * 100, "Reliability_Deadline_B": rel_deadline_b * 100, "Curtailment_GWh": curt, "C2_GWh": c2})
    
    df_stage1 = pd.DataFrame(results_stage1)
    df_stage2 = pd.DataFrame(results_stage2)
    df_stage3 = pd.DataFrame(results_stage3)
    
    print_stacked_services_summary(df_stage1, "STAGE 1: BASELOAD CEILING")
    print_stacked_services_summary(df_stage2, f"STAGE 2: LAYERED FLEXIBLE LOAD (Base: {SAFE_BASELOAD_MW} MW)")
    print_stacked_services_summary(df_stage3, f"STAGE 3: FULL STACK W/ C2 SPONGE (Base: {SAFE_BASELOAD_MW} MW)")

    # CALL THE PLOTTING FUNCTION
    print("\n--- Generating Thesis Plots ---")
    generate_thesis_plots(df_stage1, df_stage2, df_stage3, SAFE_BASELOAD_MW)
    
    # =========================================================================
    # STAGE 4: DYNAMIC WEEKLY SLA ANALYSIS (THE OVERLOAD RUN)
    # =========================================================================
    print("\n--- STAGE 4: Running Overload Scenario for Weekly Dynamic Analysis ---")
    
    OVERLOAD_A_MW = 24.0  # Ask for a massive baseload to see where the system physically breaks
    OVERLOAD_B_MW = 0.0   # Turn off B to isolate the pure baseload physical limit
    OVERLOAD_IT_MW = 24.0
    
    t_a_ts_overload = np.full(N_life, OVERLOAD_A_MW)
    t_b_ts_overload = np.zeros(N_life)
    total_load_overload = t_a_ts_overload.copy()
    total_load_overload[0] = OVERLOAD_IT_MW
    
    hpp_overload = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_a_ts_overload, tier_b_profile=t_b_ts_overload, load_profile_ts=total_load_overload, battery_deg=False
    )
    
    print(f"Evaluating overloaded {OVERLOAD_A_MW} MW system...")
    hpp_overload.evaluate(*fixed_design)
    
if __name__ == "__main__":
    run_firm_yield_analysis()