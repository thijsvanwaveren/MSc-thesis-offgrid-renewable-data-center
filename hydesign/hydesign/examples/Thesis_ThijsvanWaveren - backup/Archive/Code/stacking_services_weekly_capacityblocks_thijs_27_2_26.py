# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 12:30:00 2026

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

from hydesign.assembly.hpp_assembly_weekly_capacityblocks_thijs_27_2_26 import hpp_model_constant_output_offgrid as hpp_model

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
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

def evaluate_stack_scenario(name, t_a_ts, t_b_hourly_mw, c2_reward, ex_site, weather_fn, sim_pars_fn, fixed_design, MAX_IT, N_life):
    """
    A clean wrapper to evaluate a specific Workload Tetris configuration.
    """
    os.environ['REWARD_C2'] = str(c2_reward)
    
    # Format Tier B inputs
    t_b_hourly_ts = np.full(N_life, t_b_hourly_mw)
    t_b_daily_target_ts = np.full(N_life, t_b_hourly_mw * 24.0) 
    
    # Establish the IT hardware ceiling for the EMS solver
    total_load_for_ems = t_a_ts + t_b_hourly_ts
    total_load_for_ems = np.minimum(total_load_for_ems, MAX_IT)
    total_load_for_ems[0] = MAX_IT
    
    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_a_ts, tier_b_profile=t_b_daily_target_ts, load_profile_ts=total_load_for_ems, battery_deg=False
    )
    
    out = hpp.evaluate(*fixed_design)
    prob = hpp.prob
    
    # Extract Metrics
    unserved_a = prob.get_val('ems.Unserved_A')
    rel_time_a = 1.0 - (np.sum(unserved_a > 1e-3) / N_life)
    
    shortfall_b_ts = prob.get_val('ems.Shortfall_B')
    rel_deadline_b = 1.0 - (np.sum(np.sum(shortfall_b_ts.reshape(-1, 24), axis=1) > 1e-3) / (N_life/24))
    
    curt_gwh = np.sum(prob.get_val('ems.hpp_curt_t')) / 1000.0
    c2_gwh = np.sum(prob.get_val('ems.Served_C2')) / 1000.0
    c2_ts = prob.get_val('ems.Served_C2')
    
    # Delivered Energy Math
    dynamic_A_delivered_GWh = ((t_a_ts * rel_time_a).sum()) / 1000.0
    dynamic_B_delivered_GWh = ((t_b_hourly_mw * N_life) / 1000.0) * rel_deadline_b
    total_delivered_GWh = dynamic_A_delivered_GWh + dynamic_B_delivered_GWh + c2_gwh
    
    try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
    except Exception: lcoe_d = out[3]

    return {
        "Scenario": name,
        "Rel_A": rel_time_a * 100,
        "Rel_B": rel_deadline_b * 100 if t_b_hourly_mw > 0 else 0.0,
        "LCOE": lcoe_d,
        "Curt_GWh": curt_gwh,
        "C2_GWh": c2_gwh,
        "Total_Delivered_GWh": total_delivered_GWh,
        "C2_ts": c2_ts
    }

def plot_4_step_progression(results_df, batch_mw):
    """
    Plots the final results table to show exactly how stacking improves the business case.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"The Business Case for Dynamic SLAs (Tier B Added: {batch_mw:.1f} MW)", fontsize=16, fontweight='bold')

    x_labels = results_df["Scenario"]
    x = np.arange(len(x_labels))

    # Panel 1: Energy Evolution
    ax1.bar(x, results_df["Total_Delivered_GWh"], color='#2ca02c', label="Total Delivered Energy (GWh)")
    ax1.bar(x, results_df["Curt_GWh"], bottom=results_df["Total_Delivered_GWh"], color='#d62728', label="Wasted Curtailment (GWh)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=15, ha='right')
    ax1.set_ylabel("Energy (GWh)", fontsize=12)
    ax1.set_title("Energy Recovery via Stacking", fontsize=14)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Annotate Reliabilities on the Energy Bars
    for i, row in results_df.iterrows():
        text = f"Tier A Rel: {row['Rel_A']:.1f}%\nTier B Rel: {row['Rel_B']:.1f}%"
        ax1.annotate(text, (x[i], 1000), color='white', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 2: Financial Evolution
    ax2.plot(x, results_df["LCOE"], marker='d', markersize=10, color='#9467bd', linewidth=3)
    for i, txt in enumerate(results_df["LCOE"]):
        ax2.annotate(f"€{txt:.2f}", (x[i], results_df["LCOE"][i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', fontsize=11)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=15, ha='right')
    ax2.set_ylabel("Levelized Cost of Energy (€/MWh)", fontsize=12)
    ax2.set_title("LCOE Reduction via Stacking", fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'Thesis_4_Step_Stacking.png'), dpi=300)
    plt.show()

def plot_weekly_workload_stack(weekly_firm_a, core_mw, batch_mw, rel_a, rel_b, c2_ts):
    """
    Creates a 52-week stacked bar chart showing the exact MW composition of the Data Center.
    """
    weeks = np.arange(1, 53)
    
    # Process arrays for the 52 weeks
    core_array = np.full(52, core_mw)
    cb_array = np.array(weekly_firm_a) - core_mw
    
    # Average Tier B served (Approximated by Target MW for capacity visualization)
    batch_array = np.full(52, batch_mw)
    
    # Average C2 Power served per week
    c2_array = c2_ts[:8736].reshape(52, 168).mean(axis=1)

    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Stack the bars
    ax.bar(weeks, core_array, color='#1A5F7A', label=f'Yearly Baseload (Core: {core_mw:.1f} MW)')
    ax.bar(weeks, cb_array, bottom=core_array, color='#2298AB', label='Weekly Extra Baseload (Capacity Blocks)')
    ax.bar(weeks, batch_array, bottom=core_array+cb_array, color='#59C1CC', label=f'Daily Schedulable (Batch Target: {batch_mw:.1f} MW)')
    ax.bar(weeks, c2_array, bottom=core_array+cb_array+batch_array, color='#AEE2E9', label='Opportunistic Load (Tier C2 Avg)')

    # Formatting
    ax.set_title("Full Year Workload Stack: How the Data Center Operates Week-by-Week", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Week of the Year", fontsize=12)
    ax.set_ylabel("Average Power Consumed (MW)", fontsize=12)
    ax.set_xlim(0, 53)
    
    # Add an Annotation Box for Reliabilities
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    textstr = '\n'.join((
        r'$\bf{Final\ System\ Reliability}$',
        f'Tier A (Firm Uptime): {rel_a:.1f}%',
        f'Tier B (24h Deadline): {rel_b:.1f}%'
    ))
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.legend(loc='upper right', framealpha=1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'Thesis_Weekly_Workload_Stack.png'), dpi=300)
    plt.show()

# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_firm_yield_analysis():
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    MAX_IT_CAPACITY_MW = 35.0  
    BATCH_SIZE_MW = 4.0   # <--- Explicitly defining Tier B size here
    N_life = 25 * 8760
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    print(f"--- Starting Step-by-Step Stacking Analysis ---")
    print(f"Max IT Capacity: {MAX_IT_CAPACITY_MW} MW")
    print(f"Daily Batch Added: {BATCH_SIZE_MW} MW\n")

    # -------------------------------------------------------------------------
    # INITIALIZATION: Find the survival limits (The Sweep)
    # -------------------------------------------------------------------------
    print("Initializing: Sweeping capacities to find survival limits...")
    os.environ['REWARD_C2'] = '1.0' # Turn OFF C2
    
    tier_a_sweep = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22, 24, 26, 28, 30]
    stage1_unserved_dict = {} 
    
    for a_mw in tier_a_sweep:
        t_a_ts = np.full(N_life, a_mw)
        t_b_daily = np.zeros(N_life)
        total_load = t_a_ts.copy()
        total_load[0] = MAX_IT_CAPACITY_MW 
        
        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
            num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_ts, tier_b_profile=t_b_daily, load_profile_ts=total_load, battery_deg=False
        )
        hpp.evaluate(*fixed_design)
        stage1_unserved_dict[a_mw] = hpp.prob.get_val('ems.Unserved_A')[:8736] # 52 weeks

    # --- Build the Dynamic Profiles based on the sweep ---
    HOURS_PER_WEEK = 168
    dynamic_tier_a_ts = np.zeros(N_life)
    weekly_firm_a = []
    
    for w in range(52):
        start = w * HOURS_PER_WEEK
        end = start + HOURS_PER_WEEK
        max_surv = 0.0
        for cap in sorted(tier_a_sweep):
            if np.max(stage1_unserved_dict[cap][start:end]) < 1e-3:
                max_surv = cap
            else:
                break
        weekly_firm_a.append(max_surv)
        dynamic_tier_a_ts[start:end] = max_surv
        
    dynamic_tier_a_ts[52 * HOURS_PER_WEEK:] = max_surv
    
    TRUE_ANNUAL_CORE = min(weekly_firm_a)
    static_tier_a_ts = np.full(N_life, TRUE_ANNUAL_CORE)

    # -------------------------------------------------------------------------
    # THE 4-STEP ISOLATED STACKING ANALYSIS
    # -------------------------------------------------------------------------
    final_results = []

    print("\nExecuting Step 1: 100% Reliable Core (Yearly Baseload)...")
    res1 = evaluate_stack_scenario("1. Core Baseload", static_tier_a_ts, 0.0, 1.0, ex_site, weather_fn, sim_pars_fn, fixed_design, MAX_IT_CAPACITY_MW, N_life)
    final_results.append(res1)

    print("Executing Step 2: Unlocking Weekly Capacity Blocks...")
    res2 = evaluate_stack_scenario("2. + Capacity Blocks", dynamic_tier_a_ts, 0.0, 1.0, ex_site, weather_fn, sim_pars_fn, fixed_design, MAX_IT_CAPACITY_MW, N_life)
    final_results.append(res2)

    print(f"Executing Step 3: Stacking {BATCH_SIZE_MW} MW Daily Batch...")
    res3 = evaluate_stack_scenario("3. + Daily Batch", dynamic_tier_a_ts, BATCH_SIZE_MW, 1.0, ex_site, weather_fn, sim_pars_fn, fixed_design, MAX_IT_CAPACITY_MW, N_life)
    final_results.append(res3)

    print("Executing Step 4: Enabling Opportunistic C2 Sponge...")
    res4 = evaluate_stack_scenario("4. + C2 Sponge", dynamic_tier_a_ts, BATCH_SIZE_MW, -0.5, ex_site, weather_fn, sim_pars_fn, fixed_design, MAX_IT_CAPACITY_MW, N_life)
    final_results.append(res4)

    # -------------------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(final_results)
    
    print("\n" + "="*95)
    print(f"{'THE WORKLOAD TETRIS: 4-STEP VALUE STACKING (Tier B: ' + str(BATCH_SIZE_MW) + ' MW)':^95}")
    print("="*95)
    print(f"{'Stacking Step':<25} | {'Rel A':<8} | {'Rel B':<8} | {'LCOE (€/MWh)':<12} | {'Curt (GWh)':<10} | {'Total Deliv (GWh)':<12}")
    print("-" * 95)
    
    for _, row in df_results.iterrows():
        print(f"{row['Scenario']:<25} | {row['Rel_A']:<7.1f}% | {row['Rel_B']:<7.1f}% | "
              f"€{row['LCOE']:<11.2f} | {row['Curt_GWh']:<10.1f} | {row['Total_Delivered_GWh']:<12.1f}")
    print("="*95 + "\n")

    # Generate the requested plots
    plot_4_step_progression(df_results, BATCH_SIZE_MW)
    plot_weekly_workload_stack(
        weekly_firm_a=weekly_firm_a, 
        core_mw=TRUE_ANNUAL_CORE, 
        batch_mw=BATCH_SIZE_MW, 
        rel_a=res4['Rel_A'], 
        rel_b=res4['Rel_B'], 
        c2_ts=res4['C2_ts']
    )

if __name__ == "__main__":
    run_firm_yield_analysis()