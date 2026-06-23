# -*- coding: utf-8 -*-
"""
Tier A vs Tier B1: 2D Parameter Sweep & Pareto Frontier
(Interactive Reliability Target & Smart Pruning - No LCOE Plot)
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

def plot_pareto_frontier(df_results, target_rel):
    """
    Filters for viable combinations and plots the Tier A vs Tier B1 tradeoff frontier.
    """
    viable_df = df_results[
        (df_results['Reliability_A'] >= target_rel) & 
        (df_results['Reliability_B1'] >= target_rel)
    ].copy()

    if viable_df.empty:
        print(f"\n❌ No combinations achieved the target reliability of {target_rel}%.")
        return

    frontier_idx = viable_df.groupby('Tier_A_MW')['Tier_B1_MW'].idxmax()
    frontier_df = viable_df.loc[frontier_idx].sort_values(by='Tier_A_MW')

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Capacity Trade-off Tier A vs B1 (Reliability $\geq$ {target_rel}%)', fontsize=15, fontweight='bold')

    # Plot all viable points as a standard scatter plot
    ax.scatter(
        viable_df['Tier_A_MW'], 
        viable_df['Tier_B1_MW'], 
        color='#1f77b4', # Standard blue
        s=100, 
        alpha=0.6,
        edgecolors='black',
        label='Viable Combinations'
    )

    
    # Formatting
    ax.set_xlabel('Tier A (Static Baseload) - MW', fontsize=12)
    ax.set_ylabel('Tier B1 (24h Flexible Load) - MW', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, f'Thesis_Plot_Tradeoff_Frontier_{target_rel}pct.svg')
    fig.savefig(plot_fn, format='svg', dpi=300)
    print(f"\n✅ Trade-off Plot successfully saved to: {plot_fn}")
    plt.show()

def run_tradeoff_sweep(target_rel):
    # Standard parameters
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    MAX_IT_CAPACITY_MW = 50.0 
    N_life = 25 * 8760
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    os.environ['REWARD_C2'] = '1.0'  # Disable C2

    # =========================================================================
    # DEFINE THE SWEEP RANGES
    # =========================================================================
    tier_a_array = np.arange(0.0, 9.0, 1.0)  
    tier_b1_array = np.arange(0.0, 16.0, 1.0) 
    
    results = []
    
    # We track the maximum successful B1 for the previous A value.
    # We initialize it high so the first A loop runs the full B1 sweep.
    max_b1_ceiling = max(tier_b1_array) 

    print("\n" + "="*70)
    print(f" STARTING SMART 2D SWEEP (Pruning Unfeasible Paths) ".center(70))
    print(f" Target Reliability: {target_rel}% ".center(70))
    print("="*70)

    for a_mw in tier_a_array:
        current_max_b1_for_this_a = -1 
        
        for b1_mw in tier_b1_array:
            
            # SMART PRUNING RULE 1: The Frontier Ceiling
            if b1_mw > max_b1_ceiling:
                print(f"   ⏭️ Skipping (A={a_mw}, B1={b1_mw}): Exceeds mathematical ceiling of {max_b1_ceiling} MW.")
                continue

            print(f"Testing -> Tier A: {a_mw:04.1f} MW | Tier B1: {b1_mw:04.1f} MW...", end="")
            
            t_a_ts = np.full(N_life, a_mw)
            t_b1_hourly_ts = np.full(N_life, b1_mw)
            t_b1_daily_target_ts = np.full(N_life, b1_mw * 24.0) 
            t_b2_ts_empty = np.zeros(N_life)
            
            total_load_for_ems = t_a_ts + t_b1_hourly_ts
            total_load_for_ems[0] = MAX_IT_CAPACITY_MW
            
            hpp = hpp_model(
                latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
                num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
                tier_a_profile=t_a_ts, tier_b_profile=t_b1_daily_target_ts, tier_b2_profile=t_b2_ts_empty, load_profile_ts=total_load_for_ems, battery_deg=False
            )
            out = hpp.evaluate(*fixed_design)
            prob = hpp.prob
            
            rel_a = 100.0 * (1.0 - (np.sum(prob.get_val('ems.Unserved_A') > 1e-3) / N_life))
            shortfall_b_ts = prob.get_val('ems.Shortfall_B')
            rel_b1 = 100.0 * (1.0 - (np.sum(np.sum(shortfall_b_ts.reshape(-1, 24), axis=1) > 1e-3) / (N_life/24)))
            
            try: lcoe_d = prob.get_val('finance.LCOE_delivered')[0]
            except Exception: lcoe_d = out[3]

            is_viable = (rel_a >= target_rel and rel_b1 >= target_rel)

            if is_viable:
                print(f" ✅ Viable (LCOE: €{lcoe_d:.1f})")
                current_max_b1_for_this_a = b1_mw 
            else:
                print(f" ❌ Failed (Rel A: {rel_a:.1f}%, Rel B1: {rel_b1:.1f}%)")
                
            results.append({
                "Tier_A_MW": a_mw,
                "Tier_B1_MW": b1_mw,
                "Reliability_A": rel_a,
                "Reliability_B1": rel_b1,
                "LCOE": lcoe_d if is_viable else np.nan 
            })

            # SMART PRUNING RULE 2: The Too Heavy Rule
            if not is_viable:
                print(f"   ⏭️ Skipping remaining B1 values for A={a_mw} (Too heavy).")
                break 

        if current_max_b1_for_this_a != -1:
            max_b1_ceiling = current_max_b1_for_this_a
            
        # SMART PRUNING RULE 3: The Baseload Death Spiral
        if current_max_b1_for_this_a == -1:
            print(f"\n🛑 FATAL: A={a_mw} failed even with 0 MW of B1. Ending all sweeps.")
            break 

    df_results = pd.DataFrame(results)
    csv_fn = os.path.join(current_dir, f'Tradeoff_Sweep_Results_{target_rel}pct.csv')
    df_results.to_csv(csv_fn, index=False)
    print(f"\n✅ Raw data saved to {csv_fn}")

    plot_pareto_frontier(df_results, target_rel=target_rel)

if __name__ == "__main__":
    print("\n=== Data Center Hybrid Power Plant Trade-off Sweep ===")
    try:
        user_input = input("Enter the target reliability percentage (e.g., 99.9, 95.0): ")
        target_reliability = float(user_input.strip())
    except ValueError:
        print("Invalid input detected. Defaulting to strict 99.9% reliability.")
        target_reliability = 99.9
        
    run_tradeoff_sweep(target_rel=target_reliability)