# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:27:10 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Perfect Foresight vs. Rolling Horizon Comparison
Focus: Tier A (Baseload) Time-Based SLA vs Energy Yield
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- DIRECTORY SETUP ---

current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
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


def plot_solver_comparison(df):
    """Generates a dual-panel comparison plot between Solvers"""
    df_perfect = df[df['Run_Mode'] == 'perfect_foresight']
    df_rolling = df[df['Run_Mode'] == 'rolling']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Yearly Optimization vs. Rolling Horizon', fontsize=16, fontweight='bold')

    # --- PANEL 1: Time-Based Reliability ---
    ax1.plot(df_perfect['Tier_A_MW'], df_perfect['Reliability_Time_A'], 
            marker='o', linestyle='--', color='gray', linewidth=5, markersize=12, label='Perfect Foresight')
    ax1.plot(df_rolling['Tier_A_MW'], df_rolling['Reliability_Time_A'], 
            marker='s', linestyle='-', color='#1f77b4', linewidth=2, markersize=6, label='Rolling Horizon (7 days)')
    # ax1.axhline(99.9, color='red', linestyle=':', alpha=0.5, label='99.9% Target')
    ax1.set_ylabel('Time Reliability (% of full-load hours)', fontsize=12)
    ax1.set_xlabel('Tier A Capacity (MW)', fontsize=12)
    ax1.set_title('Time-Based Reliability')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # --- PANEL 2: Energy Yield (The Mathematical Truth) ---
    ax2.plot(df_perfect['Tier_A_MW'], df_perfect['Reliability_Energy_A'], 
            marker='o', linestyle='--', color='gray', linewidth=5, markersize=12, label='Perfect Foresight')
    ax2.plot(df_rolling['Tier_A_MW'], df_rolling['Reliability_Energy_A'], 
            marker='s', linestyle='-', color='#2ca02c', linewidth=2, markersize=6, label='Rolling Horizon (7 days)')
    # ax2.axhline(99.9, color='red', linestyle=':', alpha=0.5, label='99.9% Target')
    ax2.set_ylabel('Energy Reliability (% of demanded MWh delivered)', fontsize=12)
    ax2.set_xlabel('Tier A Capacity (MW)', fontsize=12)
    ax2.set_title('Energy-Based Reliability')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'Thesis_Plot_Solver_Comparison.svg')
    fig.savefig(plot_fn, format='svg', dpi=300)
    print(f"\n✅ Plot successfully saved to: {plot_fn}")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*85)
    print(" SOLVER COMPARISON: PERFECT FORESIGHT VS ROLLING HORIZON ".center(85, '='))
    print("="*85 + "\n")

    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    N_life = 25 * 8760
    MAX_IT_CAPACITY_MW = 100.0
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    os.environ['REWARD_C2'] = '1.0'  # Disable C2 
    
    # Evaluate across the "Cliff"
    tier_a_sweep = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    run_modes = ['perfect_foresight', 'rolling']
    all_results = []

    # Loop over the capacities first so we can compare the solvers side-by-side instantly
    for a_mw in tier_a_sweep:
        print(f"\n--- EVALUATING TIER A = {a_mw} MW ---")
        
        mw_results = {}
        
        for mode in run_modes:
            print(f"  > Running {mode.replace('_', ' ').title()}... ", end="", flush=True)
            
            t_a_ts = np.full(N_life, a_mw)
            t_zero = np.zeros(N_life)
            total_load = t_a_ts.copy()
            total_load[0] = MAX_IT_CAPACITY_MW 
            
            hpp = hpp_model(
                latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
                num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
                tier_a_profile=t_a_ts, tier_b_profile=t_zero, tier_b2_profile=t_zero, load_profile_ts=total_load, 
                battery_deg=False, run_mode=mode
            )
            hpp.evaluate(*fixed_design)
            
            prob = hpp.prob
            
            # Metrics
            rel_a_time = 100.0 * (1.0 - (np.sum(prob.get_val('ems.Unserved_A') > 1e-3) / N_life))
            demanded_a = a_mw * N_life
            unserved_a_total = np.sum(prob.get_val('ems.Unserved_A'))
            rel_a_energy = ((demanded_a - unserved_a_total) / demanded_a) * 100.0
            
            print(f"Done.")
            
            mw_results[mode] = {'time': rel_a_time, 'energy': rel_a_energy}
            all_results.append({
                'Run_Mode': mode,
                'Tier_A_MW': a_mw,
                'Reliability_Time_A': rel_a_time,
                'Reliability_Energy_A': rel_a_energy
            })

        # Instant side-by-side comparison for this MW capacity
        print(f"  [RESULTS AT {a_mw} MW]")
        print(f"    Perfect Foresight -> Time Rel: {mw_results['perfect_foresight']['time']:>6.2f}% | Energy Yield: {mw_results['perfect_foresight']['energy']:>6.2f}%")
        print(f"    Rolling Horizon   -> Time Rel: {mw_results['rolling']['time']:>6.2f}% | Energy Yield: {mw_results['rolling']['energy']:>6.2f}%")

    # --- PRINT THE FINAL SUMMARY TABLE ---
    print("\n" + "="*85)
    print(" FINAL SUMMARY TABLE ".center(85, '='))
    print("="*85)
    print(f"| {'Tier A':<8} | {'PERFECT FORESIGHT':<32} | {'ROLLING HORIZON':<32} |")
    print(f"| {'(MW)':<8} | {'Time Yield':<15} | {'Energy Yield':<14} | {'Time Yield':<15} | {'Energy Yield':<14} |")
    print("-" * 85)
    for a_mw in tier_a_sweep:
        pf = [r for r in all_results if r['Tier_A_MW'] == a_mw and r['Run_Mode'] == 'perfect_foresight'][0]
        rh = [r for r in all_results if r['Tier_A_MW'] == a_mw and r['Run_Mode'] == 'rolling'][0]
        
        pf_time = f"{pf['Reliability_Time_A']:.2f}%"
        pf_nrg  = f"{pf['Reliability_Energy_A']:.2f}%"
        rh_time = f"{rh['Reliability_Time_A']:.2f}%"
        rh_nrg  = f"{rh['Reliability_Energy_A']:.2f}%"
        
        print(f"| {a_mw:<8.1f} | {pf_time:<15} | {pf_nrg:<14} | {rh_time:<15} | {rh_nrg:<14} |")
    print("=" * 85 + "\n")

    df_results = pd.DataFrame(all_results)
    plot_solver_comparison(df_results)
    
    
    
  