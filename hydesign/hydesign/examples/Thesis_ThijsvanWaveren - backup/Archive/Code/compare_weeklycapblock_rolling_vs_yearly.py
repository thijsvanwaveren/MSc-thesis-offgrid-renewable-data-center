# -*- coding: utf-8 -*-
"""
Evaluate Dynamic Capacity Blocks (1-Year Analysis)
Yearly vs. Rolling Horizon with Dropped Hours Analysis
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

# Make sure this points to your correct assembly file!
from hydesign.assembly.assembly_rolling_horizon_thijs_10_3_26 import hpp_model_constant_output_offgrid as hpp_model

def configure_parameters(thesis_dir):
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    sim_pars['G_MW'] = 0
    os.environ['REWARD_C2'] = '1.0'  # Injects penalty into EMS to KILL Tier C2
    sim_pars['battery_charge_efficiency'] = float(np.sqrt(0.86))
    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_offgrid_sweep_temp.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    return temp_fn

def calculate_annual_metrics(unserved_a_1yr, demanded_a_1yr):
    """Calculates Energy and Temporal Reliability strictly for a single year (8760 hours)."""
    # 1. Energy Reliability
    total_demanded_mwh = np.sum(demanded_a_1yr)
    total_unserved_mwh = np.sum(unserved_a_1yr)
    total_served_mwh = total_demanded_mwh - total_unserved_mwh
    
    energy_rel = 100.0 * (1.0 - (total_unserved_mwh / total_demanded_mwh))
    
    # 2. Temporal Reliability
    hours_failed = np.sum(unserved_a_1yr > 1e-3)
    temporal_rel = 100.0 * (1.0 - (hours_failed / 8760.0))
    
    return {
        "demanded_gwh": total_demanded_mwh / 1000.0,
        "unserved_gwh": total_unserved_mwh / 1000.0,
        "energy_rel": energy_rel,
        "temporal_rel": temporal_rel,
        "hours_failed": hours_failed
    }

def plot_and_print_dropped_hours(unserved_yearly, unserved_rolling):
    """Prints the exact hours dropped and generates a comparative timeline plot."""
    # --- 1. Console Printout ---
    idx_y = np.where(unserved_yearly > 1e-3)[0]
    idx_r = np.where(unserved_rolling > 1e-3)[0]

    print("\n" + "="*80)
    print(" EXACT HOURS DROPPED: YEARLY vs. ROLLING ".center(80))
    print("="*80)

    print("YEARLY (Perfect Foresight) Failed Hours:")
    if len(idx_y) == 0:
        print("  None")
    else:
        for h in idx_y:
            print(f"  Hour {h:4d} (Week {h//168 + 1:2d}) : {unserved_yearly[h]:.2f} MW dropped")

    print("\nROLLING (Limited Foresight) Failed Hours:")
    if len(idx_r) == 0:
        print("  None")
    else:
        for h in idx_r:
            print(f"  Hour {h:4d} (Week {h//168 + 1:2d}) : {unserved_rolling[h]:.2f} MW dropped")
    print("="*80 + "\n")

    # --- 2. Timeline Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle('Timeline of Unserved Energy: Yearly vs. Rolling Horizon', fontsize=14, fontweight='bold')

    # Making the bars slightly wider (width=10) so they are visible on an 8760 scale
    ax1.bar(np.arange(8760), unserved_yearly, color='#d62728', width=15, label='Yearly Drops')
    ax1.set_title(f'Yearly Solver (Perfect Foresight) - {len(idx_y)} Hours Failed', fontsize=11)
    ax1.set_ylabel('Unserved MW')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.bar(np.arange(8760), unserved_rolling, color='#ff7f0e', width=15, label='Rolling Drops')
    ax2.set_title(f'Rolling Horizon (Limited Foresight) - {len(idx_r)} Hours Failed', fontsize=11)
    ax2.set_ylabel('Unserved MW')
    ax2.set_xlabel('Hour of the Year')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'Thesis_Plot_Dropped_Hours_Timeline.svg')
    fig.savefig(plot_fn, format='svg', dpi=300)
    print(f"✅ Dropped hours timeline plot saved to: {plot_fn}")
    plt.show()


def run_comparative_evaluation():
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    MAX_IT_CAPACITY_MW = 500.0 
    N_life_1yr = 8760
    N_life_total = 25 * 8760
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    os.environ['REWARD_C2'] = '1.0'  # Disable C2

    # =========================================================================
    # 1. LOAD THE CSV
    # =========================================================================
    csv_fn = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\Results\Weekly\annual simulation weekly result\Weekly_Capacity_Blocks_yearlysimulation.csv"
    
    try:
        df_blocks = pd.read_csv(csv_fn)
        weekly_firm_mw = df_blocks['Total_Firm_Load_MW'].values
        print(f"✅ Loaded {len(weekly_firm_mw)} weekly capacity blocks from CSV.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find CSV at {csv_fn}")
        sys.exit(1)

    # =========================================================================
    # 2. BUILD THE PROFILES
    # =========================================================================
    # 1-Year Array (For metric calculation)
    dynamic_tier_a_1yr = np.zeros(N_life_1yr)
    for w in range(52):
        dynamic_tier_a_1yr[w*168 : (w+1)*168] = weekly_firm_mw[w]
    dynamic_tier_a_1yr[52*168:] = weekly_firm_mw[-1]
    
    # 25-Year Arrays (To satisfy OpenMDAO shape requirements)
    dynamic_tier_a_25yr = np.tile(dynamic_tier_a_1yr, 25)
    t_zero_25yr = np.zeros(N_life_total)
    
    total_load_for_ems = dynamic_tier_a_25yr.copy()
    total_load_for_ems[0] = MAX_IT_CAPACITY_MW

    # =========================================================================
    # 3. RUN YEARLY SIMULATION
    # =========================================================================
    print(f"\n--- Running YEARLY Evaluation (Perfect Foresight) ---")
    hpp_yearly = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=dynamic_tier_a_25yr, tier_b_profile=t_zero_25yr, tier_b2_profile=t_zero_25yr, 
        load_profile_ts=total_load_for_ems, battery_deg=False,
        run_mode='yearly' 
    )
    hpp_yearly.evaluate(*fixed_design)
    
    # =========================================================================
    # EXTRACT ENERGY AND CURTAILMENT FOR BAR CHART (YEAR 1: 8760 Hours)
    # =========================================================================
    print("\n--- BAR CHART METRICS (Average MW over 1 Year) ---")
    
    # Extract raw arrays from OpenMDAO for the first year
    curtailed_mw_ts = hpp_yearly.prob.get_val('ems.hpp_curt_t')[:8760]
    unserved_a_ts = hpp_yearly.prob.get_val('ems.Unserved_A')[:8760]
    served_c2_ts = hpp_yearly.prob.get_val('ems.Served_C2')[:8760]
    
    # 1. Tier A (Served)
    # Requested load minus the unserved load
    served_a_ts = dynamic_tier_a_1yr - unserved_a_ts
    avg_tier_a_mw = np.mean(served_a_ts)
    print(f"Tier A (Served)       : {avg_tier_a_mw:>8.2f} MW Average")

    # 2. Tier C2 (Opportunistic Served)
    avg_tier_c2_mw = np.mean(served_c2_ts)
    print(f"Tier C2 (Served)      : {avg_tier_c2_mw:>8.2f} MW Average")

    # 3. Curtailed Power (Wasted)
    avg_curtailed_mw = np.mean(curtailed_mw_ts)
    total_curtailed_gwh = np.sum(curtailed_mw_ts) / 1000.0
    print(f"Curtailed Power       : {avg_curtailed_mw:>8.2f} MW Average ({total_curtailed_gwh:.2f} GWh total)")
    print("-" * 50)
    
    # Extract only the first 8760 hours!
    unserved_a_yearly_1yr = hpp_yearly.prob.get_val('ems.Unserved_A')[:8760]
    res_yearly = calculate_annual_metrics(unserved_a_yearly_1yr, dynamic_tier_a_1yr)

    # =========================================================================
    # 4. RUN ROLLING HORIZON SIMULATION
    # =========================================================================
    print(f"\n--- Running ROLLING HORIZON Evaluation (Limited Foresight) ---")
    hpp_rolling = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=dynamic_tier_a_25yr, tier_b_profile=t_zero_25yr, tier_b2_profile=t_zero_25yr, 
        load_profile_ts=total_load_for_ems, battery_deg=False,
        run_mode='rolling' 
    )
    hpp_rolling.evaluate(*fixed_design)
    
    # Extract only the first 8760 hours!
    unserved_a_rolling_1yr = hpp_rolling.prob.get_val('ems.Unserved_A')[:8760]
    res_rolling = calculate_annual_metrics(unserved_a_rolling_1yr, dynamic_tier_a_1yr)

    # =========================================================================
    # 5. PRINT COMPARISON
    # =========================================================================
    print("\n" + "="*80)
    print(" 1-YEAR BASELINE RELIABILITY COMPARISON: YEARLY vs. ROLLING HORIZON ".center(80))
    print("="*80)
    
    print(f"{'Metric':<25} | {'YEARLY (Perfect Foresight)':<25} | {'ROLLING (Limited Foresight)':<25}")
    print("-" * 80)
    
    print(f"{'Energy Demanded':<25} | {res_yearly['demanded_gwh']:>18.2f} GWh | {res_rolling['demanded_gwh']:>18.2f} GWh")
    print(f"{'Energy Unserved':<25} | {res_yearly['unserved_gwh']:>18.4f} GWh | {res_rolling['unserved_gwh']:>18.4f} GWh")
    print("-" * 80)
    print(f"{'Energy Reliability':<25} | {res_yearly['energy_rel']:>18.4f} %   | {res_rolling['energy_rel']:>18.4f} %")
    print(f"{'Temporal Reliability':<25} | {res_yearly['temporal_rel']:>18.4f} %   | {res_rolling['temporal_rel']:>18.4f} %")
    print(f"{'Hours Failed (out of 8760)':<25} | {res_yearly['hours_failed']:>18} h   | {res_rolling['hours_failed']:>18} h")
    print("="*80 + "\n")
    
    # Call the new visualization function
    plot_and_print_dropped_hours(unserved_a_yearly_1yr, unserved_a_rolling_1yr)

if __name__ == "__main__":
    run_comparative_evaluation()