# -*- coding: utf-8 -*-
"""
Iterative Solver: Maximum Tier B1 (Daily) and B2 (Weekly) capacity at 99.9% Reliability
Based on Deadline Completion Rates
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- HYDESIGN IMPORTS ---
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

def calculate_tier_a_reliability(unserved_a_1yr):
    """Tier A is continuous, so it is evaluated hour-by-hour (8760 hours)"""
    hours_failed = np.sum(unserved_a_1yr > 1e-1)
    return 100.0 * (1.0 - (hours_failed / 8760.0))

def calculate_b1_reliability(shortfall_b1_1yr):
    """Tier B1 has daily deadlines. Evaluates completion rate of 365 days."""
    days_failed = 0
    for d in range(365):
        # If there is any shortfall > 0 during this 24h period, the daily deadline was missed
        if np.any(shortfall_b1_1yr[d*24 : (d+1)*24] > 1e-1):
            days_failed += 1
    return 100.0 * (1.0 - (days_failed / 365.0))

def calculate_b2_reliability(shortfall_b2_1yr):
    """Tier B2 has weekly deadlines. Evaluates completion rate of 52 weeks."""
    weeks_failed = 0
    for w in range(52):
        # If there is any shortfall > 0 during this 168h period, the weekly deadline was missed
        if np.any(shortfall_b2_1yr[w*168 : (w+1)*168] > 1e-1):
            weeks_failed += 1
    return 100.0 * (1.0 - (weeks_failed / 52.0))

def run_iterative_search():
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    MAX_IT_CAPACITY_MW = 35.0 
    N_life_1yr = 8760
    N_life_total = 25 * 8760
    TARGET_REL = 99.9 # 99.9% Target
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    # Disable C2 Opportunistic Load entirely
    os.environ['REWARD_C2'] = '1.0'  

    # =========================================================================
    # 1. LOAD THE FIRM CAPACITY BLOCKS
    # =========================================================================
    csv_fn = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\Results\Weekly\annual simulation weekly result\Weekly_Capacity_Blocks_yearlysimulation.csv"
    
    try:
        df_blocks = pd.read_csv(csv_fn)
        weekly_firm_mw = df_blocks['Total_Firm_Load_MW'].values
    except FileNotFoundError:
        print(f"❌ Error: Could not find CSV at {csv_fn}")
        sys.exit(1)

    # Build the 25-year Tier A (Firm Load) profile
    dynamic_tier_a_1yr = np.zeros(N_life_1yr)
    for w in range(52):
        dynamic_tier_a_1yr[w*168 : (w+1)*168] = weekly_firm_mw[w]
    dynamic_tier_a_1yr[52*168:] = weekly_firm_mw[-1]
    dynamic_tier_a_25yr = 0.95*np.tile(dynamic_tier_a_1yr, 25) #slightly scaling the firm load to meet 99.9% reliability

    # Zero arrays for initial setup
    t_zero_25yr = np.zeros(N_life_total)

    # =========================================================================
    # 2. INITIALIZE THE OPENMDAO MODEL ONCE
    # =========================================================================
    print(f"\nInitializing OpenMDAO Model (This will only happen once)...")
    hpp_yearly = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=dynamic_tier_a_25yr, tier_b_profile=t_zero_25yr, tier_b2_profile=t_zero_25yr, 
        load_profile_ts=dynamic_tier_a_25yr, battery_deg=False,
        run_mode='yearly' 
    )

    # =========================================================================
    # 3. ITERATIVE SEARCH: TIER B1 (Daily Flexible)
    # =========================================================================
    print("\n" + "="*60)
    print(f" ITERATIVE SEARCH: TIER B1 (Target: >= {TARGET_REL}%) ".center(60))
    print("="*60)
    
    max_b1_mw = 0
    tier_b1_fixed_25yr = t_zero_25yr.copy()

    for b1_mw in range(1, 101): # Test from 1 MW up to 100 MW
        # Create continuous equivalent profile: MW * 24 hours
        tier_b1_25yr = np.full(N_life_total, b1_mw * 24.0)
        
        # Update EMS arrays
        total_load = dynamic_tier_a_25yr + tier_b1_25yr
        total_load[0] = MAX_IT_CAPACITY_MW
        
        hpp_yearly.prob.set_val('tier_b_profile', tier_b1_25yr)
        hpp_yearly.prob.set_val('load_profile_ts', total_load)
        
        hpp_yearly.evaluate(*fixed_design)
        
        unserved_a = hpp_yearly.prob.get_val('ems.Unserved_A')[:8760]
        shortfall_b1 = hpp_yearly.prob.get_val('ems.Shortfall_B')[:8760]
        
        rel_a = calculate_tier_a_reliability(unserved_a)
        rel_b1 = calculate_b1_reliability(shortfall_b1)
        
        if rel_a >= TARGET_REL and rel_b1 >= TARGET_REL:
            max_b1_mw = b1_mw
            tier_b1_fixed_25yr = tier_b1_25yr.copy()
            print(f"✅ B1 = {b1_mw:2d} MW  |  Tier A: {rel_a:.3f}%  |  Tier B1: {rel_b1:.3f}%")
        else:
            print(f"❌ B1 = {b1_mw:2d} MW  |  Tier A: {rel_a:.3f}%  |  Tier B1: {rel_b1:.3f}% (FAILED)")
            break 

    print(f"\n--> 🏆 Maximum Tier B1 Locked At: {max_b1_mw} MW")

    # =========================================================================
    # 4. ITERATIVE SEARCH: TIER B2 (Weekly Flexible)
    # =========================================================================
    print("\n" + "="*60)
    print(f" ITERATIVE SEARCH: TIER B2 (Target: >= {TARGET_REL}%) ".center(60))
    print("="*60)
    
    max_b2_mw = 0

    # Ensure successful B1 runs are locked in
    hpp_yearly.prob.set_val('tier_b_profile', tier_b1_fixed_25yr)

    for b2_mw in range(1, 101):
        # Create continuous equivalent profile: MW * 168 hours
        tier_b2_25yr = np.full(N_life_total, b2_mw * 168.0)
        
        # Update EMS arrays (A + B1 + B2)
        total_load = dynamic_tier_a_25yr + tier_b1_fixed_25yr + tier_b2_25yr
        total_load[0] = MAX_IT_CAPACITY_MW
        
        hpp_yearly.prob.set_val('tier_b2_profile', tier_b2_25yr)
        hpp_yearly.prob.set_val('load_profile_ts', total_load)
        
        hpp_yearly.evaluate(*fixed_design)
        
        unserved_a = hpp_yearly.prob.get_val('ems.Unserved_A')[:8760]
        shortfall_b1 = hpp_yearly.prob.get_val('ems.Shortfall_B')[:8760]
        shortfall_b2 = hpp_yearly.prob.get_val('ems.Shortfall_B2')[:8760]
        
        rel_a = calculate_tier_a_reliability(unserved_a)
        rel_b1 = calculate_b1_reliability(shortfall_b1)
        rel_b2 = calculate_b2_reliability(shortfall_b2)
        
        if rel_a >= TARGET_REL and rel_b1 >= TARGET_REL and rel_b2 >= TARGET_REL:
            max_b2_mw = b2_mw
            print(f"✅ B2 = {b2_mw:2d} MW  |  Tier A: {rel_a:.3f}%  |  B1: {rel_b1:.3f}%  |  B2: {rel_b2:.3f}%")
        else:
            print(f"❌ B2 = {b2_mw:2d} MW  |  Tier A: {rel_a:.3f}%  |  B1: {rel_b1:.3f}%  |  B2: {rel_b2:.3f}% (FAILED)")
            break

    print(f"\n--> 🏆 Maximum Tier B2 Locked At: {max_b2_mw} MW")
    print("="*60)
    print(f"FINAL STACK CONFIGURATION:")
    print(f"  Firm Capacity Blocks : Variable (from CSV)")
    print(f"  Tier B1 (Daily)      : {max_b1_mw} MW Constant")
    print(f"  Tier B2 (Weekly)     : {max_b2_mw} MW Constant")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_iterative_search()