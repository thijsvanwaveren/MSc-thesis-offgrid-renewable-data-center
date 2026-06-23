# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:03:05 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Discrete Black-Box Optimization using Optuna (Bayesian Optimization)
Maximizes Workload Value with strict 1 MW steps, 99.9% SLA, and >=50% Utilization.
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import optuna

warnings.filterwarnings("ignore", category=RuntimeWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- HYDESIGN IMPORTS & PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model

# --- GLOBAL SETTINGS ---
TARGET_REL = 99.9
MAX_IT_CAPACITY_MW = 16
MIN_UTILIZATION = 0.2  # NEW: 50% Minimum Utilization
MIN_IT_CAPACITY_MW = MAX_IT_CAPACITY_MW * MIN_UTILIZATION
N_LIFE = 25 * 8760
FIXED_DESIGN = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10]
SITE_NAME = 'Denmark_good_solar'

# Ensure Tier C is disabled
os.environ['REWARD_C2'] = '1'

# =============================================================================
# CONFIG / HELPERS 
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

def aggregate_into_blocks(arr, block_size):
    n = len(arr)
    n_full = n // block_size
    remainder = n % block_size

    blocks = []
    if n_full > 0:
        full_part = arr[:n_full * block_size].reshape(n_full, block_size)
        blocks.extend(np.sum(full_part, axis=1))

    if remainder > 0:
        blocks.append(np.sum(arr[n_full * block_size:]))

    return np.array(blocks)

def get_energy_served(prob, a_mw, b1_mw, b2_mw, N_life):
    demand_a = a_mw * N_life
    try:
        unserved_a = float(np.sum(prob.get_val('ems.Unserved_A')))
        energy_a = demand_a - unserved_a
    except Exception:
        energy_a = np.nan

    demand_b1 = b1_mw * N_life
    try:
        shortfall_b1 = float(np.sum(prob.get_val('ems.Shortfall_B')))
        energy_b1 = demand_b1 - shortfall_b1
    except Exception:
        energy_b1 = np.nan

    demand_b2 = b2_mw * N_life
    shortfall_b2 = 0.0
    possible_b2_names = ['ems.Shortfall_B2', 'ems.Shortfall_B_weekly', 'ems.Violation_B2']
    for nm in possible_b2_names:
        try:
            shortfall_b2 = float(np.sum(prob.get_val(nm)))
            break
        except Exception:
            continue
    energy_b2 = demand_b2 - shortfall_b2

    try:
        energy_c = float(np.sum(prob.get_val('ems.Served_C2')))
    except Exception:
        energy_c = 0.0

    return {
        "Energy_A_MWh": energy_a,
        "Energy_B1_MWh": energy_b1,
        "Energy_B2_MWh": energy_b2,
        "Energy_C_MWh": energy_c
    }

def calculate_mix_value(energies, vA=1.0, vB1=0.7, vB2=0.4, vC=0.1):
    EA = 0.0 if np.isnan(energies["Energy_A_MWh"]) else energies["Energy_A_MWh"]
    EB1 = 0.0 if np.isnan(energies["Energy_B1_MWh"]) else energies["Energy_B1_MWh"]
    EB2 = 0.0 if np.isnan(energies["Energy_B2_MWh"]) else energies["Energy_B2_MWh"]
    EC = 0.0 if np.isnan(energies["Energy_C_MWh"]) else energies["Energy_C_MWh"]

    return vA * EA + vB1 * EB1 + vB2 * EB2 + vC * EC
    
def compute_reliabilities(prob, N_life):
    unserved_a = prob.get_val('ems.Unserved_A')
    rel_a = 100.0 * (1.0 - (np.sum(unserved_a > 1e-3) / N_life))

    shortfall_b1 = prob.get_val('ems.Shortfall_B')
    daily_shortfall = aggregate_into_blocks(shortfall_b1, 24)
    rel_b1 = 100.0 * (1.0 - (np.sum(daily_shortfall > 1e-3) / len(daily_shortfall)))

    shortfall_b2 = None
    possible_b2_names = ['ems.Shortfall_B2', 'ems.Shortfall_B_weekly', 'ems.Violation_B2']
    for nm in possible_b2_names:
        try:
            shortfall_b2 = prob.get_val(nm)
            break
        except Exception:
            continue

    if shortfall_b2 is None:
        raise KeyError("Could not find Tier B2 shortfall variable.")

    weekly_shortfall = aggregate_into_blocks(shortfall_b2, 24 * 7)
    rel_b2 = 100.0 * (1.0 - (np.sum(weekly_shortfall > 1e-3) / len(weekly_shortfall)))

    return rel_a, rel_b1, rel_b2

# =============================================================================
# OPTIMIZATION LOGIC
# =============================================================================

def setup_simulation_files():
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == SITE_NAME]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)
    return ex_site, weather_fn, sim_pars_fn

EX_SITE, WEATHER_FN, SIM_PARS_FN = setup_simulation_files()

def run_simulation(a_mw, b1_mw, b2_mw):
    t_a_ts = np.full(N_LIFE, a_mw)
    t_b1_daily_target_ts = np.full(N_LIFE, b1_mw * 24.0)
    t_b2_weekly_target_ts = np.full(N_LIFE, b2_mw * 24.0 * 7.0)

    total_load_for_ems = np.full(N_LIFE, a_mw + b1_mw + b2_mw)
    total_load_for_ems[0] = MAX_IT_CAPACITY_MW

    hpp = hpp_model(
        latitude=EX_SITE['latitude'].values[0],
        longitude=EX_SITE['longitude'].values[0],
        altitude=EX_SITE['altitude'].values[0],
        num_batteries=1,
        work_dir=current_dir,
        input_ts_fn=WEATHER_FN,
        sim_pars_fn=SIM_PARS_FN,
        tier_a_profile=t_a_ts,
        tier_b_profile=t_b1_daily_target_ts,
        tier_b2_profile=t_b2_weekly_target_ts,
        load_profile_ts=total_load_for_ems,
        battery_deg=False
    )

    hpp.evaluate(*FIXED_DESIGN)
    prob = hpp.prob

    rel_a, rel_b1, rel_b2 = compute_reliabilities(prob, N_LIFE)
    energies = get_energy_served(prob, a_mw, b1_mw, b2_mw, N_LIFE)
    mix_value = calculate_mix_value(energies, vA=1.0, vB1=0.7, vB2=0.4, vC=0.1)
    
    annual_gwh_value = mix_value / (25 * 1000)

    return annual_gwh_value, rel_a, rel_b1, rel_b2

def objective(trial):
    # 1. Define Discrete Search Space
    a_mw = trial.suggest_float('Tier_A', 0, 8, step=1.0)
    b1_mw = trial.suggest_float('Tier_B1', 0, 16, step=1.0)
    b2_mw = trial.suggest_float('Tier_B2', 0, 16, step=1.0) # Changed to float for consistency
    
    total_mw = a_mw + b1_mw + b2_mw
    
    print(f"\n[Trial {trial.number}] Testing -> A:{a_mw} | B1:{b1_mw} | B2:{b2_mw} (Total: {total_mw} MW)")

    # 2. HARD PHYSICAL CONSTRAINTS (Pruning before the simulation runs)
    
    # Upper Bound: Do not exceed Max IT Capacity
    if total_mw > MAX_IT_CAPACITY_MW:
        print("   -> Pruned (Exceeds Max IT Capacity)")
        raise optuna.TrialPruned() 
        
    # Lower Bound: Must meet minimum utilization threshold
    if total_mw < MIN_IT_CAPACITY_MW:
        print(f"   -> Pruned (Below {MIN_UTILIZATION*100}% Min Utilization: {total_mw} MW < {MIN_IT_CAPACITY_MW} MW)")
        raise optuna.TrialPruned()

    # 3. Run the Heavy Simulation
    annual_value, rel_a, rel_b1, rel_b2 = run_simulation(a_mw, b1_mw, b2_mw)
    print(f"   -> Value: {annual_value:.1f} GWh | Rels: [{rel_a:.2f}%, {rel_b1:.2f}%, {rel_b2:.2f}%]")

    # 4. Enforce SLA Penalties
    min_rel = min(rel_a, rel_b1, rel_b2)
    
    if min_rel < TARGET_REL:
        penalty = (TARGET_REL - min_rel) * 1000 
        penalized_value = annual_value - penalty
        print(f"   -> ❌ Infeasible (Penalty Applied)")
        return penalized_value

    print(f"   -> ✅ Feasible")
    return annual_value

# =============================================================================
# ENTRY POINT
# =============================================================================

def run_bayesian_optimization():
    print("\n" + "="*80)
    print(" STARTING BAYESIAN BLACK-BOX OPTIMIZATION (OPTUNA) ".center(80))
    print(f" Target Reliability: {TARGET_REL}% | Min Utilization: {MIN_UTILIZATION*100}% ".center(80))
    print("="*80)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    
    # Optional seed (ensure your seed is >= 30 MW to prevent immediate pruning)
    study.enqueue_trial({'Tier_A': 8, 'Tier_B1': 0, 'Tier_B2': 0})

    try:
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Returning best results so far...")

    print("\n" + "="*80)
    print("🎉 OPTIMIZATION COMPLETE")
    print("="*80)
    
    feasible_trials = [t for t in study.trials if t.value is not None and t.value > 0]
    
    if len(feasible_trials) == 0:
        print(f"❌ No feasible combinations found meeting 99.9% reliability AND {MIN_UTILIZATION*100}% utilization.")
    else:
        best = study.best_trial
        print(f"BEST OPTIMAL COMPOSITION FOUND (Trial {best.number}):")
        print(f"Tier A (Rigid):  {best.params['Tier_A']} MW")
        print(f"Tier B1 (Daily): {best.params['Tier_B1']} MW")
        print(f"Tier B2 (Weekly):{best.params['Tier_B2']} MW")
        print(f"Total Capacity:  {sum(best.params.values())} MW")
        print(f"Maximum Value:   {best.value:.2f} (Annual GWh Weighted)")

if __name__ == "__main__":
    run_bayesian_optimization()