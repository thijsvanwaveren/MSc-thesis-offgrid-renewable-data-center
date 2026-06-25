# -*- coding: utf-8 -*-
"""
Executes a 3D parameter sweep to identify feasible workload mix combinations 
across varying data center capacities.

Evaluates firm (Tier A), daily flexible (Tier B1), and weekly flexible (Tier B2) 
workload allocations against a strict target reliability threshold. Applies 
monotonic pruning rules to optimize computational time by systematically discarding 
known infeasible search spaces. Outputs a CSV containing all viable boundary 
configurations and their corresponding energy delivery metrics.

Pruning rules applied:
1. For fixed (A, B1): if B2 = b is infeasible, the B2 loop breaks.
2. For fixed A: if B1 = b yields no feasible B2, the B1 loop breaks.
3. If A = a is infeasible at B1 = 0 and B2 = 0, the A loop breaks.
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
 
from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model
 
 
# =============================================================================
# CONFIG / HELPERS
# =============================================================================
 
def configure_parameters(thesis_dir):
    """Loads EMS parameters."""

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
    """
    Aggregates an hourly array into consecutive non-overlapping blocks.
    Keeps the final partial block if present.
    """
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
 
 
def get_lcoe_delivered(prob, out):
    """Extracts the Levelized Cost of Energy (LCOE) for delivered power."""
    try:
        return prob.get_val('finance.LCOE_delivered')[0]
    except Exception:
        try:
            return out[3]
        except Exception:
            return np.nan
 
 
def get_curtailment(prob):
    """
    Returns total curtailed energy [MWh] by summing the hourly curtailment output.
    """
    try:
        # Your custom assembly uses 'ems.hpp_curt_t'
        return float(np.sum(prob.get_val('ems.hpp_curt_t'))) 
    except Exception:
        return np.nan
 
def get_energy_served(prob, a_mw, b1_mw, b2_mw, N_life):
    """
    Returns delivered energy per tier [MWh].
    Calculated by taking the total demand and subtracting the known shortfall.
    """
    # 1. Tier A: Total Demand - Unserved
    demand_a = a_mw * N_life
    try:
        unserved_a = float(np.sum(prob.get_val('ems.Unserved_A')))
        energy_a = demand_a - unserved_a
    except Exception:
        energy_a = np.nan

    # 2. Tier B1: Total Demand - Shortfall
    demand_b1 = b1_mw * N_life
    try:
        shortfall_b1 = float(np.sum(prob.get_val('ems.Shortfall_B')))
        energy_b1 = demand_b1 - shortfall_b1
    except Exception:
        energy_b1 = np.nan

    # 3. Tier B2: Total Demand - Shortfall
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

    # 4. Tier C: Explicitly Served (The Sponge)
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
    """
    Calculates the workload-mix objective value:
        V = 1.0*A + 0.7*B1 + 0.4*B2 + 0.1*C
 
    If coefficients are interpreted as relative value multipliers,
    the output is a value-weighted energy metric.
    """
    EA = 0.0 if np.isnan(energies["Energy_A_MWh"]) else energies["Energy_A_MWh"]
    EB1 = 0.0 if np.isnan(energies["Energy_B1_MWh"]) else energies["Energy_B1_MWh"]
    EB2 = 0.0 if np.isnan(energies["Energy_B2_MWh"]) else energies["Energy_B2_MWh"]
    EC = 0.0 if np.isnan(energies["Energy_C_MWh"]) else energies["Energy_C_MWh"]
 
    return (
        vA * EA +
        vB1 * EB1 +
        vB2 * EB2 +
        vC * EC
    )
    
 
def compute_reliabilities(prob, N_life):
    """
    Reliability definitions:
    - Tier A: hourly full-service reliability based on Unserved_A
    - Tier B1: daily completion reliability based on daily shortfall buckets
    - Tier B2: weekly completion reliability based on weekly shortfall buckets
    """
 
    # --- Tier A reliability ---
    unserved_a = prob.get_val('ems.Unserved_A')
    rel_a = 100.0 * (1.0 - (np.sum(unserved_a > 1e-3) / N_life))
 
    # --- Tier B1 reliability (daily buckets) ---
    shortfall_b1 = prob.get_val('ems.Shortfall_B')
    daily_shortfall = aggregate_into_blocks(shortfall_b1, 24)
    rel_b1 = 100.0 * (1.0 - (np.sum(daily_shortfall > 1e-3) / len(daily_shortfall)))
 
    # --- Tier B2 reliability (weekly buckets) ---
    shortfall_b2 = None
    possible_b2_names = [
        'ems.Shortfall_B2',
        'ems.Shortfall_B_weekly',
        'ems.Violation_B2'
    ]
    for nm in possible_b2_names:
        try:
            shortfall_b2 = prob.get_val(nm)
            break
        except Exception:
            continue
 
    if shortfall_b2 is None:
        raise KeyError(
            "Could not find Tier B2 shortfall variable in prob outputs. "
            "Update possible_b2_names in compute_reliabilities()."
        )
 
    weekly_shortfall = aggregate_into_blocks(shortfall_b2, 24 * 7)
    rel_b2 = 100.0 * (1.0 - (np.sum(weekly_shortfall > 1e-3) / len(weekly_shortfall)))
 
    return rel_a, rel_b1, rel_b2
 
 
# =============================================================================
# MAIN SWEEP
# =============================================================================
 

def run_3d_feasible_sweep(target_rel=99.9):
    """Runs the capacity optimization sweep across Tier A, B1, and B2 using pruning rules."""
    # HPP design
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10]

    # --- CAPACITY CONSTRAINTS ---
    MAX_IT_CAPACITY_MW = 30.0  # Data center size
    MIN_UTILIZATION_PCT = 0.0 # 40% minimum target, lower utilization is expected not to generate well performing results.
    MIN_IT_CAPACITY_MW = MAX_IT_CAPACITY_MW * MIN_UTILIZATION_PCT

    N_life = 25 * 8760
    site_name = 'Denmark_good_solar'

    examples_sites = pd.read_csv(
        os.path.join(thesis_dir, '..', 'examples_sites.csv'),
        sep=';'
    )
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    # Disable Tier C for now
    os.environ['REWARD_C2'] = '-0.5' #Specify Tier C reward to 'enable' it

    # ------------------------------
    # Sweep ranges
    # ------------------------------
    tier_a_array = np.arange(0.0, MAX_IT_CAPACITY_MW + 1, 1.0)
    tier_b1_array = np.arange(0.0, MAX_IT_CAPACITY_MW + 1, 1.0)
    tier_b2_array = np.arange(0.0, MAX_IT_CAPACITY_MW + 1, 1.0)

    feasible_results = []
    max_b1_ceiling = max(tier_b1_array)

    for a_mw in tier_a_array:
        # Pre-emptive check: If A alone exceeds max cap, break completely
        if a_mw > MAX_IT_CAPACITY_MW:
            break
            
        print(f"Sweeping A = {a_mw:.1f} MW")
        current_max_b1_for_this_a = -1
        any_viable_for_this_a = False

        for b1_mw in tier_b1_array:
            # PRUNING RULE A: frontier ceiling from previous A
            if b1_mw > max_b1_ceiling:
                print(f"Skip B1={b1_mw:.1f} (above ceiling {max_b1_ceiling:.1f})")
                continue

            # Pre-emptive check: If A + B1 already exceeds max cap, break B1 loop
            if (a_mw + b1_mw) > MAX_IT_CAPACITY_MW:
                break

            pair_has_any_viable_b2 = False

            for b2_mw in tier_b2_array:
                total_provisioned = a_mw + b1_mw + b2_mw

                # 1. UPPER BOUND CHECK (IT MAX)
                if total_provisioned > MAX_IT_CAPACITY_MW:
                    # Because B2 is increasing, all future B2s will also exceed the cap. Break loop.
                    break

                # 2. LOWER BOUND CHECK (40% MINIMUM)
                if total_provisioned < MIN_IT_CAPACITY_MW:
                    # We haven't reached the 40% threshold yet. Continue to next, larger B2.
                    continue

                print(
                    f"Testing -> A={a_mw:04.1f} | B1={b1_mw:04.1f} | B2={b2_mw:04.1f} (Total {total_provisioned} MW) ...",
                    end=""
                )

                # ------------------------------
                # Build workload signals
                # ------------------------------
                t_a_ts = np.full(N_life, a_mw)
                t_b1_daily_target_ts = np.full(N_life, b1_mw * 24.0)
                t_b2_weekly_target_ts = np.full(N_life, b2_mw * 24.0 * 7.0)

                total_load_for_ems = np.full(N_life, total_provisioned)
                total_load_for_ems[0] = MAX_IT_CAPACITY_MW

                hpp = hpp_model(
                    latitude=ex_site['latitude'].values[0],
                    longitude=ex_site['longitude'].values[0],
                    altitude=ex_site['altitude'].values[0],
                    num_batteries=1,
                    work_dir=current_dir,
                    input_ts_fn=weather_fn,
                    sim_pars_fn=sim_pars_fn,
                    tier_a_profile=t_a_ts,
                    tier_b_profile=t_b1_daily_target_ts,
                    tier_b2_profile=t_b2_weekly_target_ts,
                    load_profile_ts=total_load_for_ems,
                    battery_deg=False
                )

                out = hpp.evaluate(*fixed_design)
                prob = hpp.prob

                rel_a, rel_b1, rel_b2 = compute_reliabilities(prob, N_life)

                is_viable = (
                    (rel_a >= target_rel) and
                    (rel_b1 >= target_rel) and
                    (rel_b2 >= target_rel)
                )

                if is_viable:
                    lcoe_d = get_lcoe_delivered(prob, out)
                    curt = get_curtailment(prob)
                    energies = get_energy_served(prob, a_mw, b1_mw, b2_mw, N_life)
                    
                    mix_value = calculate_mix_value(energies, vA=1.0, vB1=0.7, vB2=0.4, vC=0.1)
                    mix_value_annual_gwh = mix_value / (25 * 1000)

                    total_delivered_energy = (
                        (0.0 if np.isnan(energies["Energy_A_MWh"]) else energies["Energy_A_MWh"]) +
                        (0.0 if np.isnan(energies["Energy_B1_MWh"]) else energies["Energy_B1_MWh"]) +
                        (0.0 if np.isnan(energies["Energy_B2_MWh"]) else energies["Energy_B2_MWh"]) +
                        (0.0 if np.isnan(energies["Energy_C_MWh"]) else energies["Energy_C_MWh"])
                    )

                    realized_utilization = total_delivered_energy / (MAX_IT_CAPACITY_MW * N_life)

                    print(
                        f" Viable | RelA={rel_a:.3f}% | RelB1={rel_b1:.3f}% | "
                        f"RelB2={rel_b2:.3f}% | Value={mix_value:,.1f}"
                    )

                    feasible_results.append({
                        "Total_Provisioned_MW": total_provisioned,
                        "Tier_A_MW": a_mw,
                        "Tier_B1_MW": b1_mw,
                        "Tier_B2_MW": b2_mw,
                        "Reliability_A": rel_a,
                        "Reliability_B1": rel_b1,
                        "Reliability_B2": rel_b2,
                        "LCOE_delivered": lcoe_d,
                        "Curtailment_Annual_GWh": curt / (25 * 1000),
                        "Energy_A_Annual_GWh": energies["Energy_A_MWh"] / (25 * 1000),
                        "Energy_B1_Annual_GWh": energies["Energy_B1_MWh"] / (25 * 1000),
                        "Energy_B2_Annual_GWh": energies["Energy_B2_MWh"] / (25 * 1000),
                        "Energy_C_Annual_GWh": energies["Energy_C_MWh"] / (25 * 1000),
                        "Total_Delivered_Annual_GWh": total_delivered_energy / (25 * 1000),
                        "Objective_Value": mix_value_annual_gwh,
                        "Realized_Utilization": realized_utilization,
                    })
                     
                    pair_has_any_viable_b2 = True
                    any_viable_for_this_a = True

                else:
                    print(
                        f" Infeasible | RelA={rel_a:.3f}% | RelB1={rel_b1:.3f}% | RelB2={rel_b2:.3f}%"
                    )
                    # PRUNING RULE B: For fixed (A, B1), if B2=b fails reliability, all larger B2 will fail.
                    print(f"Break B2 loop for (A={a_mw:.1f}, B1={b1_mw:.1f})")
                    break

            # End B2 loop

            if pair_has_any_viable_b2:
                current_max_b1_for_this_a = b1_mw
            else:
                # PRUNING RULE C
                print(f"Break B1 loop at A={a_mw:.1f} because no feasible B2 existed for B1={b1_mw:.1f}")
                break

        # End B1 loop

        if current_max_b1_for_this_a != -1:
            max_b1_ceiling = current_max_b1_for_this_a

        if not any_viable_for_this_a:
            # PRUNING RULE D
            print(f"FATAL: A={a_mw:.1f} failed across all bounds. Ending A sweeps.")
            break

    # --------------------------------------------
    # Save feasible combinations
    # --------------------------------------------
    df_feasible = pd.DataFrame(feasible_results)

    if df_feasible.empty:
        print("No feasible combinations found.")
        return df_feasible

    df_feasible = df_feasible.sort_values(by="Objective_Value", ascending=False)
    csv_fn = os.path.join(current_dir, f'Feasible_3D_Sweep_Results_{target_rel:.1f}pct_IT{MAX_IT_CAPACITY_MW}.csv')
    df_feasible.to_csv(csv_fn, index=False)

    return df_feasible 
 
# =============================================================================
# ENTRY POINT
# =============================================================================
 
if __name__ == "__main__":
    print("Data Center Hybrid Power Plant 3D Feasible Sweep")
    try:
        user_input = input("Enter the target reliability percentage (e.g., 99.9, 95.0): ")
        target_reliability = float(user_input.strip())
    except ValueError:
        print("Invalid input detected. Defaulting to 99.9% reliability.")
        target_reliability = 99.9
 
    run_3d_feasible_sweep(target_rel=target_reliability)