import numpy as np
import pandas as pd
import sys
import os

# Ensure the script can find your ems module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ems_offgrid_logic_thijs_2_2_26 import ems_cplex_offgrid_full_horizon
except ImportError:
    print("❌ ERROR: Could not find 'ems_offgrid_logic_thijs_2_2_26.py'.")
    sys.exit()

def run_validation():
    print("🔬 STARTING DEFENSE-IN-DEPTH VALIDATION")
    print("="*60)
    
    # --- COMMON SETUP ---
    hours = 24
    PUE = 1.15  # Hardcoded in EMS, must match here for math to check out
    
    # Create Datetime Index to prevent TypeErrors
    time_index = pd.date_range(start='2025-01-01', periods=hours, freq='h')
    
    # System Specs
    b_P = 10.0  
    b_E = 20.0  
    it_cap = 15.0 
    
    # Dummy Empty Profile
    empty_profile = pd.Series(np.zeros(hours), index=time_index)
    
    # =========================================================================
    # TEST 1: HIERARCHY CHECK
    # Does Tier A get priority over Tier B?
    # =========================================================================
    print("\n[TEST 1] HIERARCHY CHECK: Does A get priority over B?")
    
    # Gen (5 MW) is enough for A (4 MW * 1.15 = 4.6 MW) but not B
    gen = pd.Series(np.full(hours, 5.0), index=time_index)
    load_a = np.full(hours, 4.0)
    load_b_daily = np.full(1, 4.0 * 24) 
    
    res = ems_cplex_offgrid_full_horizon(
        wind_ts=gen, solar_ts=empty_profile,
        P_batt_MW=b_P, E_batt_MWh_t=pd.Series([0]), 
        battery_depth_of_discharge=0.9, charge_efficiency=1.0,
        tier_a_profile=load_a, tier_b_profile=load_b_daily,
        it_capacity_mw=it_cap
    )
    
    unserved_a = np.sum(res[4])
    shortfall_b = np.sum(res[5])
    
    if unserved_a < 0.1 and shortfall_b > 1.0:
        print(f"✅ PASS: Tier A served (Unserved={unserved_a:.2f}), Tier B dropped.")
    else:
        print(f"❌ FAIL: Priority Broken. Unserved A: {unserved_a}, Shortfall B: {shortfall_b}")

    # =========================================================================
    # TEST 2: CAPACITY CEILING TEST
    # Does the EMS respect the IT Limit (15 MW) even with infinite power?
    # =========================================================================
    print("\n[TEST 2] CAPACITY CHECK: Do we respect the IT Limit?")
    
    gen = pd.Series(np.full(hours, 100.0), index=time_index) # Infinite Power
    load_a = np.full(hours, 5.0)
    load_b_daily = np.full(1, 1000.0) # Infinite Demand
    
    res = ems_cplex_offgrid_full_horizon(
        wind_ts=gen, solar_ts=empty_profile,
        P_batt_MW=b_P, E_batt_MWh_t=pd.Series([0]), 
        battery_depth_of_discharge=0.9, charge_efficiency=1.0,
        tier_a_profile=load_a, tier_b_profile=load_b_daily,
        it_capacity_mw=it_cap
    )
    
    hpp_output = res[0] # IT Load Served
    max_out = np.max(hpp_output)
    
    if max_out <= it_cap + 0.01:
        print(f"✅ PASS: IT Output capped at {max_out:.2f} MW (Limit: {it_cap})")
    else:
        print(f"❌ FAIL: Output breached limit! ({max_out:.2f} MW)")

    # =========================================================================
    # TEST 3: BATTERY PHYSICS TEST
    # Does energy conservation hold? (Charge -> SOC Increase)
    # =========================================================================
    print("\n[TEST 3] PHYSICS CHECK: Does Energy Conserve?")
    
    gen_vals = np.zeros(hours)
    gen_vals[0:5] = 10.0 # Charge phase
    gen = pd.Series(gen_vals, index=time_index)
    load_a = np.full(hours, 5.0) 
    
    res = ems_cplex_offgrid_full_horizon(
        wind_ts=gen, solar_ts=empty_profile,
        P_batt_MW=b_P, E_batt_MWh_t=pd.Series([b_E]), 
        battery_depth_of_discharge=0.9, charge_efficiency=0.9,
        tier_a_profile=load_a, tier_b_profile=np.zeros(1),
        it_capacity_mw=it_cap
    )
    
    soc = res[3]
    if soc[5] > soc[0]:
        print(f"✅ PASS: Battery SOC increased from {soc[0]:.1f} to {soc[5]:.1f}")
    else:
        print(f"❌ FAIL: Battery did not charge!")

    # =========================================================================
    # TEST 4: DEADLINE TEST
    # Can we catch up on flexible load at the last minute?
    # =========================================================================
    print("\n[TEST 4] DEADLINE CHECK: Can we catch up?")
    
    gen_vals = np.zeros(hours)
    gen_vals[23] = 100.0 # Spike at last hour
    gen = pd.Series(gen_vals, index=time_index)
    load_b_daily = np.full(1, 10.0) 
    
    res = ems_cplex_offgrid_full_horizon(
        wind_ts=gen, solar_ts=empty_profile,
        P_batt_MW=100.0, E_batt_MWh_t=pd.Series([0]), 
        battery_depth_of_discharge=1.0, charge_efficiency=1.0,
        tier_a_profile=np.zeros(hours), tier_b_profile=load_b_daily,
        it_capacity_mw=100.0
    )
    
    total_shortfall = np.sum(res[5])
    if total_shortfall < 0.1:
        print("✅ PASS: Deadline met by surging at the last minute.")
    else:
        print(f"❌ FAIL: Missed deadline. Shortfall: {total_shortfall:.2f}")

    # =========================================================================
    # TEST 5: RELIABILITY MATH TEST (With PUE)
    # Situation: Gen=5 MW, Load=10 MW. PUE=1.15.
    # =========================================================================
    print("\n[TEST 5] RELIABILITY CHECK: Is the math exact (w/ PUE)?")
    
    gen = pd.Series(np.full(hours, 5.0), index=time_index)
    load_a = np.full(hours, 10.0) 
    
    res = ems_cplex_offgrid_full_horizon(
        wind_ts=gen, solar_ts=empty_profile,
        P_batt_MW=0.0, E_batt_MWh_t=pd.Series([0]), 
        battery_depth_of_discharge=1.0, charge_efficiency=1.0,
        tier_a_profile=load_a, tier_b_profile=np.zeros(1),
        it_capacity_mw=100.0
    )
    
    # Manual Calculation:
    # Available Power for IT = Generation / PUE
    max_it_possible = 5.0 / PUE  # 5 / 1.15 = 4.3478 MW
    expected_unserved_mw = max(0, 10.0 - max_it_possible) # 5.652 MW
    expected_unserved_total = expected_unserved_mw * hours # 135.65 MWh
    
    simulated_unserved = np.sum(res[4])
    
    print(f"   Gen: 5.0 MW | PUE: {PUE} | Max IT Supportable: {max_it_possible:.2f} MW")
    print(f"   Target Load: 10.0 MW")
    print(f"   Expected Unserved: {expected_unserved_total:.2f} MWh")
    print(f"   Simulated Unserved: {simulated_unserved:.2f} MWh")
    
    if abs(simulated_unserved - expected_unserved_total) < 0.5:
        print("✅ PASS: Reliability math matches PUE physics.")
    else:
        print(f"❌ FAIL: Calculation Mismatch!")

    # =========================================================================
    # TEST 6: ENERGY CONSERVATION (LCOE Check)
    # Energy In = Energy Out (Accounting for PUE losses)
    # =========================================================================
    print("\n[TEST 6] LCOE INPUT CHECK: Are we losing energy?")
    
    gen = pd.Series(np.full(hours, 10.0), index=time_index) # 240 MWh Total
    load_a = np.full(hours, 5.0)                            # 120 MWh IT Load
    
    res = ems_cplex_offgrid_full_horizon(
        wind_ts=gen, solar_ts=empty_profile,
        P_batt_MW=10.0, E_batt_MWh_t=pd.Series([50]), 
        battery_depth_of_discharge=1.0, charge_efficiency=0.9,
        tier_a_profile=load_a, tier_b_profile=np.zeros(1),
        it_capacity_mw=100.0
    )
    
    P_HPP_IT = res[0] # IT Load Served
    P_Curt   = res[1] # Curtailment
    # Battery Net Flow (Discharge - Charge)
    # Negative value means charging
    Net_Batt_Flow = res[2] 
    
    total_gen = np.sum(gen)
    total_it_served = np.sum(P_HPP_IT)
    total_curtail = np.sum(P_Curt)
    
    # Calculate Real Facility Consumption
    total_facility_load = total_it_served * PUE
    
    # Battery Check
    # Energy into Battery = -(Net Flow when negative)
    # Stored Energy = Energy into Battery * Efficiency
    energy_into_batt_pin = np.sum(-Net_Batt_Flow[Net_Batt_Flow < 0])
    stored_energy_expected = energy_into_batt_pin * 0.9
    
    soc = res[3]
    actual_stored_energy = soc[-1] - soc[0]
    
    # GLOBAL BALANCE:
    # Generation = Facility_Load + Curtailment + Energy_Into_Battery
    energy_out = total_facility_load + total_curtail + energy_into_batt_pin
    diff = total_gen - energy_out
    
    print(f"   Gen Input:      {total_gen:.1f}")
    print(f"   Facility Cons:  {total_facility_load:.1f} (IT * 1.15)")
    print(f"   Curtailment:    {total_curtail:.1f}")
    print(f"   Batt Input:     {energy_into_batt_pin:.1f}")
    print(f"   Global Diff:    {diff:.2f}")
    
    if abs(diff) < 0.5:
        print("✅ PASS: Energy is conserved. LCOE denominator is valid.")
    else:
        print(f"❌ FAIL: Energy Disappeared! Diff: {diff:.2f}")

if __name__ == "__main__":
    run_validation()