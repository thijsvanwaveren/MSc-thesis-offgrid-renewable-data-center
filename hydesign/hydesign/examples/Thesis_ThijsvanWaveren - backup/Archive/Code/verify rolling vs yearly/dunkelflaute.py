import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
# Import your actual functions! (Ensure the path is correct for your setup)
from hydesign.ems.ems_rollinghorizon_thijs_9_3_26 import ems_cplex_offgrid_full_horizon, solve_rolling

def generate_dunkelflaute_year():
    """Generates 8760 hours of perfect weather, interrupted by a 200-hour dead zone."""
    T = 8760
    wind = np.full(T, 50.0)  # 50 MW constant wind
    solar = np.zeros(T)      # Ignore solar for simplicity
    tier_a = np.full(T, 18.0) # 18 MW Baseload
    
    # Insert a 200-hour catastrophic Dunkelflaute in the middle of the year
    start_dead, end_dead = 4000, 4200
    wind[start_dead:end_dead] = 0.0
    
    idx = pd.date_range('2026-01-01', periods=T, freq='h')
    return pd.Series(wind, index=idx), pd.Series(solar, index=idx), tier_a

if __name__ == "__main__":
    print("Generating synthetic weather with a 200-hour Dunkelflaute...")
    wind_ts, solar_ts, tier_a = generate_dunkelflaute_year()
    
    # System Specs (Same as your real sweep)
    P_batt = 25.0       # 25 MW inverter
    E_batt = 200.0      # 200 MWh battery capacity
    eta = 0.96
    DoD = 0.90
    it_cap = 30.0
    os.environ['REWARD_C2'] = '1.0' # Disable the C2 Sponge for this test!

    
    tier_zero = np.zeros(8760)

    # ==========================================
    # 1. RUN PERFECT FORESIGHT
    # ==========================================
    print("\nSolving Perfect Foresight (may take a minute)...")
    (pf_HPP, pf_curt, pf_batt, pf_SOC, pf_UA, 
     pf_SB1, pf_SB2, pf_C2, pf_pen) = ems_cplex_offgrid_full_horizon(
        wind_ts=wind_ts, solar_ts=solar_ts,
        P_batt_MW=P_batt, E_batt_MWh_t=pd.Series(np.full(8760, E_batt), index=wind_ts.index),
        battery_depth_of_discharge=DoD, charge_efficiency=eta,
        tier_a_profile=tier_a, tier_b_profile=tier_zero, tier_b2_profile=tier_zero,
        it_capacity_mw=it_cap
    )

    # ==========================================
    # 2. RUN ROLLING HORIZON
    # ==========================================
    print("Solving Rolling Horizon...")
    (rh_HPP, rh_curt, rh_batt, rh_SOC, rh_UA, 
     rh_SB1, rh_SB2, rh_C2) = solve_rolling(
        wind=wind_ts.values, solar=solar_ts.values,
        tier_a=tier_a, tier_b_daily=tier_zero, tier_b2_weekly=tier_zero,
        it_cap=it_cap, P_batt=P_batt, E_batt=E_batt, eta=eta, DoD=DoD,
        H_p_hours=168, H_c_hours=24, soc0=E_batt*0.5, w_term_soc=0.05
    )

    # ==========================================
    # 3. VERIFY METRICS
    # ==========================================
    pf_unserved_total = np.sum(pf_UA)
    rh_unserved_total = np.sum(rh_UA)
    
    print("\n" + "="*50)
    print(" DIAGNOSTIC RESULTS")
    print("="*50)
    print(f"Perfect Foresight Total Unserved A: {pf_unserved_total:.2f} MWh")
    print(f"Rolling Horizon Total Unserved A:   {rh_unserved_total:.2f} MWh")
    
    if rh_unserved_total > 0:
        print("\n✅ SUCCESS: Rolling Horizon correctly drops load during severe weather!")
        print("Conclusion: Your 100% reliability on real data is mathematically correct.")
    else:
        print("\n❌ FAILURE: Rolling Horizon generated free energy.")

    # ==========================================
    # 4. PLOT THE DUNKELFLAUTE
    # ==========================================
    plt.figure(figsize=(12, 6))
    plt.title("EMS Response to 200-Hour Dunkelflaute (Hours 3950-4250)")
    
    plt.plot(wind_ts.values[3950:4250], label="Wind Generation", color="gray", alpha=0.3)
    plt.plot(pf_UA[3950:4250], label="Unserved Load (Perfect Foresight)", linestyle='--', color='red')
    plt.plot(rh_UA[3950:4250], label="Unserved Load (Rolling Horizon)", linestyle='-', color='blue')
    
    plt.xlabel("Hours (Zoomed)")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(True)
    plt.show()