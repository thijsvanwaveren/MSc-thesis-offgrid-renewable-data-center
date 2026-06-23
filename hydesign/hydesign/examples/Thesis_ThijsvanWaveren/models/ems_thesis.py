# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 15:04:29 2026

@author: thijs
"""

# models/ems_thesis.py
from docplex.mp.model import Model

def optimize_thesis_dispatch(wind_ts, solar_ts, tier_a_profile, tier_b_daily_energy, params):
    mdl = Model(name="Thesis_EMS")
    
    # --- DECISION VARIABLES [cite: 483] ---
    # Power dispatch variables
    P_tier_a = mdl.continuous_var_list(8760, name="P_A")
    P_tier_b = mdl.continuous_var_list(8760, name="P_B")
    P_tier_c = mdl.continuous_var_list(8760, name="P_C") # Opportunistic [cite: 576]
    
    # Storage variables
    P_ch = mdl.continuous_var_list(8760, name="Ch")
    P_dis = mdl.continuous_var_list(8760, name="Dis")
    E_soc = mdl.continuous_var_list(8760, name="SoC")
    
    # Slack/Penalty variables
    U_tier_a = mdl.continuous_var_list(8760, name="Unserved_A")
    Short_tier_b = mdl.continuous_var_list(365, name="Shortfall_B") # Daily variable!
    Curtailment = mdl.continuous_var_list(8760, name="Curt")

    # --- OBJECTIVE FUNCTION (Eq 2.1) [cite: 499] ---
    # Minimize: Cost_A * U_A + Cost_B * Short_B + Cost_Curt * Curt + ...
    total_cost = (
        mdl.sum(params['c_a'] * U_tier_a[t] for t in range(8760)) + 
        mdl.sum(params['c_b'] * Short_tier_b[d] for d in range(365)) +
        mdl.sum(params['c_curt'] * Curtailment[t] for t in range(8760))
        # ... add battery degradation terms ...
    )
    mdl.minimize(total_cost)

    # --- CONSTRAINTS ---
    
    # 1. Power Balance (Eq 2.2) [cite: 526]
    # Supply = Demand + Charging + Curtailment
    for t in range(8760):
        supply = wind_ts[t] + solar_ts[t] + P_dis[t]
        demand = params['PUE'] * (P_tier_a[t] + P_tier_b[t] + P_tier_c[t]) + P_ch[t] + Curtailment[t]
        mdl.add_constraint(supply == demand)

    # 2. Tier A Constraint (Eq 2.8) 
    # Served + Unserved = Required
    for t in range(8760):
        mdl.add_constraint(P_tier_a[t] + U_tier_a[t] == tier_a_profile[t])

    # 3. Tier B Constraint (Eq 2.9) 
    # Sum of power over the day + Shortfall = Daily Requirement
    for d in range(365):
        start_hour = d * 24
        end_hour = (d + 1) * 24
        daily_sum = mdl.sum(P_tier_b[t] for t in range(start_hour, end_hour))
        mdl.add_constraint(daily_sum + Short_tier_b[d] == tier_b_daily_energy[d])

    # ... Add BESS constraints (Eq 2.4-2.7) ...
    # ... Add IT Capacity constraint (Eq 2.11) ...

    # Solve
    mdl.solve()
    return mdl # Return results