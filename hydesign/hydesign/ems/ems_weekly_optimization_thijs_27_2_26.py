# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:01:24 2026

@author: thijs
"""

import os
import numpy as np
import pandas as pd
import openmdao.api as om
from docplex.mp.model import Model
from hydesign.ems.ems import expand_to_lifetime

# -------------------------------------------------------------------------
# 1. MAIN OPENMDAO COMPONENT
# -------------------------------------------------------------------------
class ems_constantoutput(om.ExplicitComponent):
    def __init__(
        self, 
        N_time, 
        life_y = 25,
        life_h = 25*365*24, 
        intervals_per_hour = 1,
        weeks_per_season_per_year = None,
        ems_type='cplex',
        load_min_penalty_factor=1e6):

        super().__init__()
        self.weeks_per_season_per_year = weeks_per_season_per_year
        self.N_time = int(N_time)
        self.ems_type = ems_type
        self.life_h = int(365 * 24 * life_y)
        self.life_intervals = int(self.life_h * intervals_per_hour)
        self.load_min_penalty_factor = load_min_penalty_factor
    
    def setup(self):
        # --- INPUTS ---
        self.add_input('wind_t', desc="WPP power time series", units='MW', shape=[self.N_time])
        self.add_input('solar_t', desc="PVP power time series", units='MW', shape=[self.N_time])
        self.add_input('price_t', desc="Electricity price time series", shape=[self.N_time])
        self.add_input('b_P', desc="Battery power capacity", units='MW')
        self.add_input('b_E', desc="Battery energy storage capacity") 
        self.add_input('G_MW', desc="Grid capacity", units='MW')
        self.add_input('battery_depth_of_discharge', desc="battery depth of discharge") 
        self.add_input('battery_charge_efficiency', desc="battery charge efficiency")
        self.add_input('peak_hr_quantile', desc="Quantile of price tim sereis to define peak price hours")
        self.add_input('cost_of_battery_P_fluct_in_peak_price_ratio', desc="cost of battery power fluctuations")
        self.add_input('n_full_power_hours_expected_per_day_at_peak_price', desc="Penalty occurs if nunmber of full power hours...")
        
        # --- THESIS INPUTS ---
        self.add_input('tier_a_profile', desc="Tier A Power Profile", shape=[self.life_h], units='MW')
        self.add_input('tier_b_profile', desc="Tier B Energy Profile", shape=[self.life_h], units='MW*h') 
        self.add_input('load_profile_ts', desc="Hourly load profile", shape=[self.life_h], units='MW')

        # --- OUTPUTS ---
        self.add_output('wind_t_ext', desc="WPP power time series", units='MW', shape=[self.life_h])
        self.add_output('solar_t_ext', desc="PVP power time series", units='MW', shape=[self.life_h])
        self.add_output('price_t_ext', desc="Electricity price time series", shape=[self.life_h])
        self.add_output('hpp_t', desc="HPP power time series", units='MW', shape=[self.life_h])
        self.add_output('hpp_curt_t', desc="HPP curtailed power time series", units='MW', shape=[self.life_h])
        self.add_output('b_t', desc="Battery charge/discharge power time series", units='MW', shape=[self.life_h])
        self.add_output('b_E_SOC_t', desc="Battery energy SOC time series", units='MW*h', shape=[self.life_h + 1])
        
        self.add_output('Unserved_A', desc="Unserved Tier A Load [MW]", shape=[self.life_h], units='MW')
        self.add_output('Shortfall_B', desc="Tier B Queue Past Deadline [MWh]", shape=[self.life_h], units='MW*h') 
        self.add_output('Served_C2', desc="Opportunistic Load Served [MW]", shape=[self.life_h], units='MW') 
        self.add_output('penalty_t', desc="Total Penalty", shape=[self.life_h]) 

    def compute(self, inputs, outputs):
        wind_t = inputs['wind_t']
        solar_t = inputs['solar_t']
        price_t = inputs['price_t']
        
        b_P = inputs['b_P']
        b_E = inputs['b_E']
        
        batt_eff = inputs['battery_charge_efficiency'][0]
        batt_dod = inputs['battery_depth_of_discharge'][0]
        
        it_capacity_mw = np.max(inputs['load_profile_ts']) 
        
        if batt_eff < 0.01: batt_eff = 0.96
        if batt_dod < 0.01: batt_dod = 0.90

        tier_a_1yr = inputs['tier_a_profile'][:self.N_time]
        tier_b_1yr = inputs['tier_b_profile'][:self.N_time]
        
        WSPr_df = pd.DataFrame(index=pd.date_range(start='01-01-1991 00:00', periods=len(wind_t), freq='h'))
        WSPr_df['wind_t'] = wind_t
        WSPr_df['solar_t'] = solar_t
        WSPr_df['price_t'] = price_t
        WSPr_df['E_batt_MWh_t'] = b_E[0]
        
        # --- CALL CPLEX OPTIMIZER ---
        (P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, 
         Unserved_A_ts, Shortfall_B_ts, Served_C2_ts, penalty_ts) = ems_cplex_offgrid_full_horizon(
            wind_ts = WSPr_df.wind_t,
            solar_ts = WSPr_df.solar_t,
            P_batt_MW = b_P[0],
            E_batt_MWh_t = WSPr_df.E_batt_MWh_t,
            battery_depth_of_discharge = batt_dod,
            charge_efficiency = batt_eff,
            tier_a_profile = tier_a_1yr,
            tier_b_profile = tier_b_1yr,
            it_capacity_mw = it_capacity_mw
        )

        # Expand Results to Lifetime
        outputs['wind_t_ext'] = expand_to_lifetime(wind_t, life=self.life_intervals)
        outputs['solar_t_ext'] = expand_to_lifetime(solar_t, life=self.life_intervals)
        outputs['price_t_ext'] = expand_to_lifetime(price_t, life=self.life_intervals)
        outputs['hpp_t'] = expand_to_lifetime(P_HPP_ts, life=self.life_intervals)
        outputs['hpp_curt_t'] = expand_to_lifetime(P_curtailment_ts, life=self.life_intervals)
        outputs['b_t'] = expand_to_lifetime(P_charge_discharge_ts, life=self.life_intervals)
        outputs['b_E_SOC_t'] = expand_to_lifetime(E_SOC_ts, life=self.life_intervals + 1)
        outputs['penalty_t'] = expand_to_lifetime(penalty_ts, life=self.life_intervals)
        outputs['Unserved_A'] = expand_to_lifetime(Unserved_A_ts, life=self.life_intervals)
        outputs['Shortfall_B'] = expand_to_lifetime(Shortfall_B_ts, life=self.life_intervals)
        outputs['Served_C2'] = expand_to_lifetime(Served_C2_ts, life=self.life_intervals) 

# -------------------------------------------------------------------------
# 2. FULL HORIZON SOLVER (Queue + C2 Opportunistic)
# -------------------------------------------------------------------------
def ems_cplex_offgrid_full_horizon(
    wind_ts, solar_ts, P_batt_MW, E_batt_MWh_t, 
    battery_depth_of_discharge, charge_efficiency, 
    tier_a_profile, tier_b_profile,
    it_capacity_mw
):
    
    mdl = Model(name='EMS_OffGrid_FullYear')
    
    # --- PERFORMANCE TUNING ---
    mdl.context.cplex_parameters.threads = 4 
    mdl.context.cplex_parameters.timelimit = 180 
    mdl.context.cplex_parameters.mip.tolerances.mipgap = 0.01 
    
    time = wind_ts.index
    dt = 1.0
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))

    # --- VARIABLES ---
    P_HPP_t = mdl.continuous_var_dict(time, lb=0, name='P_HPP')
    P_curtailment_t = mdl.continuous_var_dict(time, lb=0, name='Curtailment')
    P_charge = mdl.continuous_var_dict(time, lb=0, ub=(P_batt_MW / charge_efficiency), name='Charge')
    P_discharge = mdl.continuous_var_dict(time, lb=0, ub=(P_batt_MW * charge_efficiency), name='Discharge')
    
    batt_cap = E_batt_MWh_t.iloc[0]
    min_soc = (1 - battery_depth_of_discharge) * batt_cap
    E_SOC_t = mdl.continuous_var_dict(SOCtime, lb=min_soc, ub=batt_cap, name='SOC')
    
    U_tier_A = mdl.continuous_var_dict(time, lb=0, name='Unserved_A')
    P_tier_B = mdl.continuous_var_dict(time, lb=0, name='Served_B')
    P_tier_C2 = mdl.continuous_var_dict(time, lb=0, ub=it_capacity_mw, name='Served_C2')

    # [NEW] Queue Variables instead of Daily Shortfall
    Queue_B_t = mdl.continuous_var_dict(time, lb=0, name='Queue_B')
    Missed_Deadline_B_t = mdl.continuous_var_dict(time, lb=0, name='Missed_Deadline_B')

    PUE = 1.15 
    
    # Derive hourly arrivals from the input profile (assuming it was repeated daily values)
    Arrival_B = tier_b_profile / 24.0

    # --- CONSTRAINTS --- changed for weekly optimization: every week the soc ends at 50%
    
    for w in range(53): 
        idx = min(w * 168, len(SOCtime) - 1)
        mdl.add_constraint(E_SOC_t[SOCtime[idx]] == 0.5 * batt_cap)
        
    mdl.add_constraint(E_SOC_t[SOCtime[-1]] == 0.5 * batt_cap)

    val_tier_a = tier_a_profile
    val_wind = wind_ts.values
    val_solar = solar_ts.values

    # Main Physics Loop
    for i, t in enumerate(time):
        
        # 1. HPP Output = Served A + Served B + Served C2
        mdl.add_constraint(
            P_HPP_t[t] == (val_tier_a[i] - U_tier_A[t]) + P_tier_B[t] + P_tier_C2[t]
        )
        
        # 2. Generation Balance
        mdl.add_constraint(
            val_wind[i] + val_solar[i] + P_discharge[t] 
            == 
            (PUE * P_HPP_t[t]) + P_charge[t] + P_curtailment_t[t]
        )

        # 3. IT CAPACITY LIMIT
        mdl.add_constraint(
            (val_tier_a[i] - U_tier_A[t]) + P_tier_B[t] + P_tier_C2[t] <= it_capacity_mw
        )

        # 4. SOC Evolution
        if t != time[-1]:
            tt = t + pd.Timedelta("1h")
            mdl.add_constraint(
                E_SOC_t[tt] == E_SOC_t[t] + (P_charge[t] * charge_efficiency * dt) - (P_discharge[t] / charge_efficiency * dt)
            )

        # [NEW] 5. Tier B Queue Dynamics
        if i == 0:
            mdl.add_constraint(Queue_B_t[t] == Arrival_B[i] - P_tier_B[t])
        else:
            prev_t = time[i-1]
            mdl.add_constraint(Queue_B_t[t] == Queue_B_t[prev_t] + Arrival_B[i] - P_tier_B[t])

        # [NEW] 6. 24-Hour SLA / Deadline Logic
        start_idx = max(0, i - 23)
        recent_arrivals = np.sum(Arrival_B[start_idx : i+1])
        mdl.add_constraint(Missed_Deadline_B_t[t] >= Queue_B_t[t] - recent_arrivals)

    # --- OBJECTIVE ---
    VoLL_Tier_A = 100_000_000   # Critical failure
    VoLL_Tier_B = 1_000    # Penalty for missing 24h deadline
    Cost_Queue_Latency = 0.0001 # Tiny penalty to nudge optimizer to clear queue early
    Cost_Curtail = 0.01    # Wasted energy
    Cost_Deg = 0.1         # Battery Wear
   # Dynamically grab the C2 Reward from the main script! (Defaults to -0.5)
    Reward_C2 = float(os.environ.get('REWARD_C2', '-0.5'))
    

    obj_A = mdl.sum(U_tier_A[t] for t in time) * VoLL_Tier_A
    obj_B_Deadline = mdl.sum(Missed_Deadline_B_t[t] for t in time) * VoLL_Tier_B
    obj_B_Latency = mdl.sum(Queue_B_t[t] for t in time) * Cost_Queue_Latency
    
    obj_Curt = mdl.sum(P_curtailment_t[t] for t in time) * Cost_Curtail
    obj_Deg = mdl.sum((P_charge[t] + P_discharge[t]) for t in time) * Cost_Deg
    obj_C2 = mdl.sum(P_tier_C2[t] for t in time) * Reward_C2

    mdl.minimize(obj_A + obj_B_Deadline + obj_B_Latency + obj_Curt + obj_Deg + obj_C2)
    
    # --- SOLVE ---
    print("   ... CPLEX solving 8760 hours with Queue & C2 logic (this may take 30-90s) ...")
    sol = mdl.solve(log_output=False)
    
    if sol is None:
        print("❌ SOLVER FAILED to find integer solution. Returning Zeros.")
        return (np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time)), 
                np.zeros(len(time)+1), tier_a_profile, np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time)))

    print("   ✅ Solution found!")
    
    def get_series(var_dict):
        df = pd.DataFrame.from_dict(sol.get_value_dict(var_dict), orient='index')
        if df.empty: return np.zeros(len(time))
        return df.iloc[:,0].reindex(time, fill_value=0).values

    P_HPP_res = get_series(P_HPP_t)
    P_curt_res = get_series(P_curtailment_t)
    P_ch_res = get_series(P_charge)
    P_dis_res = get_series(P_discharge)
    P_batt_res = P_dis_res - P_ch_res 

    E_SOC_res = pd.DataFrame.from_dict(sol.get_value_dict(E_SOC_t), orient='index').iloc[:,0].reindex(SOCtime, fill_value=0).values
    U_A_res = get_series(U_tier_A)
    P_C2_res = get_series(P_tier_C2) 
    
    # Output the SLA violations to 'Shortfall_B' so your reliability metrics pick it up
    S_B_res = get_series(Missed_Deadline_B_t)

    Penalty_res = U_A_res * VoLL_Tier_A + S_B_res * VoLL_Tier_B 
    mdl.end()
    
    return P_HPP_res, P_curt_res, P_batt_res, E_SOC_res, U_A_res, S_B_res, P_C2_res, Penalty_res


# -------------------------------------------------------------------------
# 4. LONG TERM OPERATION (Pass-through)
# -------------------------------------------------------------------------
class ems_long_term_operation(om.ExplicitComponent):
    def __init__(self, N_time, life_y=25, *args, **kwargs):
        super().__init__()
        self.life_h = int(365 * 24 * life_y)

    def setup(self):
        self.add_input('SoH', desc="Battery State of Health", shape=[self.life_h])
        self.add_input('b_P', desc="Battery power capacity", units='MW')
        self.add_input('b_E', desc="Battery energy capacity")
        self.add_input('G_MW', desc="Grid capacity", units='MW')
        self.add_input('battery_charge_efficiency', desc="Battery charge eff")
        self.add_input('battery_depth_of_discharge', desc="Battery DoD") 

        self.add_input('hpp_t', desc="HPP Power TS", units='MW', shape=[self.life_h])
        self.add_input('load_profile_ts', desc="Load Profile", units='MW', shape=[self.life_h])

        self.add_input('wind_t_ext_deg', units='MW', shape=[self.life_h])
        self.add_input('solar_t_ext_deg', units='MW', shape=[self.life_h])
        self.add_input('wind_t_ext', units='MW', shape=[self.life_h])
        self.add_input('solar_t_ext', units='MW', shape=[self.life_h])
        self.add_input('price_t_ext', shape=[self.life_h])
        self.add_input('hpp_curt_t', units='MW', shape=[self.life_h])
        self.add_input('b_t', units='MW', shape=[self.life_h])
        self.add_input('b_E_SOC_t', units='MW*h', shape=[self.life_h + 1])

        self.add_output('total_curtailment', units='GW*h')
        self.add_output('hpp_t_with_deg', units='MW', shape=[self.life_h])
        self.add_output('penalty_t_with_deg', shape=[self.life_h])
        self.add_output('hpp_curt_t_with_deg', units='MW', shape=[self.life_h])
        self.add_output('b_t_with_deg', units='MW', shape=[self.life_h])
        self.add_output('b_E_SOC_t_with_deg', units='MW*h', shape=[self.life_h + 1])
        self.add_output('wind_t_deg', units='MW', shape=[self.life_h])
        self.add_output('solar_t_deg', units='MW', shape=[self.life_h])

    def compute(self, inputs, outputs):
        outputs['hpp_t_with_deg'] = inputs['hpp_t']
        outputs['total_curtailment'] = np.sum(inputs['hpp_curt_t']) / 1e3
        outputs['hpp_curt_t_with_deg'] = inputs['hpp_curt_t']
        outputs['b_t_with_deg'] = inputs['b_t']
        outputs['b_E_SOC_t_with_deg'] = inputs['b_E_SOC_t']
        outputs['wind_t_deg'] = inputs['wind_t_ext_deg']
        outputs['solar_t_deg'] = inputs['solar_t_ext_deg']
        outputs['penalty_t_with_deg'] = np.zeros(self.life_h)