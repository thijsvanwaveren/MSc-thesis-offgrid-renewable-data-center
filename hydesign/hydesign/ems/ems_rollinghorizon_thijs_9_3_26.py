# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:52:18 2026

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
        run_mode ='rolling', 
        H_p_hours = 168,         # weekly planning horizon
        H_c_hours = 24,         # daily control horizon
        w_terminal_soc = 0.05,     #termial soc penalty weight
        load_min_penalty_factor=1e6):

        super().__init__()
        self.weeks_per_season_per_year = weeks_per_season_per_year
        self.N_time = int(N_time)
        self.ems_type = ems_type
        self.life_h = int(365 * 24 * life_y)
        self.life_intervals = int(self.life_h * intervals_per_hour)
        self.run_mode = run_mode
        self.H_p_hours = int(H_p_hours)
        self.H_c_hours = int(H_c_hours)
        self.w_terminal_soc = float(w_terminal_soc)
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
        self.add_input('tier_b2_profile', desc="Tier B2 Energy Profile", shape=[self.life_h], units='MW*h')
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
        self.add_output('Shortfall_B2', desc="Tier B2 Queue Past Deadline [MWh]", shape=[self.life_h], units='MW*h')
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
        tier_b2_1yr = inputs['tier_b2_profile'][:self.N_time]  # <-- ADD THIS
        
        WSPr_df = pd.DataFrame(index=pd.date_range(start='01-01-1991 00:00', periods=len(wind_t), freq='h'))
        WSPr_df['wind_t'] = wind_t
        WSPr_df['solar_t'] = solar_t
        WSPr_df['price_t'] = price_t
        WSPr_df['E_batt_MWh_t'] = b_E[0]
        
        # --- CALL CPLEX OPTIMIZER ---
        if self.run_mode == 'rolling':
    # Rolling-horizon (recommended for weekly/realistic operation)
            (P_HPP_ts, P_curt_ts, P_batt_ts, E_SOC_ts,
                 Unserved_A_ts, Shortfall_B_ts, Shortfall_B2_ts, Served_C2_ts) = solve_rolling(
                wind=WSPr_df.wind_t.values,
                solar=WSPr_df.solar_t.values,
                tier_a=tier_a_1yr,
                tier_b_daily=tier_b_1yr,
                tier_b2_weekly=tier_b2_1yr,
                it_cap=it_capacity_mw,
                P_batt=b_P[0], E_batt=b_E[0],
                eta=batt_eff, DoD=batt_dod,
                H_p_hours=self.H_p_hours,   # e.g., 168 or 336
                H_c_hours=self.H_c_hours,   # e.g., 24 or 48
                soc0=b_E[0]*0.5,
                w_term_soc=self.w_terminal_soc,
                threads=4, timelimit=120
            )
                     
                     # 👇 INJECT THESE 3 LINES RIGHT HERE 👇
            print(f"\n--- DEBUG ROLLING HORIZON ---")
            print(f"Target Baseload (Max): {np.max(tier_a_1yr)} MW")
            print(f"Total Unserved A:      {np.sum(Unserved_A_ts)} MWh")
            
            penalty_ts = (Unserved_A_ts * 1.0e8) + (Shortfall_B_ts * 1.0e3) + (Shortfall_B2_ts * 1.0e2)  
        else:
            # Annual perfect-foresight baseline
            (P_HPP_ts, P_curt_ts, P_charge_discharge_ts, E_SOC_ts,
             Unserved_A_ts, Shortfall_B_ts, Shortfall_B2_ts, Served_C2_ts, penalty_ts) = ems_cplex_offgrid_full_horizon(
                wind_ts=WSPr_df.wind_t,
                solar_ts=WSPr_df.solar_t,
                P_batt_MW=b_P[0],
                E_batt_MWh_t=WSPr_df.E_batt_MWh_t,
                battery_depth_of_discharge=batt_dod,
                charge_efficiency=batt_eff,
                tier_a_profile=tier_a_1yr,
                tier_b_profile=tier_b_1yr,
                tier_b2_profile=tier_b2_1yr,
                it_capacity_mw=it_capacity_mw
            )
            P_batt_ts = P_charge_discharge_ts

       
 
        # Expand Results to Lifetime
        outputs['wind_t_ext'] = expand_to_lifetime(wind_t, life=self.life_intervals)
        outputs['solar_t_ext'] = expand_to_lifetime(solar_t, life=self.life_intervals)
        outputs['price_t_ext'] = expand_to_lifetime(price_t, life=self.life_intervals)
        outputs['hpp_t'] = expand_to_lifetime(P_HPP_ts, life=self.life_intervals)
        outputs['hpp_curt_t'] = expand_to_lifetime(P_curt_ts, life=self.life_intervals)
        if self.run_mode == 'rolling':
            outputs['b_t'] = expand_to_lifetime(P_batt_ts, life=self.life_intervals)
        else:
            outputs['b_t'] = expand_to_lifetime(P_charge_discharge_ts, life=self.life_intervals)
            
        outputs['b_E_SOC_t'] = expand_to_lifetime(E_SOC_ts, life=self.life_intervals + 1)
        outputs['penalty_t'] = expand_to_lifetime(penalty_ts, life=self.life_intervals)
        outputs['Unserved_A'] = expand_to_lifetime(Unserved_A_ts, life=self.life_intervals)
        outputs['Shortfall_B'] = expand_to_lifetime(Shortfall_B_ts, life=self.life_intervals)
        outputs['Shortfall_B2'] = expand_to_lifetime(Shortfall_B2_ts, life=self.life_intervals)
        outputs['Served_C2'] = expand_to_lifetime(Served_C2_ts, life=self.life_intervals) 


def ems_cplex_window(
    wind_ts, solar_ts,                     # pandas Series, hourly DateTimeIndex
    P_batt_MW, E_batt_MWh,                 # scalars
    battery_depth_of_discharge, charge_efficiency,
    tier_a, tier_b_daily, tier_b2_weekly,  # numpy arrays, len = |window|
    it_capacity_mw,
    soc_init, qB_init, qB2_init, prev_arrivals_B1, prev_arrivals_B2,
    soc_target=None, w_terminal_soc=0.05,
    threads=4, timelimit=120, mipgap=0.001
):
    from docplex.mp.model import Model
    mdl = Model(name='EMS_Window')
    mdl.context.cplex_parameters.threads = threads
    mdl.context.cplex_parameters.timelimit = timelimit
    mdl.context.cplex_parameters.mip.tolerances.mipgap = mipgap
    
    time = wind_ts.index
    dt = 1.0
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1h')]))

    batt_cap = float(E_batt_MWh)
    min_soc  = (1.0 - battery_depth_of_discharge) * batt_cap
    soc_init = max(min_soc, min(batt_cap, float(soc_init)))
    
    # --- VARIABLES ---
    P_HPP_t         = mdl.continuous_var_dict(time, lb=0, name='P_HPP')
    P_curtail_t     = mdl.continuous_var_dict(time, lb=0, name='Curt')
    
    # PATCH 1: Inverter bounds must strictly be P_batt_MW
    P_charge        = mdl.continuous_var_dict(time, lb=0, ub=P_batt_MW, name='Ch')
    P_discharge     = mdl.continuous_var_dict(time, lb=0, ub=P_batt_MW, name='Dis')
    
    E_SOC_t         = mdl.continuous_var_dict(SOCtime, lb=min_soc, ub=batt_cap, name='SOC')
    
    U_tier_A        = mdl.continuous_var_dict(time, lb=0, name='U_A')
    P_tier_B1       = mdl.continuous_var_dict(time, lb=0, name='P_B1')
    P_tier_B2       = mdl.continuous_var_dict(time, lb=0, name='P_B2')
    P_tier_C2       = mdl.continuous_var_dict(time, lb=0, ub=it_capacity_mw, name='P_C2')
    
    Q_B1_t          = mdl.continuous_var_dict(time, lb=0, name='Q_B1')
    V_B1_t          = mdl.continuous_var_dict(time, lb=0, name='V_B1')  
    Q_B2_t          = mdl.continuous_var_dict(time, lb=0, name='Q_B2')
    V_B2_t          = mdl.continuous_var_dict(time, lb=0, name='V_B2')  
    
    dev_pos = mdl.continuous_var(lb=0, name='soc_dev_pos')
    dev_neg = mdl.continuous_var(lb=0, name='soc_dev_neg')
    soc_ref = soc_target if soc_target is not None else soc_init
    
    PUE = 1.15
    Arr_B1  = tier_b_daily  / 24.0      
    Arr_B2  = tier_b2_weekly / 168.0    
    val_A   = tier_a
    val_w   = wind_ts.values
    val_s   = solar_ts.values

    # --- CONSTRAINTS ---
    mdl.add_constraint(E_SOC_t[SOCtime[0]] == soc_init)
    mdl.add_constraint(E_SOC_t[SOCtime[-1]] - soc_ref == dev_pos - dev_neg)

    for i, t in enumerate(time):
        mdl.add_constraint(
            P_HPP_t[t] == (val_A[i] - U_tier_A[t]) + P_tier_B1[t] + P_tier_B2[t] + P_tier_C2[t]
        )
        mdl.add_constraint(
            val_w[i] + val_s[i] + P_discharge[t] ==
            PUE * P_HPP_t[t] + P_charge[t] + P_curtail_t[t]
        )
        mdl.add_constraint(
            (val_A[i] - U_tier_A[t]) + P_tier_B1[t] + P_tier_B2[t] + P_tier_C2[t] <= it_capacity_mw
        )

        # PATCH 2: Removed the `if t != time[-1]:` loophole. Physics must apply to all hours.
        tt = t + pd.Timedelta('1h')
        mdl.add_constraint(
            E_SOC_t[tt] == E_SOC_t[t]
                           + charge_efficiency * P_charge[t] * dt
                           - (1.0/charge_efficiency) * P_discharge[t] * dt
        )

        if i == 0:
            mdl.add_constraint(Q_B1_t[t] == qB_init + Arr_B1[i] - P_tier_B1[t])
            mdl.add_constraint(Q_B2_t[t] == qB2_init + Arr_B2[i] - P_tier_B2[t])
        else:
            prev_t = time[i-1]
            mdl.add_constraint(Q_B1_t[t] == Q_B1_t[prev_t] + Arr_B1[i] - P_tier_B1[t])
            mdl.add_constraint(Q_B2_t[t] == Q_B2_t[prev_t] + Arr_B2[i] - P_tier_B2[t])

        recent_B1 = (np.sum(prev_arrivals_B1[i:]) if i < 23 else 0.0) + np.sum(Arr_B1[max(0, i-23):i+1])
        mdl.add_constraint(V_B1_t[t] >= Q_B1_t[t] - recent_B1)

        recent_B2 = (np.sum(prev_arrivals_B2[i:]) if i < 167 else 0.0) + np.sum(Arr_B2[max(0, i-167):i+1])
        mdl.add_constraint(V_B2_t[t] >= Q_B2_t[t] - recent_B2)

    # --- OBJECTIVE ---
    VoLL_A          = 1.0e8
    Pen_B1_deadline = 1.0e3
    Pen_B2_deadline = 1.0e2
    Pen_latency     = 1.0e-4
    Pen_curt        = 1.0e-2
    Pen_deg         = 1.0e-1
    Reward_C2       = float(os.environ.get('REWARD_C2', '-0.5'))

    obj  = mdl.sum(U_tier_A[t]      for t in time) * VoLL_A
    obj += mdl.sum(V_B1_t[t]        for t in time) * Pen_B1_deadline
    obj += mdl.sum(Q_B1_t[t]        for t in time) * Pen_latency
    obj += mdl.sum(V_B2_t[t]        for t in time) * Pen_B2_deadline
    obj += mdl.sum(Q_B2_t[t]        for t in time) * Pen_latency
    obj += mdl.sum(P_curtail_t[t]   for t in time) * Pen_curt
    obj += mdl.sum(P_charge[t] + P_discharge[t] for t in time) * Pen_deg
    obj += mdl.sum(P_tier_C2[t]     for t in time) * Reward_C2
    obj += w_terminal_soc * (dev_pos + dev_neg)

    mdl.minimize(obj)
    sol = mdl.solve(log_output=False)
    
    if sol is None:
        print("⚠️ WARNING: CPLEX Window Infeasible! Dropping load for this week.")
        zero = np.zeros(len(time))
        E_soc = np.r_[soc_init, zero] 
        mdl.end()
        return (zero, val_w + val_s, zero, E_soc, val_A, zero, zero, zero, zero, zero)

    # PATCH 3: Bulletproof strict index-based extraction prevents array scrambling
    def get_series(var_dict, idx):
        return np.array([sol.get_value(var_dict[k]) for k in idx])

    P_HPP_res   = get_series(P_HPP_t, time)
    P_curt_res  = get_series(P_curtail_t, time)
    P_batt_res  = get_series(P_discharge, time) - get_series(P_charge, time)
    E_SOC_res   = get_series(E_SOC_t, SOCtime)
    U_A_res     = get_series(U_tier_A, time)
    S_B1_res    = get_series(V_B1_t, time)
    S_B2_res    = get_series(V_B2_t, time)
    P_C2_res    = get_series(P_tier_C2, time)
    Q_B1_res    = get_series(Q_B1_t, time)
    Q_B2_res    = get_series(Q_B2_t, time)

    mdl.end()
    return (P_HPP_res, P_curt_res, P_batt_res, E_SOC_res,
            U_A_res, S_B1_res, S_B2_res, P_C2_res,
            Q_B1_res, Q_B2_res)


def solve_rolling(
    wind, solar,                         # numpy arrays, len = T
    tier_a, tier_b_daily, tier_b2_weekly,
    it_cap, P_batt, E_batt, eta, DoD,
    H_p_hours=168, H_c_hours=24,
    soc0=None, w_term_soc=0.05,
    threads=4, timelimit=120, mipgap=0.001
):
    """
    Rolling-horizon wrapper. Slides the window by H_c_hours, optimizes over H_p_hours,
    and carries forward SoC, queues, and arrival memories.
    Returns stitched time series for a full year.
    """
    T = len(wind)
 
    # outputs
    P_HPP = np.zeros(T)
    P_curt = np.zeros(T)
    out_P_batt = np.zeros(T) # <--- CHANGED: Renamed to out_P_batt
    U_A    = np.zeros(T)
    S_B1   = np.zeros(T)
    S_B2   = np.zeros(T)
    P_C2   = np.zeros(T)
    E_soc  = np.zeros(T + 1)
 
    # initial states
    soc  = E_batt * 0.5 if soc0 is None else float(soc0)
    qB1  = 0.0
    qB2  = 0.0
    prev_B1 = np.zeros(23)    # last 23 arrivals for B1
    prev_B2 = np.zeros(167)   # last 167 arrivals for B2
    E_soc[0] = soc
 
    t = 0
    while t < T:
        t_end = min(t + H_p_hours, T)
        # build window series with a simple hourly index
        idx = pd.date_range('2000-01-01', periods=t_end - t, freq='h')
        w_win  = pd.Series(wind[t:t_end],  index=idx)
        s_win  = pd.Series(solar[t:t_end], index=idx)
        A_win  = np.array(tier_a[t:t_end])
        B1_win = np.array(tier_b_daily[t:t_end])
        B2_win = np.array(tier_b2_weekly[t:t_end])
        
        # solve window
        (P_HPP_win, P_curt_win, P_batt_win, E_soc_win,
         U_A_win, S_B1_win, S_B2_win, P_C2_win,
         Q_B1_win, Q_B2_win) = ems_cplex_window(
            w_win, s_win,
            P_batt_MW=P_batt, E_batt_MWh=E_batt,
            battery_depth_of_discharge=DoD, charge_efficiency=eta,
            tier_a=A_win, tier_b_daily=B1_win, tier_b2_weekly=B2_win,
            it_capacity_mw=it_cap,
            soc_init=soc, qB_init=qB1, qB2_init=qB2,
            prev_arrivals_B1=prev_B1, prev_arrivals_B2=prev_B2,
            soc_target=soc, w_terminal_soc=w_term_soc,
            threads=threads, timelimit=timelimit, mipgap=mipgap
        )
 
        # apply first H_c hours
        h_apply = min(H_c_hours, len(P_HPP_win))
        if h_apply == 0:
            break
        sl = slice(t, t + h_apply)
        P_HPP[sl] = P_HPP_win[:h_apply]
        P_curt[sl] = P_curt_win[:h_apply]
        out_P_batt[sl] = P_batt_win[:h_apply] # <--- CHANGED: Updated name here        U_A[sl]    = U_A_win[:h_apply]
        S_B1[sl]   = S_B1_win[:h_apply]
        S_B2[sl]   = S_B2_win[:h_apply]
        P_C2[sl]   = P_C2_win[:h_apply]
        E_soc[sl.start:sl.stop + 1] = E_soc_win[:h_apply + 1]
 
        # advance states to boundary at t+h_apply
        soc = float(E_soc_win[h_apply])
 
        # queues at the cut: take queues at the cut-1 inside the window
        #   (index h_apply-1 in arrays), or 0 if h_apply==0
        if h_apply > 0:
            qB1 = float(max(Q_B1_win[h_apply - 1], 0.0))
            qB2 = float(max(Q_B2_win[h_apply - 1], 0.0))
 
            # update arrival memories with the last h_apply hours of arrivals
            arr_B1_now = (B1_win[:h_apply] / 24.0)
            arr_B2_now = (B2_win[:h_apply] / 168.0)
 
            # keep last 23 / 167 hours
            prev_B1 = np.r_[prev_B1, arr_B1_now][-23:]
            prev_B2 = np.r_[prev_B2, arr_B2_now][-167:]
 
        # slide window
        t += h_apply
 
    return P_HPP, P_curt, out_P_batt, E_soc, U_A, S_B1, S_B2, P_C2 # <--- CHANGED
# -----------------------------
#-------------------------------------------
# 2. FULL HORIZON SOLVER (Queue + C2 Opportunistic)
# -------------------------------------------------------------------------
def ems_cplex_offgrid_full_horizon(
    wind_ts, solar_ts, P_batt_MW, E_batt_MWh_t, 
    battery_depth_of_discharge, charge_efficiency, 
    tier_a_profile, tier_b_profile, tier_b2_profile,
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

    P_tier_B2 = mdl.continuous_var_dict(time, lb=0, name='Served_B2')
    Queue_B2_t = mdl.continuous_var_dict(time, lb=0, name='Queue_B2')
    Missed_Deadline_B2_t = mdl.continuous_var_dict(time, lb=0, name='Missed_Deadline_B2')
    PUE = 1.15 
    
    # Derive hourly arrivals from the input profile (assuming it was repeated daily values)
    Arrival_B = tier_b_profile / 24.0
    Arrival_B2 = tier_b2_profile / 168.0  

    # --- CONSTRAINTS ---
    
    mdl.add_constraint(E_SOC_t[SOCtime[0]] == 0.5 * batt_cap)
    mdl.add_constraint(E_SOC_t[SOCtime[-1]] == 0.5 * batt_cap)

    val_tier_a = tier_a_profile
    val_wind = wind_ts.values
    val_solar = solar_ts.values

    # Main Physics Loop
    for i, t in enumerate(time):
        
        # 1. HPP Output = Served A + Served B + Served C2
        mdl.add_constraint(
            P_HPP_t[t] == (val_tier_a[i] - U_tier_A[t]) + P_tier_B[t] + P_tier_B2[t] + P_tier_C2[t]
        )
        
        # 2. Generation Balance
        mdl.add_constraint(
            val_wind[i] + val_solar[i] + P_discharge[t] 
            == 
            (PUE * P_HPP_t[t]) + P_charge[t] + P_curtailment_t[t]
        )

        # 3. IT CAPACITY LIMIT
        mdl.add_constraint(
            (val_tier_a[i] - U_tier_A[t]) + P_tier_B[t] + P_tier_B2[t] + P_tier_C2[t] <= it_capacity_mw
        )

        # 4. SOC Evolution
        # Remove the if statement entirely! 
        tt = t + pd.Timedelta('1h')
        mdl.add_constraint(
            E_SOC_t[tt] == E_SOC_t[t]
                           + charge_efficiency * P_charge[t] * dt
                           - (1.0/charge_efficiency) * P_discharge[t] * dt
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
        
        # 7. Tier B2 Queue Dynamics
        if i == 0:
            mdl.add_constraint(Queue_B2_t[t] == Arrival_B2[i] - P_tier_B2[t])
        else:
            prev_t = time[i-1]
            mdl.add_constraint(Queue_B2_t[t] == Queue_B2_t[prev_t] + Arrival_B2[i] - P_tier_B2[t])

        # 8. 168-Hour SLA / Deadline Logic
        start_idx_b2 = max(0, i - 167)
        recent_arrivals_b2 = np.sum(Arrival_B2[start_idx_b2 : i+1])
        mdl.add_constraint(Missed_Deadline_B2_t[t] >= Queue_B2_t[t] - recent_arrivals_b2)

   # --- OBJECTIVE ---
    VoLL_Tier_A = 100_000_000   
    VoLL_Tier_B = 1_000    
    VoLL_Tier_B2 = 100          # <-- 10^2 penalty as per your methodology table
    Cost_Queue_Latency = 0.0001 
    Cost_Curtail = 0.01    
    Cost_Deg = 0.1         
    Reward_C2 = float(os.environ.get('REWARD_C2', '-0.5'))
    
    obj_A = mdl.sum(U_tier_A[t] for t in time) * VoLL_Tier_A
    obj_B_Deadline = mdl.sum(Missed_Deadline_B_t[t] for t in time) * VoLL_Tier_B
    obj_B_Latency = mdl.sum(Queue_B_t[t] for t in time) * Cost_Queue_Latency
    
    # <-- ADD B2 OBJECTIVES
    obj_B2_Deadline = mdl.sum(Missed_Deadline_B2_t[t] for t in time) * VoLL_Tier_B2
    obj_B2_Latency = mdl.sum(Queue_B2_t[t] for t in time) * Cost_Queue_Latency 
    
    obj_Curt = mdl.sum(P_curtailment_t[t] for t in time) * Cost_Curtail
    obj_Deg = mdl.sum((P_charge[t] + P_discharge[t]) for t in time) * Cost_Deg
    obj_C2 = mdl.sum(P_tier_C2[t] for t in time) * Reward_C2

    mdl.minimize(obj_A + obj_B_Deadline + obj_B_Latency + obj_B2_Deadline + obj_B2_Latency + obj_Curt + obj_Deg + obj_C2)
    
    # --- SOLVE ---
    print("   ... CPLEX solving 8760 hours with Queue & C2 logic (this may take 30-90s) ...")
    sol = mdl.solve(log_output=False)
    
    if sol is None:
        print("❌ SOLVER FAILED to find integer solution. Returning Zeros.")
        return (np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time)), 
                np.zeros(len(time)+1), tier_a_profile, np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time)))

    print("   ✅ Solution found!")
    
    def get_series(var_dict, idx):
        return np.array([sol.get_value(var_dict[k]) for k in idx])

    P_HPP_res  = get_series(P_HPP_t, time)
    P_curt_res = get_series(P_curtailment_t, time)
    P_batt_res = get_series(P_discharge, time) - get_series(P_charge, time)
    
    E_SOC_res  = get_series(E_SOC_t, SOCtime)
    
    U_A_res    = get_series(U_tier_A, time)
    P_C2_res   = get_series(P_tier_C2, time) 
    
    S_B_res    = get_series(Missed_Deadline_B_t, time)
    S_B2_res   = get_series(Missed_Deadline_B2_t, time)

    Penalty_res = U_A_res * VoLL_Tier_A + S_B_res * VoLL_Tier_B + S_B2_res * VoLL_Tier_B2
    mdl.end()
    
    return P_HPP_res, P_curt_res, P_batt_res, E_SOC_res, U_A_res, S_B_res, S_B2_res, P_C2_res, Penalty_res

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