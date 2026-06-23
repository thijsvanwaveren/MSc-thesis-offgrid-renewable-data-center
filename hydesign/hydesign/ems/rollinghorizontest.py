# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:15:47 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Rolling Horizon EMS with Corrected Extractions and Numeric Stability
"""

import os
import numpy as np
import pandas as pd
import openmdao.api as om
from docplex.mp.model import Model
from hydesign.ems.ems import expand_to_lifetime

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
        H_c_hours = 24,          # daily control horizon
        w_terminal_soc = 0.05,   # terminal soc penalty weight
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
        tier_b2_1yr = inputs['tier_b2_profile'][:self.N_time] 
        
        WSPr_df = pd.DataFrame(index=pd.date_range(start='01-01-1991 00:00', periods=len(wind_t), freq='h'))
        WSPr_df['wind_t'] = wind_t
        WSPr_df['solar_t'] = solar_t
        WSPr_df['price_t'] = price_t
        WSPr_df['E_batt_MWh_t'] = b_E[0]
        
        # --- CALL CPLEX OPTIMIZER ---
        if self.run_mode == 'rolling':
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
                H_p_hours=self.H_p_hours,
                H_c_hours=self.H_c_hours,
                soc0=b_E[0]*0.5,
                w_term_soc=self.w_terminal_soc,
                threads=4, timelimit=120
            )
            penalty_ts = (Unserved_A_ts * 1.0e5) + (Shortfall_B_ts * 1.0e3) + (Shortfall_B2_ts * 1.0e2)  
        else:
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
        
        # Expand Results to Lifetime (FIXED P_curt_ts BUG HERE)
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

# -------------------------------------------------------------------------
# ROLLING HORIZON SOLVER
# -------------------------------------------------------------------------
def ems_cplex_window(
    wind_ts, solar_ts,                     
    P_batt_MW, E_batt_MWh,                 
    battery_depth_of_discharge, charge_efficiency,
    tier_a, tier_b_daily, tier_b2_weekly,  
    it_capacity_mw,
    soc_init,                              
    qB_init, qB2_init,                     
    prev_arrivals_B1, prev_arrivals_B2,    
    soc_target=None, w_terminal_soc=0.0,
    threads=4, timelimit=120, mipgap=0.001
):
    mdl = Model(name='EMS_Window')
    mdl.context.cplex_parameters.threads = threads
    mdl.context.cplex_parameters.timelimit = timelimit
    mdl.context.cplex_parameters.mip.tolerances.mipgap = mipgap
    # Default lpmethod is more stable than forced barrier for numerically dense penalties
 
    time = wind_ts.index
    dt = 1.0
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1h')]))

    # BOUND CLAMPING
    batt_cap = float(E_batt_MWh)
    min_soc  = (1.0 - battery_depth_of_discharge) * batt_cap
    soc_init = max(min_soc, min(batt_cap, float(soc_init)))
    qB_init = max(0.0, float(qB_init))
    qB2_init = max(0.0, float(qB2_init))
 
    # VARIABLES
    P_HPP_t         = mdl.continuous_var_dict(time, lb=0, name='P_HPP')
    P_curtail_t     = mdl.continuous_var_dict(time, lb=0, name='Curt')
    P_charge        = mdl.continuous_var_dict(time, lb=0, ub=(P_batt_MW / charge_efficiency), name='Ch')
    P_discharge     = mdl.continuous_var_dict(time, lb=0, ub=(P_batt_MW * charge_efficiency), name='Dis')
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
    val_A   = np.array(tier_a)
    val_w   = np.array(wind_ts.values)
    val_s   = np.array(solar_ts.values)
 
    # CONSTRAINTS
    mdl.add_constraint(E_SOC_t[SOCtime[0]] == soc_init)
    mdl.add_constraint(E_SOC_t[SOCtime[-1]] - soc_ref == dev_pos - dev_neg)
 
    for i, t in enumerate(time):
        # Strict physical limit: You cannot un-serve more load than you requested
        mdl.add_constraint(U_tier_A[t] <= val_A[i])

        mdl.add_constraint(P_HPP_t[t] == (val_A[i] - U_tier_A[t]) + P_tier_B1[t] + P_tier_B2[t] + P_tier_C2[t])
        mdl.add_constraint(val_w[i] + val_s[i] + P_discharge[t] == PUE * P_HPP_t[t] + P_charge[t] + P_curtail_t[t])
        mdl.add_constraint((val_A[i] - U_tier_A[t]) + P_tier_B1[t] + P_tier_B2[t] + P_tier_C2[t] <= it_capacity_mw)
 
        tt = SOCtime[i+1]
        mdl.add_constraint(
            E_SOC_t[tt] == E_SOC_t[t] + charge_efficiency * P_charge[t] * dt - (1.0/charge_efficiency) * P_discharge[t] * dt
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
 
    # OBJECTIVE (Scaled for numerical matrix stability)
    obj  = mdl.sum(U_tier_A[t] for t in time) * 1.0e5
    obj += mdl.sum(V_B1_t[t] for t in time) * 1.0e3
    obj += mdl.sum(Q_B1_t[t] for t in time) * 1.0e-2
    obj += mdl.sum(V_B2_t[t] for t in time) * 1.0e2
    obj += mdl.sum(Q_B2_t[t] for t in time) * 1.0e-2
    obj += mdl.sum(P_curtail_t[t] for t in time) * 1.0e-1
    obj += mdl.sum(P_charge[t] + P_discharge[t] for t in time) * 5.0e-1
    obj += mdl.sum(P_tier_C2[t] for t in time) * float(os.environ.get('REWARD_C2', '-0.5'))
    obj += w_terminal_soc * (dev_pos + dev_neg)
 
    mdl.minimize(obj)
    sol = mdl.solve(log_output=False)
    
    # Strictly enforce fallback if matrix fails or is infeasible
    if sol is None or not mdl.solve_details.status.startswith('opt'):
        print("⚠️ WARNING: CPLEX Window Infeasible or Failed! Dropping load for this week.")
        zero = np.zeros(len(time))
        E_soc = np.r_[soc_init, zero] 
        mdl.end()
        return (zero, val_w + val_s, zero, E_soc, val_A, zero, zero, zero, zero, zero)
 
    def extract(var_dict, index=time):
        return np.array([sol.get_value(var_dict[t]) for t in index])
 
    P_HPP_res   = extract(P_HPP_t)
    P_curt_res  = extract(P_curtail_t)
    P_ch_res    = extract(P_charge)
    P_dis_res   = extract(P_discharge)
    P_batt_res  = P_dis_res - P_ch_res
    E_SOC_res   = extract(E_SOC_t, index=SOCtime)
    U_A_res     = extract(U_tier_A)
    P_C2_res    = extract(P_tier_C2)
    S_B1_res    = extract(V_B1_t)
    S_B2_res    = extract(V_B2_t)
    Q_B1_res    = extract(Q_B1_t)
    Q_B2_res    = extract(Q_B2_t)
 
    mdl.end()
    return (P_HPP_res, P_curt_res, P_batt_res, E_SOC_res, U_A_res, S_B1_res, S_B2_res, P_C2_res, Q_B1_res, Q_B2_res)

def solve_rolling(
    wind, solar,                                 
    tier_a, tier_b_daily, tier_b2_weekly,
    it_cap, P_batt, E_batt, eta, DoD,
    H_p_hours=168, H_c_hours=24,
    soc0=None, w_term_soc=0.05,
    threads=4, timelimit=120, mipgap=0.001
):
    T = len(wind)
    P_HPP = np.zeros(T)
    P_curt = np.zeros(T)
    out_P_batt = np.zeros(T)
    U_A    = np.zeros(T)
    S_B1   = np.zeros(T)
    S_B2   = np.zeros(T)
    P_C2   = np.zeros(T)
    E_soc  = np.zeros(T + 1)
 
    soc  = E_batt * 0.5 if soc0 is None else float(soc0)
    qB1  = 0.0
    qB2  = 0.0
    prev_B1 = np.zeros(23)    
    prev_B2 = np.zeros(167)   
    E_soc[0] = soc
 
    t = 0
    while t < T:
        t_end = min(t + H_p_hours, T)
        idx = pd.date_range('2000-01-01', periods=t_end - t, freq='h')
        w_win  = pd.Series(wind[t:t_end],  index=idx)
        s_win  = pd.Series(solar[t:t_end], index=idx)
        A_win  = np.array(tier_a[t:t_end])
        B1_win = np.array(tier_b_daily[t:t_end])
        B2_win = np.array(tier_b2_weekly[t:t_end])
 
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
 
        h_apply = min(H_c_hours, len(P_HPP_win))
        if h_apply == 0:
            break
        sl = slice(t, t + h_apply)
        P_HPP[sl] = P_HPP_win[:h_apply]
        P_curt[sl] = P_curt_win[:h_apply]
        out_P_batt[sl] = P_batt_win[:h_apply]
        U_A[sl]    = U_A_win[:h_apply]
        S_B1[sl]   = S_B1_win[:h_apply]
        S_B2[sl]   = S_B2_win[:h_apply]
        P_C2[sl]   = P_C2_win[:h_apply]
        E_soc[sl.start:sl.stop + 1] = E_soc_win[:h_apply + 1]
 
        soc = float(E_soc_win[h_apply])
        if h_apply > 0:
            qB1 = float(max(Q_B1_win[h_apply - 1], 0.0))
            qB2 = float(max(Q_B2_win[h_apply - 1], 0.0))
            arr_B1_now = (B1_win[:h_apply] / 24.0)
            arr_B2_now = (B2_win[:h_apply] / 168.0)
            prev_B1 = np.r_[prev_B1, arr_B1_now][-23:]
            prev_B2 = np.r_[prev_B2, arr_B2_now][-167:]
 
        t += h_apply
 
    return P_HPP, P_curt, out_P_batt, E_soc, U_A, S_B1, S_B2, P_C2

# -------------------------------------------------------------------------
# FULL HORIZON SOLVER
# -------------------------------------------------------------------------
def ems_cplex_offgrid_full_horizon(
    wind_ts, solar_ts, P_batt_MW, E_batt_MWh_t, 
    battery_depth_of_discharge, charge_efficiency, 
    tier_a_profile, tier_b_profile, tier_b2_profile,
    it_capacity_mw
):
    mdl = Model(name='EMS_OffGrid_FullYear')
    mdl.context.cplex_parameters.threads = 4 
    mdl.context.cplex_parameters.timelimit = 180 
    mdl.context.cplex_parameters.mip.tolerances.mipgap = 0.01 
    
    time = wind_ts.index
    dt = 1.0
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))

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

    Queue_B_t = mdl.continuous_var_dict(time, lb=0, name='Queue_B')
    Missed_Deadline_B_t = mdl.continuous_var_dict(time, lb=0, name='Missed_Deadline_B')

    P_tier_B2 = mdl.continuous_var_dict(time, lb=0, name='Served_B2')
    Queue_B2_t = mdl.continuous_var_dict(time, lb=0, name='Queue_B2')
    Missed_Deadline_B2_t = mdl.continuous_var_dict(time, lb=0, name='Missed_Deadline_B2')
    PUE = 1.15 
    
    Arrival_B = tier_b_profile / 24.0
    Arrival_B2 = tier_b2_profile / 168.0  

    mdl.add_constraint(E_SOC_t[SOCtime[0]] == 0.5 * batt_cap)
    mdl.add_constraint(E_SOC_t[SOCtime[-1]] == 0.5 * batt_cap)

    val_tier_a = tier_a_profile
    val_wind = wind_ts.values
    val_solar = solar_ts.values

    for i, t in enumerate(time):
        mdl.add_constraint(U_tier_A[t] <= val_tier_a[i])
        mdl.add_constraint(P_HPP_t[t] == (val_tier_a[i] - U_tier_A[t]) + P_tier_B[t] + P_tier_B2[t] + P_tier_C2[t])
        mdl.add_constraint(val_wind[i] + val_solar[i] + P_discharge[t] == (PUE * P_HPP_t[t]) + P_charge[t] + P_curtailment_t[t])
        mdl.add_constraint((val_tier_a[i] - U_tier_A[t]) + P_tier_B[t] + P_tier_B2[t] + P_tier_C2[t] <= it_capacity_mw)

        if t != time[-1]:
            tt = t + pd.Timedelta("1h")
            mdl.add_constraint(E_SOC_t[tt] == E_SOC_t[t] + (P_charge[t] * charge_efficiency * dt) - (P_discharge[t] / charge_efficiency * dt))

        if i == 0:
            mdl.add_constraint(Queue_B_t[t] == Arrival_B[i] - P_tier_B[t])
        else:
            prev_t = time[i-1]
            mdl.add_constraint(Queue_B_t[t] == Queue_B_t[prev_t] + Arrival_B[i] - P_tier_B[t])

        start_idx = max(0, i - 23)
        recent_arrivals = np.sum(Arrival_B[start_idx : i+1])
        mdl.add_constraint(Missed_Deadline_B_t[t] >= Queue_B_t[t] - recent_arrivals)
        
        if i == 0:
            mdl.add_constraint(Queue_B2_t[t] == Arrival_B2[i] - P_tier_B2[t])
        else:
            prev_t = time[i-1]
            mdl.add_constraint(Queue_B2_t[t] == Queue_B2_t[prev_t] + Arrival_B2[i] - P_tier_B2[t])

        start_idx_b2 = max(0, i - 167)
        recent_arrivals_b2 = np.sum(Arrival_B2[start_idx_b2 : i+1])
        mdl.add_constraint(Missed_Deadline_B2_t[t] >= Queue_B2_t[t] - recent_arrivals_b2)

    VoLL_Tier_A = 1000000
    VoLL_Tier_B = 100    
    VoLL_Tier_B2 = 10          
    Cost_Queue_Latency = 0.01 
    Cost_Curtail = 0.1    
    Cost_Deg = 0.5       
    Reward_C2 = float(os.environ.get('REWARD_C2', '-0.5'))
    
    obj_A = mdl.sum(U_tier_A[t] for t in time) * VoLL_Tier_A
    obj_B_Deadline = mdl.sum(Missed_Deadline_B_t[t] for t in time) * VoLL_Tier_B
    obj_B_Latency = mdl.sum(Queue_B_t[t] for t in time) * Cost_Queue_Latency
    obj_B2_Deadline = mdl.sum(Missed_Deadline_B2_t[t] for t in time) * VoLL_Tier_B2
    obj_B2_Latency = mdl.sum(Queue_B2_t[t] for t in time) * Cost_Queue_Latency 
    obj_Curt = mdl.sum(P_curtailment_t[t] for t in time) * Cost_Curtail
    obj_Deg = mdl.sum((P_charge[t] + P_discharge[t]) for t in time) * Cost_Deg
    obj_C2 = mdl.sum(P_tier_C2[t] for t in time) * Reward_C2

    mdl.minimize(obj_A + obj_B_Deadline + obj_B_Latency + obj_B2_Deadline + obj_B2_Latency + obj_Curt + obj_Deg + obj_C2)
    print("   ... CPLEX solving 8760 hours ...")
    sol = mdl.solve(log_output=False)
    
    if sol is None:
        print("❌ SOLVER FAILED to find integer solution. Returning Zeros.")
        return (np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time)), 
                np.zeros(len(time)+1), tier_a_profile, np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time)))

    def extract(var_dict):
        return np.array([sol.get_value(var_dict[t]) for t in time])

    P_HPP_res = extract(P_HPP_t)
    P_curt_res = extract(P_curtailment_t)
    P_ch_res = extract(P_charge)
    P_dis_res = extract(P_discharge)
    P_batt_res = P_dis_res - P_ch_res 

    E_SOC_res = np.array([sol.get_value(E_SOC_t[t]) for t in SOCtime])
    U_A_res = extract(U_tier_A)
    P_C2_res = extract(P_tier_C2) 
    S_B_res = extract(Missed_Deadline_B_t)
    S_B2_res = extract(Missed_Deadline_B2_t)  

    Penalty_res = U_A_res * VoLL_Tier_A + S_B_res * VoLL_Tier_B + S_B2_res * VoLL_Tier_B2
    mdl.end()
    
    return P_HPP_res, P_curt_res, P_batt_res, E_SOC_res, U_A_res, S_B_res, S_B2_res, P_C2_res, Penalty_res