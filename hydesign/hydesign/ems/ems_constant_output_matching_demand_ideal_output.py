import numpy as np
import pandas as pd
import openmdao.api as om
import os

from docplex.mp.model import Model
from hydesign.ems.ems import expand_to_lifetime, split_in_batch



class ems_long_term_operation(om.ExplicitComponent):
    def __init__(
        self, 
        N_time,
        num_batteries = 1,
        life_y = 25,
        life_h = 25*365*24,
        intervals_per_hour = 1,
        ems_type='energy_penalty',
        load_min_penalty_factor=1e6,
        ):

        super().__init__()
        self.N_time = N_time
        self.life_h = 365 * 24 * life_y
        self.life_intervals = self.life_h * intervals_per_hour
        self.ems_type = ems_type
        self.load_min_penalty_factor = load_min_penalty_factor


    def setup(self):
        
        self.add_input(
            'SoH',
            desc="Battery state of health at discretization levels",
            shape=[self.life_h])
        self.add_input(
            'wind_t_ext_deg',
            desc="Wind time series including degradation",
            units='MW',
            shape=[self.life_h]) 
        self.add_input(
            'solar_t_ext_deg',
            desc="PV time series including degradation",
            units='MW',
            shape=[self.life_h]) 
        self.add_input(
            'wind_t_ext',
            desc="WPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'solar_t_ext',
            desc="PVP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'price_t_ext',
            desc="Electricity price time series",
            shape=[self.life_h])
        self.add_input(
            'b_P',
            desc="Battery power capacity",
            units='MW')
        self.add_input(
            'b_E',
            desc="Battery energy storage capacity")
        self.add_input(
            'G_MW',
            desc="Grid capacity",
            units='MW')
        self.add_input(
            'battery_depth_of_discharge',
            desc="battery depth of discharge",
            units='MW')
        self.add_input(
            'battery_charge_efficiency',
            desc="battery charge efficiency")
        self.add_input(
            'hpp_curt_t',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'b_t',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'b_E_SOC_t',
            desc="Battery energy SOC time series",
            shape=[self.life_h + 1])
        self.add_input(
            'peak_hr_quantile',
            desc="Quantile of price tim sereis to define peak price hours (above this quantile).\n"+
                 "Only used for peak production penalty and for cost of battery degradation.")
        self.add_input(
            'n_full_power_hours_expected_per_day_at_peak_price',
            desc="Pnealty occurs if nunmber of full power hours expected per day at peak price are not reached.")
        self.add_input(
            'load_profile_ts',
            desc="Hourly load profile to be satisfied [MW]",
            shape=[self.life_h],
            units='MW')
        self.add_input(
        'hpp_t',
        desc="Ideal HPP power time series from the optimizer",
        units='MW',
        shape=[self.life_h])
#------------------------------------------------------------------------------------------
        self.add_output(
            'hpp_t_with_deg',
            desc="HPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'hpp_curt_t_with_deg',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'b_t_with_deg',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'b_E_SOC_t_with_deg',
            desc="Battery energy SOC time series",
            shape=[self.life_h + 1])
        self.add_output(
            'penalty_t_with_deg',
            desc="penalty for not reaching expected energy productin at peak hours",
            shape=[self.life_h])   
        self.add_output(
            'total_curtailment',
            desc="total curtailment in the lifetime",
            units='GW*h',
           )
        self.add_output(
            'wind_t_deg',
            desc="Wind time series including degradation",
            units='MW',
            shape=[self.life_h])  
        self.add_output(
            'solar_t_deg',
            desc="PV time series including degradation",
            units='MW',
            shape=[self.life_h])  

        
    def compute(self, inputs, outputs):
        
        SoH = inputs['SoH']
        wind_t_ext_deg = inputs['wind_t_ext_deg']
        solar_t_ext_deg = inputs['solar_t_ext_deg']
        hpp_t = inputs['hpp_t']
        wind_t_ext = inputs['wind_t_ext']
        solar_t_ext = inputs['solar_t_ext']
        price_t_ext = inputs['price_t_ext']
        b_P = inputs['b_P']
        b_E = inputs['b_E']
        G_MW = inputs['G_MW']
        battery_depth_of_discharge = inputs['battery_depth_of_discharge']
        battery_charge_efficiency = inputs['battery_charge_efficiency']
        hpp_curt_t = inputs['hpp_curt_t']
        b_t = inputs['b_t']
        b_E_SOC_t = inputs['b_E_SOC_t']
        load_profile_ts = inputs['load_profile_ts']

        time_index = pd.date_range(
            start='1991-01-01 00:00',
            periods=len(inputs['load_profile_ts']),
            freq='1h')
        load_profile_ts = pd.Series(
            inputs['load_profile_ts'],
            index=time_index)

        peak_hr_quantile = inputs['peak_hr_quantile'][0]
        n_full_power_hours_expected_per_day_at_peak_price = inputs['n_full_power_hours_expected_per_day_at_peak_price'][0]

        life_h = self.life_h
        
        if self.ems_type == 'constant_output':
            ems_longterm = operation_rule_base_no_penalty
        else:
            raise Warning("This class should only be used for constant_output")

        args = dict(wind_t_deg = wind_t_ext_deg,
            solar_t_deg = solar_t_ext_deg,
            batt_degradation = SoH,
            wind_t = wind_t_ext,
            solar_t = solar_t_ext,
            hpp_curt_t = hpp_curt_t,
            b_t = b_t,
            b_E_SOC_t = b_E_SOC_t,
            hpp_t = hpp_t,
            G_MW = G_MW[0],
            b_E = b_E[0],
            battery_depth_of_discharge = battery_depth_of_discharge[0],
            battery_charge_efficiency = battery_charge_efficiency[0],
            b_E_SOC_0 = None,
            load_profile_ts = load_profile_ts,
            load_min_penalty_factor = self.load_min_penalty_factor,)
        Hpp_deg, P_curt_deg, b_t_sat, b_E_SOC_t_sat, penalty_t_with_deg = ems_longterm(**args)
            
        outputs['hpp_t_with_deg'] = Hpp_deg
        outputs['hpp_curt_t_with_deg'] = P_curt_deg
        outputs['b_t_with_deg'] = b_t_sat
        outputs['b_E_SOC_t_with_deg'] = b_E_SOC_t_sat
        outputs['penalty_t_with_deg'] = penalty_t_with_deg
        outputs['total_curtailment'] = P_curt_deg.sum()
        outputs['wind_t_deg'] = wind_t_ext_deg
        outputs['solar_t_deg'] = solar_t_ext_deg


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
        self.add_input(
            'wind_t',
            desc="WPP power time series",
            units='MW',
            shape=[self.N_time])
        self.add_input(
            'solar_t',
            desc="PVP power time series",
            units='MW',
            shape=[self.N_time])
        self.add_input(
            'price_t',
            desc="Electricity price time series",
            shape=[self.N_time])
        self.add_input(
            'b_P',
            desc="Battery power capacity",
            units='MW')
        self.add_input(
            'b_E',
            desc="Battery energy storage capacity")
        self.add_input(
            'G_MW',
            desc="Grid capacity",
            units='MW')
        self.add_input(
            'battery_depth_of_discharge',
            desc="battery depth of discharge",
            units='MW')
        self.add_input(
            'battery_charge_efficiency',
            desc="battery charge efficiency")
        self.add_input(
            'peak_hr_quantile',
            desc="Quantile of price tim sereis to define peak price hours (above this quantile).\n"+
                 "Only used for peak production penalty and for cost of battery degradation.")
        self.add_input(
            'cost_of_battery_P_fluct_in_peak_price_ratio',
            desc="cost of battery power fluctuations computed as a peak price ratio.")
        self.add_input(
            'n_full_power_hours_expected_per_day_at_peak_price',
            desc="Pnealty occurs if nunmber of full power hours expected per day at peak price are not reached.")
        self.add_input(
            'load_profile_ts',
            desc="Hourly load profile to be satisfied [MW]",
            shape=[self.life_h],
            units='MW')
# ----------------------------------------------------------------------------------------------------------
        self.add_output(
            'wind_t_ext',
            desc="WPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'solar_t_ext',
            desc="PVP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'price_t_ext',
            desc="Electricity price time series",
            shape=[self.life_h])
        self.add_output(
            'hpp_t',
            desc="HPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'hpp_curt_t',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'b_t',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'b_E_SOC_t',
            desc="Battery energy SOC time series",
            shape=[self.life_h + 1])
        self.add_output(
            'penalty_t',
            desc="penalty for not reaching expected energy productin at peak hours",
            shape=[self.life_h]) 


    def compute(self, inputs, outputs):
    # Gather the input variables     
        wind_t = inputs['wind_t']
        solar_t = inputs['solar_t']
        price_t = inputs['price_t']
        #load_profile_ts = inputs['load_profile_ts'] * p

        time_index = pd.date_range(
            start='1991-01-01 00:00',
            periods=len(inputs['load_profile_ts']),
            freq='1h')
        load_profile_ts = pd.Series(
            inputs['load_profile_ts'],
            index=time_index)

        # time_index = pd.date_range(start='1991-01-01 00:00', periods=len(inputs['load_profile_ts']), freq='1h')
        # load_profile_ts = pd.Series(inputs['load_profile_ts'] * p, index=time_index)
        b_P = inputs['b_P']
        b_E = inputs['b_E']
        G_MW = inputs['G_MW']
    
    # CHoose the EMS solver function 
        if self.ems_type == 'cplex':
            ems_WSB = ems_cplex_constantoutput
        else:
            raise Warning("This class should only be used for ems_cplex_constantoutput")
    
    # Read more scalar inputs 
        battery_depth_of_discharge = inputs['battery_depth_of_discharge']
        battery_charge_efficiency = inputs['battery_charge_efficiency']
        peak_hr_quantile = inputs['peak_hr_quantile'][0]
        cost_of_battery_P_fluct_in_peak_price_ratio = inputs['cost_of_battery_P_fluct_in_peak_price_ratio'][0]
        n_full_power_hours_expected_per_day_at_peak_price = inputs[
            'n_full_power_hours_expected_per_day_at_peak_price'][0]
        # Build a sintetic time to avoid problems with time sereis 
        # indexing in ems

    # Build data frame to hold time series 
        WSPr_df = pd.DataFrame(
            index=pd.date_range(
                start='01-01-1991 00:00',
                periods=len(wind_t),
                freq='1h'))
    # Populate the data  frame 
        WSPr_df['wind_t'] = wind_t
        WSPr_df['solar_t'] = solar_t
        WSPr_df['price_t'] = price_t
        WSPr_df['E_batt_MWh_t'] = b_E[0]
        
        #print(WSPr_df.head())

    # Run the EMS (optimization/dispatch logic)
        P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts = ems_WSB(
            wind_ts = WSPr_df.wind_t,
            solar_ts = WSPr_df.solar_t,
            price_ts = WSPr_df.price_t,
            P_batt_MW = b_P[0],
            E_batt_MWh_t = WSPr_df.E_batt_MWh_t,
            hpp_grid_connection = G_MW[0],
            battery_depth_of_discharge = battery_depth_of_discharge[0],
            charge_efficiency = battery_charge_efficiency[0],
            peak_hr_quantile = peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio = cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price = n_full_power_hours_expected_per_day_at_peak_price,
            load_profile_ts = load_profile_ts,
            load_min_penalty_factor=self.load_min_penalty_factor,
        )

        # Extend (by repeating them and stacking) all variable to full lifetime 
        outputs['wind_t_ext'] = expand_to_lifetime(
            wind_t, life = self.life_intervals, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['solar_t_ext'] = expand_to_lifetime(
            solar_t, life = self.life_intervals, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['price_t_ext'] = expand_to_lifetime(
            price_t, life = self.life_intervals, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['hpp_t'] = expand_to_lifetime(
            P_HPP_ts, life = self.life_intervals, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['hpp_curt_t'] = expand_to_lifetime(
            P_curtailment_ts, life = self.life_intervals, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['b_t'] = expand_to_lifetime(
            P_charge_discharge_ts, life = self.life_intervals, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['b_E_SOC_t'] = expand_to_lifetime(
            E_SOC_ts, life = self.life_intervals + 1, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['penalty_t'] = expand_to_lifetime(
            penalty_ts, life = self.life_intervals, weeks_per_season_per_year = self.weeks_per_season_per_year)        


def ems_cplex_constantoutput(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    load_profile_ts,
    peak_hr_quantile = 0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio = 0.5, #[0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price = 3,    
    batch_size = 24,  # could be as large as 4*24 (note this EMS needs a whole number x 24 to work) but this don't improve performance
    #load_min=3,
    load_min_penalty_factor=1e6,
):

# Split the time series into  batches   
    # split in batches, ussually a week
    batches_all = split_in_batch(list(range(len(wind_ts))), batch_size)
 # Make sure the last batch is not smaller than the others
    # instead append it to the previous last one
    batches = batches_all[:-1]
    batches[-1] = batches_all[-2]+batches_all[-1]
    
# Allocate empty arrays to store results
    # allocate vars
    P_HPP_ts = np.zeros(len(wind_ts))
    P_curtailment_ts = np.zeros(len(wind_ts))
    P_charge_discharge_ts = np.zeros(len(wind_ts))
    E_SOC_ts = np.zeros(len(wind_ts)+1)
    penalty_ts = np.zeros(len(wind_ts))
    
# Loop over each batch and optimize 
    for ib, batch in enumerate(batches):
#  Select the current chunck of data 
        wind_ts_sel = wind_ts.iloc[batch]
        solar_ts_sel = solar_ts.iloc[batch]
        price_ts_sel = price_ts.iloc[batch]
        E_batt_MWh_t_sel = E_batt_MWh_t.iloc[batch]
# Rus the  EMS optimizatio for this batch    
        #print(f'batch {ib+1} out of {len(batches)}')
        P_HPP_ts_batch, P_curtailment_ts_batch, P_charge_discharge_ts_batch,\
        E_SOC_ts_batch, penalty_batch = ems_cplex_parts_constantoutput(
            wind_ts = wind_ts_sel,
            solar_ts = solar_ts_sel,
            price_ts = price_ts_sel,
            P_batt_MW = P_batt_MW,
            E_batt_MWh_t = E_batt_MWh_t_sel,
            hpp_grid_connection = hpp_grid_connection,
            battery_depth_of_discharge = battery_depth_of_discharge,
            charge_efficiency = charge_efficiency,
            peak_hr_quantile = peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio = cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price = n_full_power_hours_expected_per_day_at_peak_price,
            load_profile_ts = load_profile_ts,
            load_min_penalty_factor=load_min_penalty_factor,
        )

# Save the batch results in the full output arrays 
        P_HPP_ts[batch] = P_HPP_ts_batch
        P_curtailment_ts[batch] = P_curtailment_ts_batch
        P_charge_discharge_ts[batch] = P_charge_discharge_ts_batch
        E_SOC_ts[batch] = E_SOC_ts_batch[:-1]
        penalty_ts[batch] = penalty_batch

# close the loop in SoC, start SoC = end SoC
    E_SOC_ts[-1] = E_SOC_ts[0] 
    
    return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts

def ems_cplex_parts_constantoutput(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    load_profile_ts,
    peak_hr_quantile = 0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio = 0.5, #[0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price = 3,
    load_min_penalty_factor = 1e26,
):
    """EMS solver implemented in cplex
    """

# Basic time related variables 
    # Penalties 
    N_t = len(price_ts.index) 
    N_days = N_t/24

# Initialize CPLEX model - Set solver parameters (e.g., focus on feasibility over optimality)  
    mdl = Model(name='EMS')
    mdl.context.cplex_parameters.threads = 1
    mdl.context.cplex_parameters.emphasis.mip = 1 
 
    time = price_ts.index

    load_profile_ts = load_profile_ts.reindex(time)
    assert load_profile_ts.notna().all(), "some of your timestamps didn't line up!"

    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))

# Define time index for SOC 
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))

# Define decision variables 
    P_HPP_t = mdl.continuous_var_dict(
        time, lb=0, ub= hpp_grid_connection, 
        name='HPP power output')
    P_curtailment_t = mdl.continuous_var_dict(
        time, lb=0, 
        name='Curtailment')
    P_charge_discharge = mdl.continuous_var_dict(
        time, lb=-P_batt_MW/charge_efficiency, ub=P_batt_MW*charge_efficiency, 
        name='Battery power')
    E_SOC_t = mdl.continuous_var_dict(
        SOCtime, lb=0, ub=E_batt_MWh_t.max(), 
        name='Energy level')

    U_t = mdl.continuous_var_dict(time, lb=0, name='under_prod')
    O_t = mdl.continuous_var_dict(time, lb=0, name='over_prod')

    slack = mdl.continuous_var_dict(time, lb=0, name='slack')

    P_charge   = mdl.continuous_var_dict(time, lb=0, ub=P_batt_MW, name='P_charge')
    P_discharge = mdl.continuous_var_dict(time, lb=0, ub=P_batt_MW, name='P_discharge')

# Penalty terms
    penalty = mdl.continuous_var(name='penalty', lb=1e12)
    penalty_under = mdl.continuous_var(name='penalty_under', lb=0)
    penalty_above = mdl.continuous_var(name='penalty_above', lb=0)
    x = mdl.continuous_var_dict(time, lb=0, name='dummy for batt disch')
    curtail_penalty_factor = 1e6
    # penalty_curt = mdl.continuous_var(name='penalty_curt', lb=0)
    # fneg(x) returns only negative values (used to compute penalties)
    fneg = mdl.piecewise(-1, [(0,0)], 0)
    slack_penalty_factor = 1e6
# Penalty constraits 
    # under production penalty, charges only when P_HPP is below the load profile 
    mdl.add_constraint(penalty_under== mdl.sum(load_min_penalty_factor* fneg(P_HPP_t[t] - load_profile_ts.iloc[ time.get_loc(t) ])for t in time))
    # over production penalty, charges only when P_HPP is above the load profile
    mdl.add_constraint(penalty_above== mdl.sum(load_min_penalty_factor* fneg(load_profile_ts.iloc[ time.get_loc(t) ] - P_HPP_t[t])for t in time))
    # Curtailment costs 
    curtail_cost = mdl.sum(curtail_penalty_factor * P_curtailment_t[t] for t in time)


# Mismatch 
    mismatch = mdl.sum((load_profile_ts.iloc[ time.get_loc(t) ] - P_HPP_t[t]) for t in time)

# Revenue
    revenue = mdl.sum(price_ts[t] * P_HPP_t[t] for t in time)
    fac_rev = 1e3

# Curtailment 
    curt = mdl.sum(P_curtailment_t[t] for t in time)
    fac_curt = 1e6

# Terminal constraints      
    # Intitial and end SOC
    mdl.add_constraint( E_SOC_t[SOCtime[0]] == 0.5 * E_batt_MWh_t[time[0]] )
    
    # SOC at the end of the year has to be equal to SOC at the beginning of the year
    mdl.add_constraint( E_SOC_t[SOCtime[-1]] == 0.5 * E_batt_MWh_t[time[0]] )

    # pircewise linear representation of charge vs dischrage effciency 
    f2 = mdl.piecewise(charge_efficiency,[(0,0)],1/charge_efficiency)
    # PWL function to model a unit step function
    f3 = mdl.piecewise(0, [(0, 0), (0, 1)], 0)


    dt = 1.0
    for t in time:
        # 1) pull the load at time t by label, not by position
        lp = load_profile_ts.loc[t]
        
        # 2) compute the “next” timestamp (or None on the last step)
        if t != time[-1]:
            tt = t + pd.Timedelta("1h")  
        else:
            tt = None



        # power balance
        mdl.add_constraint(
            P_HPP_t[t]
            == wind_ts[t]
            + solar_ts[t]
            + P_charge_discharge[t]
            - P_curtailment_t[t]
        )

        # your meet-minimum‐load slack constraints
        mdl.add_constraint(P_HPP_t[t] >= lp - slack[t])
        mdl.add_constraint(P_HPP_t[t] <= lp + slack[t])
        #mdl.add_constraint(P_HPP_t[t] >= lp)

        # SoC update (skip on last timestamp)
        if tt is not None:
            mdl.add_constraint(
                E_SOC_t[tt]
                == E_SOC_t[t]
                - f2(P_charge_discharge[t]) * dt
            )

        # battery energy & power bounds
        # Constraining battery energy level to minimum battery level        
        mdl.add_constraint(E_SOC_t[t] >= (1 - battery_depth_of_discharge) * E_batt_MWh_t[t])
        # Constraining battery energy level to maximum battery level
        mdl.add_constraint(E_SOC_t[t] <=                      E_batt_MWh_t[t])
        # Battery charge/discharge within its power rating
        mdl.add_constraint(P_charge_discharge[t] <= P_batt_MW * charge_efficiency)
        mdl.add_constraint(P_charge_discharge[t] >= -P_batt_MW / charge_efficiency)



# Objective function 
    mdl.minimize(mismatch + penalty_under + penalty_above)


# Solving the problem
    # sol = mdl.solve(
    #     log_output=False)
        #log_output=True) 
    sol = mdl.solve(log_output="cplex.log")
    #aa = mdl.get_solve_details()
    if sol is None:
    # dump the CPLEX log so you can see why it failed
        #print("CPLEX failed:\n", mdl.get_solve_details())
        raise RuntimeError("No feasible solution found")
    # Get the solve statistics
    # details = mdl.get_solve_details()
    # print("Simplex iterations:", details.nb_iterations)


    # Write out a .sol file
    df = sol.as_df(name_key="variable", value_key="value")
    df.to_csv("solution.csv", index=False)
 
 # Extracting values 
    P_HPP_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_HPP_t), orient='index').loc[:,0]

    # P_curtailment_ts_df = pd.DataFrame.from_dict(
    #     sol.get_value_dict(P_curtailment_t), orient='index').loc[:,0]
    

    curt_dict = sol.get_value_dict(P_curtailment_t)
    if curt_dict:
        # normal case: build a 1-col DataFrame, then reindex to full time
        df_curt = pd.DataFrame.from_dict(curt_dict, orient='index')
        P_curtailment_ts = (
            df_curt.iloc[:,0]
            .reindex(time, fill_value=0)
            .values
        )
    else:
        # no curtailment ever occurred → all zeros
        P_curtailment_ts = np.zeros(len(time))

    # P_charge_discharge_ts_df = pd.DataFrame.from_dict(
    #     sol.get_value_dict(P_charge_discharge), orient='index').loc[:,0]
    
    batt_dict = sol.get_value_dict(P_charge_discharge)
    if batt_dict:
        df_batt = pd.DataFrame.from_dict(batt_dict, orient='index')
        P_charge_discharge_ts = (
            df_batt.iloc[:, 0]               # first (and only) column
                .reindex(time, fill_value=0)  # align to full index
                .values
        )
    else:
        # if CPLEX never wrote any battery‐dispatch entries, just zero it out
        P_charge_discharge_ts = np.zeros(len(time))

    E_SOC_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(E_SOC_t), orient='index').loc[:,0]
    
    #make a time series like P_HPP with a constant penalty 
    penalty_2 = sol.get_value(penalty)
    penalty_ts = np.ones(N_t) * (penalty_2/N_t)
    
    mdl.end()

# Handling missing values  
    # Cplex sometimes returns missing values :O
    P_HPP_ts = P_HPP_ts_df.reindex(time,fill_value=0).values
    #P_curtailment_ts = P_curtailment_ts_df.reindex(time,fill_value=0).values
    # P_charge_discharge_ts = P_charge_discharge_ts_df.reindex(time,fill_value=0).values
    E_SOC_ts = E_SOC_ts_df.reindex(SOCtime,fill_value=0).values
    
    if len(P_HPP_ts_df) < len(wind_ts):
        #print('recomputing p_hpp')
        P_HPP_ts = wind_ts.values + solar_ts.values +\
            - P_curtailment_ts + P_charge_discharge_ts

# Return all  the time series 
    return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts


# This is a rule-based EMS function that adjusts battery operation and curtailment to account for degradation in wind, solar, and battery systems.
# It does not enforce a constant output — just tries to track a previously planned EMS (b_t, b_E_SOC_t) as closely as possible under degraded conditions.
def operation_rule_base_no_penalty(
    wind_t_deg,
    solar_t_deg,
    batt_degradation,
    wind_t,
    solar_t,
    hpp_curt_t,
    b_t,
    b_E_SOC_t,
    G_MW,
    b_E,
    battery_depth_of_discharge,
    battery_charge_efficiency,
    load_profile_ts,
    hpp_t,
    b_E_SOC_0 = None,
    load_min_penalty_factor = 1e6,
    change_BES_charging = 'only_for_less_power',
):

    """EMS operation for degraded PV and battery based on an existing EMS.
    """

# Computes maximum battery power 
    B_p = np.max(np.abs(b_t))
    
# Computes total generation before/after degradation 
    wind_solar_t = solar_t + wind_t
    #print(f"wind_solar_t: {wind_solar_t}")
    wind_solar_t_deg = solar_t_deg + wind_t_deg
    
    P_deg_t_sat_loss = (solar_t - solar_t_deg) + (wind_t - wind_t_deg)
    P_loss =  np.maximum( 0 , P_deg_t_sat_loss  - hpp_curt_t)
    
    b_t_less_sol = b_t.copy()
    #print(f"b_t_less_sol: {b_t_less_sol}")
    dt = 1
    
# Adjust battery charging behaviour 
    # Reduction in power to battery due to reduction of solar
    for i in range(len(b_t)):
        if b_t[i] < 0:
            if change_BES_charging == 'proportional':
                if wind_solar_t[i] != 0:
                    # Try to keep the ratio of b_t[i] / wind_solar_t[i]  SoC to the maximum
                    b_t_less_sol[i] = ( b_t[i] / wind_solar_t[i] ) * ( wind_solar_t_deg[i] )

            elif change_BES_charging == 'only_for_less_power':
                if -b_t[i] > P_loss[i]:    
                    b_t_less_sol[i] = b_t_less_sol[i] + P_loss[i]
            
            elif change_BES_charging == 'always':
                # Try to follow SoC to the maximum
                if -b_t[i] > wind_solar_t_deg[i]:    
                    b_t_less_sol[i] = -wind_solar_t_deg[i]

            b_t_less_sol[i] = np.clip(
                b_t_less_sol[i], 
                -B_p,
                B_p)

# SoC initialization  
    # Initialize the SoC
    b_E_SOC_t_sat = b_E_SOC_t.copy()
    if b_E_SOC_0 == None:
        try:
            b_E_SOC_t_sat[0]= b_E_SOC_t[0]
        except:
            raise('len(b_E_SOC_t):', len(b_E_SOC_t))
    else:
        b_E_SOC_t_sat[0]= b_E_SOC_0

# Update battery SoC over time 
    # Update the SoC
    for i in range(len(b_t_less_sol)):
        if b_t_less_sol[i] < 0: # charging
            b_E_SOC_t_sat[i+1] = b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt * battery_charge_efficiency
        if b_t_less_sol[i] >= 0: # discharging
            b_E_SOC_t_sat[i+1] = b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt / battery_charge_efficiency
        
        b_E_SOC_t_sat[i+1] = np.clip(
            b_E_SOC_t_sat[i+1], 
            (1-battery_depth_of_discharge) * b_E * batt_degradation[i], 
            b_E * batt_degradation[i]  )

# Recomputes actual power used       
    # Recompute the battery power
    b_t_sat = b_t.copy()
    for i in range(len(b_t_sat)):
        if b_t[i] < 0:
            b_t_sat[i] = ( ( b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i+1] ) / battery_charge_efficiency ) / dt
        elif b_t[i] >= 0:
            b_t_sat[i] = ( (b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i+1] )  * battery_charge_efficiency ) / dt 
    #print(f"b_t_sat: {b_t_sat}")
    # Hpp_deg = np.minimum( wind_t_deg + solar_t_deg + b_t_sat, G_MW)
    # P_curt_deg = np.maximum( wind_t_deg + solar_t_deg + b_t_sat - G_MW, 0)


    ## -- dispatch strategy -----------------------------------------------

    # NOW: instead of simply doing Hpp_deg = min(...) and P_curt_deg = max(...),
    # we match to the original Hpp_plan using battery and curtailment.
    # First, pull in the Hpp_plan array that you passed as a new argument:

    # assume Hpp_plan was added to the function signature:
    #    ..., load_profile_ts, Hpp_plan, b_E_SOC_0=None, ...


    N = len(wind_t_deg)
    Hpp_deg    = np.zeros(N)
    P_curt_deg = np.zeros(N)

    # b_t_sat is already computed above
    # b_E_SOC_t_sat is already fully up to date

    

    for i in range(N):
        gen_deg  = wind_solar_t_deg [i]
        target = min(hpp_t[i], G_MW)

        #     # only print for the first ten steps
        # if i < 10:
        #     print(f"i={i:2d}  gen_deg={gen_deg:.3f}  target={target:.3f}")
        #     print ("wind_solar_t", wind_solar_t[i], 
        #            "wind_solar_t_deg", wind_solar_t_deg[i], 
        #            "hpp_curt_t", hpp_curt_t[i], 
        #            "b_t_sat", b_t_sat[i], 
        #            "b_E_SOC_t_sat", b_E_SOC_t_sat[i])

        gap = target - gen_deg

        if gap > 0:
            # need to discharge battery
            max_dis = min(
                b_E_SOC_t_sat[i] - (1 - battery_depth_of_discharge) * b_E * batt_degradation[i],
                B_p
            )
            discharge = min(gap, max_dis)
            b_t_use    = discharge
            curtail    = 0.0
        else:
            # surplus: charge battery first
            surplus = -gap
            max_ch = min(
                (b_E * batt_degradation[i] - b_E_SOC_t_sat[i]),
                B_p * battery_charge_efficiency
            )
            charge = min(surplus, max_ch)
            b_t_use = -charge
            curtail = surplus - charge

        # record outputs
        b_t_sat[i]    = b_t_use
        P_curt_deg[i] = curtail
        Hpp_deg[i]    = gen_deg + b_t_use - curtail

    # Finally recompute penalty as before
    def fneg(x): return (x < 0)
    penalty_ts = load_min_penalty_factor * fneg(Hpp_deg - load_profile_ts)

    return Hpp_deg, P_curt_deg, b_t_sat, b_E_SOC_t_sat, penalty_ts




# This builds on top of operation_rule_base_no_penalty, and enforces a constant daily output profile.
def operation_constant_output(
    wind_t_deg,
    solar_t_deg,
    batt_degradation,
    wind_t,
    solar_t,
    hpp_curt_t,
    b_t,
    b_E_SOC_t,
    G_MW,
    b_E,
    battery_depth_of_discharge,
    battery_charge_efficiency,
    load_profile_ts,
    b_E_SOC_0 = None,
    load_min_penalty_factor = 1e6,
    change_BES_charging = 'only_for_less_power'
):

    """EMS operation for degraded PV and battery based on an existing EMS.
    """

# Initial degraded EMS run
# Uses degraded inputs to get initial response
    Hpp_deg, P_curt_deg, b_t_sat, b_E_SOC_t_sat =  operation_rule_base_no_penalty(
        wind_t_deg = wind_t_deg,
        solar_t_deg = solar_t_deg,
        batt_degradation = batt_degradation,
        wind_t = wind_t,
        solar_t = solar_t,
        hpp_curt_t = hpp_curt_t,
        b_t = b_t,
        b_E_SOC_t = b_E_SOC_t,
        G_MW = G_MW,
        b_E = b_E,
        battery_depth_of_discharge = battery_depth_of_discharge,
        battery_charge_efficiency = battery_charge_efficiency,
        b_E_SOC_0 = b_E_SOC_0,
        load_profile_ts = load_profile_ts,
        load_min_penalty_factor = load_min_penalty_factor,
        change_BES_charging = 'proportional',
    )


##----------------------------------------------------------------------------------------------------------------
# Enforce constant power per hour (in a loop)
    iterations = 1
    for ii in range(iterations):
        # fix opreation to constant daily power
        # Creates a DataFrame to group data by day
        df_aux = pd.DataFrame(
            index = range(len(b_t_sat)),
        )
        df_aux['hour'] = df_aux.index.values
        df_aux['Hpp_deg'] = Hpp_deg
        df_aux['P_curt_deg'] = P_curt_deg
        df_aux['b_t_sat'] = b_t_sat

        # Calculate the minimum Hpp_deg per hour (which in this case is just Hpp_deg itself)
        df_aux['Hpp_deg_actual'] = df_aux['Hpp_deg'].round(decimals=3)
        df_aux['P_to_b_removed'] = df_aux['Hpp_deg'] - df_aux['Hpp_deg_actual']

        df_aux['P_to_b_removed_to_charge_battery'] = 0.0
        df_aux.loc[df_aux['b_t_sat'] <= 0, 'P_to_b_removed_to_charge_battery'] = -df_aux.loc[df_aux['b_t_sat'] <= 0, 'P_to_b_removed']

        df_aux['P_to_b_removed_to_curtailment'] = 0.0
        df_aux.loc[df_aux['b_t_sat'] > 0, 'P_to_b_removed_to_curtailment'] = -df_aux.loc[df_aux['b_t_sat'] > 0, 'P_to_b_removed']

        # Update curtailment and battery charge to meet constant output
        P_curt_deg = P_curt_deg + df_aux['P_to_b_removed_to_curtailment'].values
        b_t_sat = b_t_sat + df_aux['P_to_b_removed_to_charge_battery'].values



# Re-run the EMS with new values
        Hpp_deg, P_curt_deg, b_t_sat, b_E_SOC_t_sat =  operation_rule_base_no_penalty(
            #wind_t_deg = wind_t_deg - df_aux['P_to_b_removed_to_curtailment'].values,
            wind_t_deg = wind_t_deg,
            solar_t_deg = solar_t_deg,
            batt_degradation = batt_degradation,
            wind_t = wind_t,
            solar_t = solar_t,
            hpp_curt_t = P_curt_deg,        
            #b_t = b_t_sat +  df_aux['P_to_b_removed_to_charge_battery'].values,
            b_t = b_t_sat, 
            b_E_SOC_t = b_E_SOC_t_sat,
            G_MW = G_MW,
            b_E = b_E,
            battery_depth_of_discharge = battery_depth_of_discharge,
            battery_charge_efficiency = battery_charge_efficiency,
            b_E_SOC_0 = b_E_SOC_0,
            load_profile_ts = load_profile_ts,
            load_min_penalty_factor = load_min_penalty_factor,
            change_BES_charging = 'only_for_less_power',  # keep the same as before
        )
        

# Final enforcement of constant output per hour
    df_aux = pd.DataFrame(index=range(len(b_t_sat)))
    df_aux['hour']        = df_aux.index.values          # each row is its own “hour”
    df_aux['Hpp_deg']     = Hpp_deg
    df_aux['P_curt_deg']  = P_curt_deg
    df_aux['b_t_sat']     = b_t_sat

    # minimum in each 1-h bucket is just that hour’s value → round to desired precision
    df_aux['Hpp_deg_actual'] = df_aux['Hpp_deg'].round(decimals=3)
    df_aux['P_to_b_removed']  = df_aux['Hpp_deg'] - df_aux['Hpp_deg_actual']


    # # split the “removed” energy exactly as BLPP does:
    # df_aux['to_batt']   = 0.0
    # df_aux.loc[df_aux['b_t_sat'] <= 0,    'to_batt'] = -df_aux.loc[df_aux['b_t_sat'] <= 0,    'P_to_b_removed']
    # df_aux['to_curt']   = 0.0
    # df_aux.loc[df_aux['b_t_sat'] >  0,    'to_curt'] = -df_aux.loc[df_aux['b_t_sat'] >  0,    'P_to_b_removed']

    # # update your series
    # P_curt_deg = P_curt_deg + df_aux['to_curt'].values
    # b_t_sat    = b_t_sat    + df_aux['to_batt'].values
    # Hpp_deg    = df_aux['Hpp_deg_actual'].values


    # Update curtailment and battery charge to meet constant output
    Hpp_deg = df_aux['Hpp_deg_actual'].values
    P_curt_deg = P_curt_deg + df_aux['P_to_b_removed'].values    
    
# Compute penalty
    def fneg(x):
       return (x < 0)
    
    penalty_ts = load_min_penalty_factor * fneg(Hpp_deg - load_profile_ts)
       
    return Hpp_deg, P_curt_deg, b_t_sat, b_E_SOC_t_sat, penalty_ts




