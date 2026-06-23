import numpy as np
import pandas as pd
import openmdao.api as om

from docplex.mp.model import Model
from hydesign.ems.ems import expand_to_lifetime, split_in_batch
# work from ems consntant output, change min load with min demand 
# implement load profile 
# load_profile = pd.read_csv(
#     "/Users/lisavanweegen/hydesign/hydesign/examples/mendelev_pharma.csv",
#     index_col=0,
#     parse_dates=True
# )

# load_profile = pd.read_csv(
#     "/Users/lisavanweegen/hydesign/hydesign/examples/step_profile_pattern_comma.csv",
#     index_col=0,
#     parse_dates=True
# )

# load_profile = pd.read_csv(
#     "/Users/lisavanweegen/hydesign/hydesign/examples/baseload_profile_8MW.csv",
#     index_col=0,
#     parse_dates=True
# )

# load_profile = pd.read_csv(
#     "/Users/lisavanweegen/hydesign/hydesign/examples/baseload_profile_16MW.csv",
#     index_col=0,
#     parse_dates=True
# )

# load_profile = pd.read_csv(
#     "/Users/lisavanweegen/hydesign/hydesign/examples/cos_profile_4_10MW_updated.csv",
#     index_col=0,
#     parse_dates=True
# )

# load_profile = pd.read_csv(
#     "/Users/lisavanweegen/hydesign/hydesign/examples/step_profile_6_12MW_corrected.csv",
#     index_col=0,
#     parse_dates=True
# )

# load_profile = pd.read_csv(
#     "/Users/lisavanweegen/hydesign/hydesign/examples/mendelev_pharma_higher8.csv",
#     index_col=0,
#     parse_dates=True
# )

# load_profile = pd.read_csv(
#     "/Users/lisavanweegen/hydesign/hydesign/examples/Cleaned_College_Data.csv",
#     index_col=0,
#     parse_dates=True
# )

# load_profile = pd.read_csv(
#     "/Users/lisavanweegen/hydesign/hydesign/examples/general_manufacturer.csv",delimiter=';',
#     index_col=0,
#     parse_dates=True
# )

# load_profile_ts = load_profile['Power [MW]']
# p=1

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
# for your demad profile 
        self.add_input(
            'load_profile_ts',
            desc="Hourly load profile to be satisfied [MW]",
            shape=[self.life_h],
            units='MW')

        # self.add_input(
        #     'load_min',
        #     desc="Minimum electrical load to meet")
        
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
            shape=[self.life_h])  # Add wind_t_deg to outputs
        self.add_output(
            'solar_t_deg',
            desc="PV time series including degradation",
            units='MW',
            shape=[self.life_h])  # Add solar_t_deg to outputs
        
    def compute(self, inputs, outputs):
        
        SoH = inputs['SoH']
        wind_t_ext_deg = inputs['wind_t_ext_deg']
        solar_t_ext_deg = inputs['solar_t_ext_deg']
        
        wind_t_ext = inputs['wind_t_ext']
        solar_t_ext = inputs['solar_t_ext']
        price_t_ext = inputs['price_t_ext']
        # b_P = inputs['b_P']
        b_E = inputs['b_E']
        G_MW = inputs['G_MW']
        battery_depth_of_discharge = inputs['battery_depth_of_discharge']
        battery_charge_efficiency = inputs['battery_charge_efficiency']
        hpp_curt_t = inputs['hpp_curt_t']
        b_t = inputs['b_t']
        b_E_SOC_t = inputs['b_E_SOC_t']
        load_profile_ts = inputs['load_profile_ts'] 
        peak_hr_quantile = inputs['peak_hr_quantile'][0]
        n_full_power_hours_expected_per_day_at_peak_price = inputs['n_full_power_hours_expected_per_day_at_peak_price'][0]

        life_h = self.life_h
        
        if self.ems_type == 'constant_output':
            ems_longterm = operation_constant_output
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
            G_MW = G_MW[0],
            b_E = b_E[0],
            battery_depth_of_discharge = battery_depth_of_discharge[0],
            battery_charge_efficiency = battery_charge_efficiency[0],
            b_E_SOC_0 = None,
            #load_min = inputs['load_min'],
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

        # self.add_input(
        #     'load_min',
        #     desc="Minimum electrical load to meet")
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
        #load_min = inputs['load_min']
        load_profile_ts = inputs['load_profile_ts'] 
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
            #load_min=load_min,
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
            #load_min=load_min,
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
    #load_min = 2, # MW
    load_min_penalty_factor = 1e6,
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

# Define time index for SOC 
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))

# Define decision variables 
    P_HPP_t = mdl.continuous_var_dict(
        time, lb=0, ub=hpp_grid_connection, 
        name='HPP power output')
    P_curtailment_t = mdl.continuous_var_dict(
        time, lb=0, 
        name='Curtailment')
    P_charge_discharge = mdl.continuous_var_dict(
        time, lb=-P_batt_MW/charge_efficiency, ub=P_batt_MW*charge_efficiency, 
        name='Battery power')
    E_SOC_t = mdl.continuous_var_dict(
        SOCtime, lb=0, #ub=E_batt_MWh_t.max(), 
        name='Energy level')
    #constant output per day
    #P_constant = mdl.continuous_var_dict(range(int(N_days)), lb=0, name='constant output')
    # instead of range(int(N_days)) do the following for constant output per hour 
    P_constant = mdl.continuous_var_dict(range(N_t), lb=0, name='constant_per_timestep')


    penalty = mdl.continuous_var(name='penalty', lb=-1e12)
    x = mdl.continuous_var_dict(time, lb=0, name='dummy for batt disch')
# Define objective function - inlcudes revenue as well as the penalty  
    # fneg(x) returns only negative values (used to compute penalties)
    fneg = mdl.piecewise(-1, [(0,0)], 0)
    mdl.maximize(
        # maximize revenue: price * output
        mdl.sum(
            price_ts[t] * P_HPP_t[t]
            for t in time)
#    ) 
            - penalty     
        # Penalty: Subtracted for not reaching minimum load (load_min) 
    ) 

# # Define objective function - includes only revenue 
#     mdl.maximize(
#         # maximize revenue: price * output
#         mdl.sum(
#             price_ts[t] * P_HPP_t[t]
#             for t in time)) 
#Constraints
    # This constraint links the penalty to how often power output is below the required load.
    mdl.add_constraint(penalty == mdl.sum(load_min_penalty_factor * fneg(P_HPP_t[t] - load_profile_ts[time.get_loc(t)])for t in time))
        
    # Intitial and end SOC
    mdl.add_constraint( E_SOC_t[SOCtime[0]] == 0.5 * E_batt_MWh_t[time[0]] )
    
    # SOC at the end of the year has to be equal to SOC at the beginning of the year
    mdl.add_constraint( E_SOC_t[SOCtime[-1]] == 0.5 * E_batt_MWh_t[time[0]] )

    # pircewise linear representation of charge vs dischrage effciency 
    f2 = mdl.piecewise(charge_efficiency,[(0,0)],1/charge_efficiency)
    # PWL function to model a unit step function
    f3 = mdl.piecewise(0, [(0, 0), (0, 1)], 0)

    # Time based constraints - all constraints applied per hour
    for t in time:
        # Time index for successive time step
        tt = t + pd.Timedelta('1hour')
        # Delta_t of 1 hour
        dt = 1
        # Only one variable for battery
        # Power balance 
        mdl.add_constraint(
            P_HPP_t[t] == wind_ts[t] +
            solar_ts[t] +
            - P_curtailment_t[t] +
            P_charge_discharge[t])
        
        # Charge/dischrage equation
        # Battery SoC update 
        mdl.add_constraint(
            E_SOC_t[tt] == E_SOC_t[t] - 
            f2(P_charge_discharge[t]) * dt)
        
        # Battery energy bounnds  
        # Constraining battery energy level to minimum battery level
        mdl.add_constraint(E_SOC_t[t] >= (1 - battery_depth_of_discharge) * E_batt_MWh_t[t])
        # Constraining battery energy level to maximum battery level
        mdl.add_constraint(E_SOC_t[t] <= E_batt_MWh_t[t])

        # Battery power bounds 
        # Battery charge/discharge within its power rating
        mdl.add_constraint(P_charge_discharge[t] <= P_batt_MW*charge_efficiency)
        mdl.add_constraint(P_charge_discharge[t] >= -P_batt_MW/charge_efficiency)
        
        # # Enforce constant output per day -  so that your plant’s output at each hour is tied to its own “flat” variable for that hour
        # idx = time.get_loc(t)    
        # #mdl.add_constraint(P_HPP_t[t] == P_constant[idx])

        # # Matching the load profile exactly each hour
        # mdl.add_constraint(P_HPP_t[t] == load_profile_ts[idx])

        # Enforce connstant power output per day
        #mdl.add_constraint(P_HPP_t[t] == P_constant[int(time.get_loc(t)) // 24])
    # If you p_hpp is exactly equal to your load profile 
        #P_HPP_t[t] == load_profile_ts[time.get_loc(t)]
        # Enforce connstant power output per hour
        #idx = time.get_loc(t)    # this is already an integer 0,1,2… len(time)-1
        #mdl.add_constraint(P_HPP_t[t] == P_constant[idx])
        #mdl.add_constraint(P_HPP_t[t] == P_constant[int(time.get_loc(t)) / 1])
        #mdl.add_constraint(P_HPP_t[t] == load_profile_ts[idx])



# # Added constraints from Baseload EMS 
# # PWL function will set P_discharge to 0, if the battery SOC is not sufficient to meet the base load 
# # else it’s value will be 1.  
#         if (hpp_grid_connection - wind_ts[t] - solar_ts[t]) > 0:
#             mdl.add_constraint (x[t] == f3(E_SOC_t[t]- hpp_grid_connection + wind_ts[t] + solar_ts[t])) 
#         else:
#             mdl.add_constraint(x[t] == 0)
#         mdl.add_constraint(P_charge_discharge[t] <= P_batt_MW * charge_efficiency * x[t])

#         # Ensures battery output is bounded by grid connection capacity.
#         mdl.add_constraint(P_charge_discharge[t] <= hpp_grid_connection * 1.0)

# Solving the problem
    sol = mdl.solve(
        log_output=False)
        #log_output=True)
    aa = mdl.get_solve_details()
 
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
    b_E_SOC_0 = None,
    #load_min = 3,
    load_min_penalty_factor = 1e6,
    change_BES_charging = 'only_for_less_power',
):

    """EMS operation for degraded PV and battery based on an existing EMS.
    """

# Computes maximum battery power 
    B_p = np.max(np.abs(b_t))
    
# Computes total generation before/after degradation 
    wind_solar_t = solar_t + wind_t
    wind_solar_t_deg = solar_t_deg + wind_t_deg
    
    P_deg_t_sat_loss = (solar_t - solar_t_deg) + (wind_t - wind_t_deg)
    P_loss =  np.maximum( 0 , P_deg_t_sat_loss  - hpp_curt_t)
    
    b_t_less_sol = b_t.copy()
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
    
    Hpp_deg = np.minimum( wind_t_deg + solar_t_deg + b_t_sat, G_MW)
    P_curt_deg = np.maximum( wind_t_deg + solar_t_deg + b_t_sat - G_MW, 0)


    return Hpp_deg, P_curt_deg, b_t_sat, b_E_SOC_t_sat
    
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
    #load_min = 3,
    load_min_penalty_factor = 1e6,
    change_BES_charging = 'only_for_less_power',
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
            change_BES_charging = 'proportional',
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




