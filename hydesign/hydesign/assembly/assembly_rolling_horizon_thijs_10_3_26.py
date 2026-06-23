# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:43:43 2026

@author: thijs
"""

# basic libraries
import numpy as np
import pandas as pd
import openmdao.api as om

from hydesign.weather.weather import ABL
from hydesign.wind.wind import genericWT_surrogate, genericWake_surrogate, wpp, wpp_with_degradation, get_rotor_d
from hydesign.pv.pv import pvp, pvp_with_degradation
# Import the class as 'ems'
from hydesign.ems.rollinghorizontest import ems_constantoutput as ems
from hydesign.ems.ems_rollinghorizon_thijs_9_3_26 import ems_long_term_operation
from hydesign.battery_degradation import battery_degradation, battery_loss_in_capacity_due_to_temp
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
from hydesign.finance.finance_offgrid_thijs_2_2_26 import finance

from hydesign.assembly.hpp_assembly import hpp_base

class hpp_model_constant_output_offgrid(hpp_base):
    """HPP design evaluator"""

    def __init__(
        self,
        tier_a_profile, 
        tier_b_profile,
        tier_b2_profile,  
        sim_pars_fn,
        battery_deg=False,
        load_profile_ts=None,
        load_min_penalty = 100,
        run_mode='rolling',  # <--- NEW: Expose run mode to your top-level scripts
        **kwargs
        ):
        """Initialization of the hybrid power plant evaluator"""
        self.tier_a_profile = tier_a_profile
        self.tier_b_profile = tier_b_profile
        self.tier_b2_profile = tier_b2_profile
        self.load_profile_ts = load_profile_ts
        self.load_min_penalty = load_min_penalty
        self.battery_deg = battery_deg
        self.run_mode = run_mode # <--- NEW
        
        hpp_base.__init__(self,
                          sim_pars_fn=sim_pars_fn,
                          **kwargs
                          )

        N_time = self.N_time
        N_ws = self.N_ws
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
        N_life = sim_pars['N_life']
        self.life_h = N_life * 365 * 24
        life_h = self.life_h
        wind_deg_yr = self.wind_deg_yr
        wind_deg = self.wind_deg
        share_WT_deg_types = self.share_WT_deg_types
        life_y = self.life_y
        price = self.price
        
        input_ts_fn = sim_pars['input_ts_fn']
        genWT_fn = sim_pars['genWT_fn']
        genWake_fn = sim_pars['genWake_fn']
        latitude = sim_pars['latitude']
        longitude = sim_pars['longitude']
        altitude = sim_pars['altitude']
        weeks_per_season_per_year = sim_pars['weeks_per_season_per_year']
        ems_type = sim_pars['ems_type']
        max_num_batteries_allowed = sim_pars['max_num_batteries_allowed']

  
        model = om.Group()
        
        model.add_subsystem(
            'abl', 
            ABL(
                weather_fn=input_ts_fn, 
                N_time=N_time),
            promotes_inputs=['hh']
            )
        model.add_subsystem(
            'genericWT', 
            genericWT_surrogate(
                genWT_fn=genWT_fn,
                N_ws = N_ws),
            promotes_inputs=[
               'hh',
               'd',
               'p_rated',
            ])
        
        model.add_subsystem(
            'genericWake', 
            genericWake_surrogate(
                genWake_fn=genWake_fn,
                N_ws = N_ws),
            promotes_inputs=[
                'Nwt',
                'Awpp',
                'd',
                'p_rated',
                ])
        
        model.add_subsystem(
            'wpp', 
            wpp(
                N_time = N_time,
                N_ws = N_ws,
                wpp_efficiency = wpp_efficiency,)
                )
        
        model.add_subsystem(
            'pvp', 
            pvp(
                weather_fn = input_ts_fn, 
                N_time = N_time,
                latitude = latitude,
                longitude = longitude,
                altitude = altitude,
                tracking = sim_pars['tracking']
               ),
            promotes_inputs=[
                'surface_tilt',
                'surface_azimuth',
                'DC_AC_ratio',
                'solar_MW',
                'land_use_per_solar_MW',
                ])
        
        # --- FIXED BLOCK ---
        model.add_subsystem(
            'ems', 
            ems(  
                N_time = N_time,
                weeks_per_season_per_year = weeks_per_season_per_year,
                life_h = life_h, 
                ems_type = ems_type,
                run_mode = self.run_mode), # <--- NEW: Pass the run mode down
            promotes_inputs=[
                'tier_a_profile',
                'tier_b_profile',
                'tier_b2_profile',
                'price_t', 
                'b_P',
                'b_E',
                'G_MW',
                'battery_depth_of_discharge',
                'battery_charge_efficiency',
                'peak_hr_quantile', 
                'cost_of_battery_P_fluct_in_peak_price_ratio',
                'n_full_power_hours_expected_per_day_at_peak_price',
                'load_profile_ts',
                ],
            # <--- CHANGED: Promoted C2 and Penalty
            promotes_outputs=['Unserved_A', 'Shortfall_B', 'Shortfall_B2', 'Served_C2', 'penalty_t'] 
        ) 
        # -------------------
        
        model.add_subsystem(
            'battery_degradation', 
            battery_degradation(
                weather_fn = input_ts_fn,
                num_batteries = max_num_batteries_allowed,
                life_y = life_y,
                weeks_per_season_per_year = weeks_per_season_per_year,
                battery_deg = battery_deg
            ),
            promotes_inputs=[
                'min_LoH'
                ])

        model.add_subsystem(
            'battery_loss_in_capacity_due_to_temp', 
            battery_loss_in_capacity_due_to_temp(
                weather_fn = input_ts_fn, 
                life_y = life_y,
                weeks_per_season_per_year = weeks_per_season_per_year,
                battery_deg = battery_deg
            ),
            )

        model.add_subsystem(
            'wpp_with_degradation', 
            wpp_with_degradation(
                N_time = N_time,
                N_ws = N_ws,
                wpp_efficiency = wpp_efficiency,
                life_y = life_y,
                wind_deg_yr = wind_deg_yr,
                wind_deg = wind_deg,
                share_WT_deg_types = share_WT_deg_types,
                weeks_per_season_per_year = weeks_per_season_per_year,
                
            )
        )

        model.add_subsystem(
            'pvp_with_degradation', 
            pvp_with_degradation(
                life_y = life_y,
                pv_deg_yr = sim_pars['pv_deg_yr'],
                pv_deg = sim_pars['pv_deg'],
            )
        )
        
        
        model.add_subsystem(
            'ems_long_term_operation', 
            ems_long_term_operation(
                N_time = N_time,
                life_y = life_y,
                ems_type = 'constant_output',
                load_min_penalty_factor = load_min_penalty,
                ),
            promotes_inputs=[
                'b_P',
                'b_E',
                'G_MW',
                'battery_depth_of_discharge',
                'battery_charge_efficiency',
                'load_profile_ts',
                ],
            promotes_outputs=[
                'total_curtailment'
            ])
        
        model.add_subsystem(
            'wpp_cost',
            wpp_cost(
                wind_turbine_cost=sim_pars['wind_turbine_cost'],
                wind_civil_works_cost=sim_pars['wind_civil_works_cost'],
                wind_fixed_onm_cost=sim_pars['wind_fixed_onm_cost'],
                wind_variable_onm_cost=sim_pars['wind_variable_onm_cost'],
                d_ref=sim_pars['d_ref'],
                hh_ref=sim_pars['hh_ref'],
                p_rated_ref=sim_pars['p_rated_ref'],
                N_time = N_time, 
            ),
            promotes_inputs=[
                'Nwt',
                'Awpp',
                'hh',
                'd',
                'p_rated'])
        model.add_subsystem(
            'pvp_cost',
            pvp_cost(
                solar_PV_cost=sim_pars['solar_PV_cost'],
                solar_hardware_installation_cost=sim_pars['solar_hardware_installation_cost'],
                solar_inverter_cost=sim_pars['solar_inverter_cost'],
                solar_fixed_onm_cost=sim_pars['solar_fixed_onm_cost'],
            ),
            promotes_inputs=['solar_MW', 'DC_AC_ratio'])

        model.add_subsystem(
            'battery_cost',
            battery_cost(
                battery_energy_cost=sim_pars['battery_energy_cost'],
                battery_power_cost=sim_pars['battery_power_cost'],
                battery_BOP_installation_commissioning_cost=sim_pars['battery_BOP_installation_commissioning_cost'],
                battery_control_system_cost=sim_pars['battery_control_system_cost'],
                battery_energy_onm_cost=sim_pars['battery_energy_onm_cost'],
                life_y = life_y,
            ),
            promotes_inputs=[
                'b_P',
                'b_E',
                'battery_price_reduction_per_year'])

        model.add_subsystem(
            'shared_cost',
            shared_cost(
                hpp_BOS_soft_cost=sim_pars['hpp_BOS_soft_cost'],
                hpp_grid_connection_cost=sim_pars['hpp_grid_connection_cost'],
                land_cost=sim_pars['land_cost'],
            ),
            promotes_inputs=[
                'G_MW',
                'Awpp',
            ])

        model.add_subsystem(
            'finance', 
            finance(
                N_time = N_time, 
                depreciation_yr = sim_pars['depreciation_yr'],
                depreciation = sim_pars['depreciation'],
                inflation_yr = sim_pars['inflation_yr'],
                inflation = sim_pars['inflation'],
                ref_yr_inflation = sim_pars['ref_yr_inflation'],
                phasing_yr = sim_pars['phasing_yr'],
                phasing_CAPEX = sim_pars['phasing_CAPEX'],
                life_y = life_y,
                ),
            promotes_inputs=['wind_WACC',
                             'solar_WACC', 
                             'battery_WACC',
                             'tax_rate',
                             'load_profile_ts'
                            ],
            promotes_outputs=['NPV',
                              'IRR',
                              'NPV_over_CAPEX',
                              'LCOE',
                              'mean_AEP',
                              'penalty_lifetime',
                              'CAPEX',
                              'OPEX',
                              'break_even_PPA_price',
                              ],
        )
                  
                      
        model.connect('genericWT.ws', 'genericWake.ws')
        model.connect('genericWT.pc', 'genericWake.pc')
        model.connect('genericWT.ct', 'genericWake.ct')
        model.connect('genericWT.ws', 'wpp.ws')
        model.connect('genericWake.pcw', 'wpp.pcw')
        model.connect('abl.wst', 'wpp.wst')
        
        model.connect('wpp.wind_t', 'ems.wind_t')
        model.connect('pvp.solar_t', 'ems.solar_t')
        
        model.connect('ems.b_E_SOC_t', 'battery_degradation.b_E_SOC_t')
        
        model.connect('battery_degradation.SoH', 'battery_loss_in_capacity_due_to_temp.SoH')
        model.connect('battery_loss_in_capacity_due_to_temp.SoH_all', 'ems_long_term_operation.SoH')
        

        model.connect('genericWT.ws', 'wpp_with_degradation.ws')
        model.connect('genericWake.pcw', 'wpp_with_degradation.pcw')
        model.connect('abl.wst', 'wpp_with_degradation.wst')
        model.connect('wpp_with_degradation.wind_t_ext_deg', 'ems_long_term_operation.wind_t_ext_deg')

        model.connect('ems.solar_t_ext','pvp_with_degradation.solar_t_ext')
        model.connect('pvp_with_degradation.solar_t_ext_deg', 'ems_long_term_operation.solar_t_ext_deg')
        
        model.connect('ems.wind_t_ext', 'ems_long_term_operation.wind_t_ext')
        model.connect('ems.solar_t_ext', 'ems_long_term_operation.solar_t_ext')
        model.connect('ems.price_t_ext', 'ems_long_term_operation.price_t_ext')
        model.connect('ems.hpp_curt_t', 'ems_long_term_operation.hpp_curt_t')
        model.connect('ems.b_E_SOC_t', 'ems_long_term_operation.b_E_SOC_t')
        model.connect('ems.b_t', 'ems_long_term_operation.b_t')
        model.connect('ems.hpp_t', 'ems_long_term_operation.hpp_t')

        model.connect('wpp.wind_t', 'wpp_cost.wind_t')
        
        model.connect('battery_degradation.SoH','battery_cost.SoH')
        
        model.connect('pvp.Apvp', 'shared_cost.Apvp')
        
        model.connect('wpp_cost.CAPEX_w', 'finance.CAPEX_w')
        model.connect('wpp_cost.OPEX_w', 'finance.OPEX_w')

        model.connect('pvp_cost.CAPEX_s', 'finance.CAPEX_s')
        model.connect('pvp_cost.OPEX_s', 'finance.OPEX_s')

        model.connect('battery_cost.CAPEX_b', 'finance.CAPEX_b')
        model.connect('battery_cost.OPEX_b', 'finance.OPEX_b')

        model.connect('shared_cost.CAPEX_sh', 'finance.CAPEX_el')
        model.connect('shared_cost.OPEX_sh', 'finance.OPEX_el')

        model.connect('ems.price_t_ext', 'finance.price_t_ext')
        model.connect('ems_long_term_operation.hpp_t_with_deg', 'finance.hpp_t_with_deg')
        
        # <--- CHANGED: Bypassed the zero-array in long_term_operation. Connected directly from EMS.
        model.connect('penalty_t', 'finance.penalty_t')    
        
        prob = om.Problem(
            model,
            reports=None
        )

        prob.setup()        
        
        # Additional parameters
        prob.set_val('price_t', price)
        prob.set_val('G_MW', sim_pars['G_MW'])
        prob.set_val('battery_depth_of_discharge', sim_pars['battery_depth_of_discharge'])
        prob.set_val('battery_charge_efficiency', sim_pars['battery_charge_efficiency'])      
        prob.set_val('peak_hr_quantile',sim_pars['peak_hr_quantile'] )
        prob.set_val('n_full_power_hours_expected_per_day_at_peak_price',
                     sim_pars['n_full_power_hours_expected_per_day_at_peak_price'])        
        prob.set_val('min_LoH', sim_pars['min_LoH'])
        prob.set_val('wind_WACC', sim_pars['wind_WACC'])
        prob.set_val('solar_WACC', sim_pars['solar_WACC'])
        prob.set_val('battery_WACC', sim_pars['battery_WACC'])
        prob.set_val('tax_rate', sim_pars['tax_rate'])
        prob.set_val('land_use_per_solar_MW', sim_pars['land_use_per_solar_MW'])
        prob.set_val('load_profile_ts', self.load_profile_ts)

        # Set the new profiles!
        prob.set_val('tier_a_profile', self.tier_a_profile)
        prob.set_val('tier_b_profile', self.tier_b_profile)
        prob.set_val('tier_b2_profile', self.tier_b2_profile)
        

        self.prob = prob
    
        self.list_out_vars = [
            'NPV_over_CAPEX',
            'NPV [MEuro]',
            'IRR',
            'LCOE [Euro/MWh]',
            'CAPEX [MEuro]',
            'OPEX [MEuro]',
            'Wind CAPEX [MEuro]',
            'Wind OPEX [MEuro]',
            'PV CAPEX [MEuro]',
            'PV OPEX [MEuro]',
            'Batt CAPEX [MEuro]',
            'Batt OPEX [MEuro]',
            'Shared CAPEX [MEuro]',
            'Shared Opex [MEuro]',
            'penalty lifetime [MEuro]',
            'AEP [GWh]',
            'GUF',
            'grid [MW]',
            'wind [MW]',
            'solar [MW]',
            'Battery Energy [MWh]',
            'Battery Power [MW]',
            'Total curtailment [GWh]',
            'Awpp [km2]',
            'Apvp [km2]',
            'Plant area [km2]',
            'Rotor diam [m]',
            'Hub height [m]',
            'Number of batteries used in lifetime',
            'Break-even PPA price [Euro/MWh]',
            'Capacity factor wind [-]'
            ]

        self.list_vars = [
            'clearance [m]', 
            'sp [W/m2]', 
            'p_rated [MW]', 
            'Nwt', 
            'wind_MW_per_km2 [MW/km2]', 
            'solar_MW [MW]', 
            'surface_tilt [deg]', 
            'surface_azimuth [deg]', 
            'DC_AC_ratio', 
            'b_P [MW]', 
            'b_E_h [h]',
            'cost_of_battery_P_fluct_in_peak_price_ratio'
            ]   
    
    
    def evaluate(
        self,
        clearance, sp, p_rated, Nwt, wind_MW_per_km2,
        solar_MW,  surface_tilt, surface_azimuth, DC_AC_ratio,
        b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio
        ):

        prob = self.prob

        d = get_rotor_d(p_rated*1e6/sp)
        hh = (d/2)+clearance
        wind_MW = Nwt * p_rated
        Awpp = wind_MW / wind_MW_per_km2 
        b_E = b_E_h * b_P
        
        # pass design variables        
        prob.set_val('hh', hh)
        prob.set_val('d', d)
        prob.set_val('p_rated', p_rated)
        prob.set_val('Nwt', Nwt)
        prob.set_val('Awpp', Awpp)

        prob.set_val('surface_tilt', surface_tilt)
        prob.set_val('surface_azimuth', surface_azimuth)
        prob.set_val('DC_AC_ratio', DC_AC_ratio)
        prob.set_val('solar_MW', solar_MW)
        
        prob.set_val('b_P', b_P)
        prob.set_val('b_E', b_E)
        prob.set_val('cost_of_battery_P_fluct_in_peak_price_ratio',cost_of_battery_P_fluct_in_peak_price_ratio)        
        
        prob.run_model()
        
        self.prob = prob
        
        if Nwt == 0:
            cf_wind = np.nan
        else:
            cf_wind = prob.get_val('wpp_with_degradation.wind_t_ext_deg').mean() / p_rated / Nwt 

        return np.hstack([
            prob['NPV_over_CAPEX'], 
            prob['NPV']/1e6,
            prob['IRR'],
            prob['LCOE'],
            prob['CAPEX']/1e6,
            prob['OPEX']/1e6,
            prob.get_val('finance.CAPEX_w')/1e6,
            prob.get_val('finance.OPEX_w')/1e6,
            prob.get_val('finance.CAPEX_s')/1e6,
            prob.get_val('finance.OPEX_s')/1e6,
            prob.get_val('finance.CAPEX_b')/1e6,
            prob.get_val('finance.OPEX_b')/1e6,
            prob.get_val('finance.CAPEX_el')/1e6,
            prob.get_val('finance.OPEX_el')/1e6,
            prob['penalty_lifetime']/1e6,
            prob['mean_AEP']/1e3,
            prob['mean_AEP']/(self.sim_pars['G_MW']*365*24),
            self.sim_pars['G_MW'],
            wind_MW,
            solar_MW,
            b_E,
            b_P,
            prob['total_curtailment']/1e3,
            Awpp,
            prob.get_val('shared_cost.Apvp'),
            max( Awpp , prob.get_val('shared_cost.Apvp') ),
            d,
            hh,
            prob.get_val('battery_degradation.n_batteries') * (b_P>0),
            prob['break_even_PPA_price'],
            cf_wind,
            ])