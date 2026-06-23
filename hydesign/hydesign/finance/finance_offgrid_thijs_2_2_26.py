# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:25:57 2026

@author: thijs
"""

import glob
import os
import time

# basic libraries
import numpy as np
from numpy import newaxis as na
import numpy_financial as npf
import pandas as pd
import openmdao.api as om
import yaml
import scipy as sp

class finance(om.ExplicitComponent):
    """Hybrid power plant financial model to estimate the overall profitability of the hybrid power plant.
    Adjusted for Data Center / Off-grid applications where LCOE depends on Energy Served vs Demand.
    """

    def __init__(
        self, 
        N_time, 
        # Depreciation curve
        depreciation_yr,
        depreciation,
        # Inflation curve
        inflation_yr,
        inflation,
        ref_yr_inflation,
        # Early paying or CAPEX Phasing
        phasing_yr,
        phasing_CAPEX,
        life_y = 25,
        ):
        """Initialization of the HPP finance model""" 
        super().__init__()
        self.N_time = int(N_time)
        self.life_y = life_y
        self.life_h = int(life_y*365*24)

        # Depreciation curve
        self.depreciation_yr = depreciation_yr
        self.depreciation = depreciation

        # Inflation curve
        self.inflation_yr = inflation_yr
        self.inflation = inflation
        self.ref_yr_inflation = ref_yr_inflation

        # Early paying or CAPEX Phasing
        self.phasing_yr = phasing_yr
        self.phasing_CAPEX = phasing_CAPEX

    def setup(self):
        self.add_input('price_t_ext',
                       desc="Electricity price time series",
                       shape=[self.life_h])
        
        self.add_input('hpp_t_with_deg',
                       desc="HPP power time series",
                       units='MW',
                       shape=[self.life_h])
        
        # Note: We treat hpp_t and hpp_t_with_deg the same for this logic
        # usually hpp_t comes from EMS ideal, with_deg includes degradation
        self.add_input('hpp_t',
                       desc="HPP power time series without degradation",
                       units='MW',
                       shape=[self.life_h])
        
        self.add_input('penalty_t',
                        desc="penalty for not reaching expected energy production",
                        shape=[self.life_h])

        self.add_input('CAPEX_w', desc="CAPEX wpp")
        self.add_input('OPEX_w', desc="OPEX wpp")

        self.add_input('CAPEX_s', desc="CAPEX solar pvp")
        self.add_input('OPEX_s', desc="OPEX solar pvp")

        self.add_input('CAPEX_b', desc="CAPEX battery")
        self.add_input('OPEX_b', desc="OPEX battery")

        self.add_input('CAPEX_el', desc="CAPEX electrical infrastructure")
        self.add_input('OPEX_el', desc="OPEX electrical infrastructure")

        self.add_input('wind_WACC', desc="After tax WACC for onshore WT")
        self.add_input('solar_WACC', desc="After tax WACC for solar PV")
        self.add_input('battery_WACC', desc="After tax WACC for stationary storge li-ion batteries")
        self.add_input('tax_rate', desc="Corporate tax rate")
        
        self.add_output('CAPEX', desc="CAPEX")
        self.add_output('OPEX', desc="OPEX")
        self.add_output('NPV', desc="NPV")
        self.add_output('IRR', desc="IRR")
        self.add_output('NPV_over_CAPEX', desc="NPV/CAPEX")
        self.add_output('level_costs', desc="level costs")
        self.add_output('mean_AEP', desc="mean AEP")
        self.add_output('LCOE', desc="LCOE")
        self.add_output('revenues', desc="Revenues")
        self.add_output('penalty_lifetime', desc="penalty_lifetime")

        self.add_output('break_even_PPA_price',
                        desc='PPA price of electricity that results in NPV=0',
                        val=0)
        
        self.add_input('load_profile_ts',
                    desc="Industry demand time series [MW]",
                    shape=(self.life_h,),
                    units='MW')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        
        N_time = self.N_time
        life_h = self.life_h
        life_yr = int(np.ceil(life_h/N_time))

        depreciation_yr = self.depreciation_yr
        depreciation = self.depreciation

        inflation_yr = self.inflation_yr
        inflation = self.inflation
        ref_yr_inflation = self.ref_yr_inflation

        phasing_yr = self.phasing_yr
        phasing_CAPEX = self.phasing_CAPEX

        # Prepare DataFrame
        df = pd.DataFrame()
        # Use degradation signal for financial reality
        df['hpp_t'] = inputs['hpp_t_with_deg'] 
        df['penalty_t'] = inputs['penalty_t']
        df['load_profile_ts'] = inputs['load_profile_ts']
        df['i_year'] = np.hstack([np.array([ii]*N_time) for ii in range(life_yr)])[:life_h]

        # --- 1. REVENUE CALCULATION ---
        # Note: Penalties are ignored for LCOE/Revenue calculation as requested
        revenues = calculate_revenues(inputs['price_t_ext'], df)
        
        # --- 2. COST AGGREGATION ---
        CAPEX = inputs['CAPEX_w'] + inputs['CAPEX_s'] + \
            inputs['CAPEX_b'] + inputs['CAPEX_el']
        OPEX = inputs['OPEX_w'] + inputs['OPEX_s'] + \
            inputs['OPEX_b'] + inputs['OPEX_el']

        # Discount rate
        hpp_discount_factor = calculate_WACC(
            inputs['CAPEX_w'], inputs['CAPEX_s'], inputs['CAPEX_b'], inputs['CAPEX_el'],
            inputs['wind_WACC'], inputs['solar_WACC'], inputs['battery_WACC'],
            )
        
        # CAPEX Phasing
        inflation_index_phasing = get_inflation_index(
            yr = phasing_yr,
            inflation_yr = inflation_yr, 
            inflation = inflation,
            ref_yr_inflation = ref_yr_inflation,
        )
        CAPEX_eq = calculate_CAPEX_phasing(
            CAPEX = CAPEX,
            phasing_yr = phasing_yr,
            phasing_CAPEX = phasing_CAPEX,
            discount_rate = hpp_discount_factor,
            inflation_index = inflation_index_phasing,
            )

        # Inflation Index for Lifetime
        iy = np.arange(len(revenues)) + 1 
        inflation_index = get_inflation_index(
                yr = np.arange(len(revenues)+1), 
                inflation_yr = inflation_yr, 
                inflation = inflation,
                ref_yr_inflation = ref_yr_inflation,
        )
        
        revenues = revenues.values.flatten()
        outputs['CAPEX'] = CAPEX
        outputs['OPEX'] = OPEX
        outputs['revenues'] = revenues.mean()

        DEVEX = 0

        # NPV & IRR
        NPV, IRR = calculate_NPV_IRR(
            Net_revenue_t = revenues,
            investment_cost = CAPEX_eq, 
            maintenance_cost_per_year = OPEX,
            tax_rate = inputs['tax_rate'],
            discount_rate = hpp_discount_factor,
            depreciation_yr = depreciation_yr,
            depreciation = depreciation,
            development_cost = DEVEX, 
            inflation_index = inflation_index,
        )
        
        break_even_PPA_price = np.maximum(
            0, 
            calculate_break_even_PPA_price(
                df = df, 
                CAPEX = CAPEX_eq, 
                OPEX = OPEX, 
                tax_rate = inputs['tax_rate'],
                discount_rate = hpp_discount_factor,
                depreciation_yr = depreciation_yr, 
                depreciation = depreciation, 
                DEVEX = DEVEX,
                inflation_index = inflation_index,
                ))

        outputs['NPV'] = NPV
        outputs['IRR'] = IRR
        outputs['NPV_over_CAPEX'] = NPV / CAPEX

        # --- 3. LCOE CALCULATION (CRITICAL FIX) ---
        level_costs = np.sum(OPEX / (1 + hpp_discount_factor)**iy) + CAPEX
        
        # [FIX] Trust the EMS output. 
        # hpp_t is strictly defined as "Served Load" in the EMS.
        # It excludes curtailment and battery losses.
        df['energy_served'] = df['hpp_t'] 
        
        # Sum annual energy served
        AEP_per_year = df.groupby('i_year')['energy_served'].sum()
        
        # For debug
        # print("AEP per year (GWh):", AEP_per_year.values / 1e3)

        level_AEP = np.sum(AEP_per_year / (1 + hpp_discount_factor)**iy)
        
        outputs['level_costs'] = level_costs
        
        mean_AEP_per_year = np.mean(AEP_per_year)
        
        if level_AEP > 0:
            outputs['LCOE'] = level_costs / level_AEP # Euro/MWh
        else:
            outputs['LCOE'] = 1e6 # Penalty if no energy served

        outputs['mean_AEP'] = mean_AEP_per_year
        outputs['penalty_lifetime'] = df['penalty_t'].sum()
        outputs['break_even_PPA_price'] = break_even_PPA_price


# -----------------------------------------------------------------------
# Auxiliar functions for financial modelling
# -----------------------------------------------------------------------

def calculate_NPV_IRR(
    Net_revenue_t,
    investment_cost,
    maintenance_cost_per_year,
    tax_rate,
    discount_rate,
    depreciation_yr,
    depreciation,
    development_cost,
    inflation_index
):
    yr = np.arange(len(Net_revenue_t)+1) 
    depre = np.interp(yr, depreciation_yr, depreciation)

    # EBITDA: earnings before interest and taxes in nominal prices
    EBITDA = (Net_revenue_t - maintenance_cost_per_year) * inflation_index[1:]

    # EBIT taxable income
    depreciation_on_each_year = np.diff(investment_cost*depre)
    EBIT = EBITDA - depreciation_on_each_year
    
    # Taxes
    Taxes = EBIT*tax_rate
    
    Net_income = EBITDA - Taxes
    Cashflow = np.insert(Net_income, 0, -investment_cost-development_cost)
    NPV = npf.npv(discount_rate, Cashflow)
    if NPV > 0:
        IRR = npf.irr(Cashflow)     
    else:
        IRR = 0
    return NPV, IRR

def calculate_WACC(
    CAPEX_w, CAPEX_s, CAPEX_b, CAPEX_el,
    wind_WACC, solar_WACC, battery_WACC,
    ):
    # Weighted average cost of capital 
    # Handle zero division if CAPEX is 0
    total_capex = CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_el
    if total_capex == 0:
        return 0.05 # Default return

    WACC_after_tax = \
        ( CAPEX_w * wind_WACC + \
          CAPEX_s * solar_WACC + \
          CAPEX_b * battery_WACC + \
          CAPEX_el * (wind_WACC + solar_WACC + battery_WACC)/3 ) / total_capex
    return WACC_after_tax


def calculate_revenues(price_el, df):
    # CHANGED: Penalties are NOT subtracted from revenue for LCOE calculation
    # We calculate revenue based on Energy Delivered * Price
    # Ensure price_el broadcasts correctly if it's a scalar or time series
    price_ts = np.broadcast_to(price_el, df['hpp_t'].shape)
    
    # Revenue is based on energy actually delivered (hpp_t)
    df['revenue'] = df['hpp_t'] * price_ts
    
    # Return annual revenue sum
    return df.groupby('i_year').revenue.sum()

    

def calculate_break_even_PPA_price(df, CAPEX, OPEX, tax_rate, discount_rate,
                                   depreciation_yr, depreciation, DEVEX, inflation_index):
    def fun(price_el):
        revenues = calculate_revenues(price_el, df)
        NPV, _ = calculate_NPV_IRR(
            Net_revenue_t = revenues.values.flatten(),
            investment_cost = CAPEX,
            maintenance_cost_per_year = OPEX,
            tax_rate = tax_rate,
            discount_rate = discount_rate,
            depreciation_yr = depreciation_yr,
            depreciation = depreciation,
            development_cost = DEVEX,
            inflation_index = inflation_index,
        )
        return NPV ** 2
    out = sp.optimize.minimize(
        fun=fun, 
        x0=50, 
        method='SLSQP',
        tol=1e-10)
    return out.x


def calculate_CAPEX_phasing(
    CAPEX, phasing_yr, phasing_CAPEX, discount_rate, inflation_index):
    
    if np.sum(phasing_CAPEX) == 0:
        return CAPEX

    phasing_CAPEX = inflation_index*CAPEX*phasing_CAPEX/np.sum(phasing_CAPEX)
    CAPEX_eq = np.sum([phasing_CAPEX[ii]/( 1 + discount_rate)**yr for ii,yr in enumerate(phasing_yr)])
    
    return CAPEX_eq

def get_inflation_index(
    yr, inflation_yr, inflation, ref_yr_inflation = 0):
    
    infl = np.interp(yr, inflation_yr, inflation)
    
    ind_ref = np.where(np.array(yr)==ref_yr_inflation)[0]
    # Handle case where ref year is not found
    if len(ind_ref) == 0: ind_ref = [0]
        
    inflation_index = np.cumprod(1+np.array(infl))
    inflation_index = inflation_index/inflation_index[ind_ref]
    
    return inflation_index