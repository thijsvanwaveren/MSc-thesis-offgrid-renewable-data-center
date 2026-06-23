# %%

import pandas as pd
import numpy as np
import os
import yaml
import math 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import cm
from hydesign.assembly.hpp_assembly_costantoutput_matching_demand_ideal_output import hpp_model_constant_output as hpp_model
from hydesign.examples import examples_filepath
from hydesign.ems.ems import expand_to_lifetime
from itertools import product


## --- IMPORTS -----------------------------------------------------------------------
examples_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "\\"
examples_sites = pd.read_csv(f"{examples_filepath}examples_sites.csv", sep=';')


name = 'France_good_wind'
ex_site = examples_sites.loc[examples_sites.name == name]

longitude = ex_site['longitude'].values[0]
latitude = ex_site['latitude'].values[0]
altitude = ex_site['altitude'].values[0]

input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]
sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]


with open(sim_pars_fn, 'r') as file:
    sim_pars = yaml.safe_load(file)

weeks_per_season_per_year = sim_pars.get('weeks_per_season_per_year', 1)

## --- DEMAND PROFILES ---------------------------------------------------


# Set base path
base_path = examples_filepath

# Define profiles and categories
profiles_info = {
    "DataCenter_Constant_Load.csv": "datacenter"
}

# Store data
load_profiles_MW = {}          # original in MW
load_profiles_scaled_MW = {}   # scaled in MW
profile_categories = {}        # profile → category
original_annual_energy_MWh = {}# Store original annual energy in MWh



for filename, category in profiles_info.items():
    full_path = os.path.join(base_path, filename)
    
    # Load and clean
    df = pd.read_csv(full_path, index_col=0, delimiter=';', parse_dates=True)
    df.columns = df.columns.str.strip()
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # Ensure 'Power [kW]' is numeric
    df['Power [kW]'] = pd.to_numeric(df['Power [kW]'], errors='coerce')

    # Drop NaNs and convert to MW
    load_ts_MW = df['Power [kW]'].dropna() * 0.001
    
    # Compute current annual energy (assumes hourly resolution)
    annual_energy_MWh = load_ts_MW.sum()
    
    # Save results
    profile_name = filename.replace('.csv', '')
    load_profiles_MW[profile_name] = load_ts_MW
    profile_categories[profile_name] = category



#Define category groups and corresponding profile names
category_profiles = {
    "Baseload": ["Commercial-Supermarket", "Commercial-Wastewater", "Baseload"],
    "Daily": ["Industrial-Manufacturer", "Industrial-1Metals", "Daily"],
    "Peaky": ["Commercial-450Hospital", "Commercial-College", "Peaky"],
    "Seasonal": ["Commercial-Office", "Industrial-Services", "seasonal"]
}

# After defining category_profiles:
profile_categories = {}
for category, profile_list in category_profiles.items():
    for profile in profile_list:
        profile_categories[profile] = category

target_annual_energy_MWh = 140000

for filename, category in profiles_info.items():
    full_path = os.path.join(base_path, filename)
    
    # Load and clean
    df = pd.read_csv(full_path, index_col=0, delimiter=';', parse_dates=True)
    df.columns = df.columns.str.strip()
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # Ensure 'Power [kW]' is numeric
    df['Power [kW]'] = pd.to_numeric(df['Power [kW]'], errors='coerce')

    # Drop NaNs and convert to MW
    load_ts_MW = df['Power [kW]'].dropna() * 0.001
    
    # Compute current annual energy (assumes hourly resolution)
    annual_energy_MWh = load_ts_MW.sum()
    
    # Scale to target
    scaling_factor = target_annual_energy_MWh / annual_energy_MWh
    load_ts_scaled = load_ts_MW * scaling_factor
    
    # Save results
    profile_name = filename.replace('.csv', '')
    load_profiles_MW[profile_name] = load_ts_MW
    load_profiles_scaled_MW[profile_name] = load_ts_scaled
    #profile_categories[profile_name] = category



# # --- FIXED VARIABLES ---------------------------------------------------
clearance = 35
sp = 300
p_rated = 5
#Nwt = 8
wind_MW_per_km2 = 7
#solar_MW = 10
surface_tilt = 39
surface_azimuth = 180
solar_DCAC = 1.25
#b_P = 10
b_E_h  = 4
cost_of_batt_degr = 10
#PPA = 40 # Euro/MWh -->>>> CHECK!!!!!


## --- DESIGN PARAMETERS -----------------------------------------------
Nwt_list = [6, 9, 12] # , 12, 15, 18, 21, 24]                                 #[3, 6, 9, 12, 15, 18] 
b_P_list = [30, 40, 50] # 60, 70, 80, 90] 
solar_MW_list = [35, 45, 55] #[100, 110,120, 130, 140]                         #[25, 35, 45, 55, 65]


## --- LISTS -----------------------------------------------------------
records = []
under_loads_ideal_outputs = {}
under_loads_deg_outputs = {}
over_loads_ideal_outputs = {}
over_loads_deg_outputs = {}
hpp_outputs = {}
hpp_outputs_deg = {}
hpp_t_first_year_all = {} 
results = {profile: {"FLF": [], "LCOE": []} for group in category_profiles.values() for profile in group}
total_capacity_MW_list = []


flf_cplex_list  = []
flf_rule_list   = []
ulf_cplex_list  = []
ulf_rule_list   = []
lcoe_list       = []
component_size_record       = []  
summary_records = []

## --- TIME SETUP ------------------------------------------------------
n_start = int(24*365*0) 
n_days_plot = 365
n_hours = 24*n_days_plot
t = np.arange(n_hours)


# --- RUN THE SIMULATION ---------------------------------------------------
for category, profile_list in category_profiles.items():
    for profile_name in profile_list:
        print(f"Running simulation for profile: {profile_name} in category: {category}")
        
        load_profile_ts_raw = load_profiles_scaled_MW[profile_name]
        load_profile_ts = expand_to_lifetime(
        load_profile_ts_raw,
        life=25*365*24,  # 25 years in hours
        weeks_per_season_per_year=weeks_per_season_per_year
        )

        for nwt, solar_MW_val, b_P_val in product(Nwt_list, solar_MW_list, b_P_list):
            print(f"Running: Nwt = {nwt}, Solar = {solar_MW_val} MW, Battery = {b_P_val} MW …", end=' ')
        
            hpp = hpp_model(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                num_batteries = 10,
                work_dir = './',
                input_ts_fn = input_ts_fn,
                sim_pars_fn=sim_pars_fn,
                load_profile_ts=load_profile_ts,
                load_min_penalty = 1e6,  # MEuro
                battery_deg = True,
            )

            x = [clearance, sp, p_rated, nwt, wind_MW_per_km2,
                solar_MW_val, surface_tilt, surface_azimuth, solar_DCAC,
                b_P_val, b_E_h, cost_of_batt_degr]

            outs = hpp.evaluate(*x)

            #hpp.print_design(x, outs) 

            G_MW = hpp.prob['G_MW'] 
            b_P = hpp.prob['b_P']
            b_E = hpp.prob['b_E']

            b_E_SOC_t = hpp.prob.get_val('ems.b_E_SOC_t')
            b_t = hpp.prob.get_val('ems.b_t')
            price_t = hpp.prob.get_val('ems.price_t')
            wind_t = hpp.prob.get_val('ems.wind_t')
            solar_t = hpp.prob.get_val('ems.solar_t')

            hpp_t = hpp.prob.get_val('ems.hpp_t')
            hpp_curt_t = hpp.prob.get_val('ems.hpp_curt_t')
            grid_MW = hpp.prob.get_val('G_MW')

            hpp_t_first_year = hpp_t[n_start:n_start+24*n_days_plot]  # 24*n_days_plot Replace 'start_date' and 'end_date' with the actual start and end dates of the first year
            wind_t_first_year = wind_t[n_start:n_start+24*n_days_plot]
            solar_t_first_year = solar_t[n_start:n_start+24*n_days_plot]
            batt_t_first_year = b_t[n_start:n_start+24*n_days_plot]
            batt_t_positivie_first_year = np.sum(batt_t_first_year[batt_t_first_year > 0])
            hpp_curt_t_first_year = hpp_curt_t[n_start:n_start+24*n_days_plot]
        
            mismatch_cplex = hpp_t - load_profile_ts
            mismatch_cplex_first_year = mismatch_cplex[n_start:n_start+24*n_days_plot]
            load_profile_ts_first_year = load_profile_ts[n_start:n_start+24*n_days_plot]

            under_load = np.where(mismatch_cplex_first_year < 0, mismatch_cplex_first_year, 0)[n_start:n_start+24*n_days_plot]
            over_load = hpp_curt_t[n_start:n_start+24*n_days_plot]


            under_loads_ideal_outputs[profile_name] = under_load
            over_loads_ideal_outputs[profile_name] = over_load



            CAPEX = hpp.prob.get_val('CAPEX')
            OPEX = hpp.prob.get_val('OPEX')
            NPV = hpp.prob.get_val('NPV')

        ## --- FLF and UFL arrays -------------------------------------
            FLF_array_ideal = np.where(hpp_t_first_year>=load_profile_ts_first_year, ((hpp_t_first_year-load_profile_ts_first_year)/load_profile_ts_first_year), 0)[n_start:n_start+24*n_days_plot]

            ULF_array_ideal = np.where(mismatch_cplex_first_year<0, -mismatch_cplex_first_year/load_profile_ts_first_year, 0)[n_start:n_start+24*n_days_plot]

        ## -- Full load factor --------------------------------------------------------------------------
            CF_cplex = np.sum(hpp_t_first_year >= load_profile_ts_first_year) / len(hpp_t_first_year)
            print(f'CF_cplex = {CF_cplex:.2%}')


        ## -- Under load factor --------------------------------------------------------------------------
            ULF_cplex = np.sum(np.where(mismatch_cplex_first_year<0, -mismatch_cplex_first_year, 0)) / np.sum(load_profile_ts_first_year)
            print(f'ULF_cplex = {ULF_cplex:.2%}')


        ## -- LCOE --------------------------------------------------------------------------
            #LCOE = hpp.prob.get_val('LCOE')
            # Calculate the number of hours where hpp_t is equal to 1.0
            hours_equal_load_min = (hpp_t_first_year >= load_profile_ts_first_year).sum()
            print(f'Hours at load_min = {hours_equal_load_min}')
            hours_less_than_load_min = (hpp_t_first_year < load_profile_ts_first_year*0.99).sum()
            hours_at_load_min = len(hpp_t_first_year[(hpp_t_first_year == load_profile_ts_first_year)])
            total_hours_in_first_year = len(hpp_t_first_year)

            # Calculate the proportion of hours at load_min relative to the total hours in the first year
            GUF_BL = hours_equal_load_min / total_hours_in_first_year #(hours_at_load_min*load_min) / (total_hours_in_first_year*load_min)
            print(f'GUF_BL = {GUF_BL:.2%}')

            AEPy = hours_equal_load_min * load_profile_ts_first_year  
            wind_WACC= 0.052                                                                 # Markup of after tax WACC for onshore WT
            solar_WACC= 0.048                                                               # After tax WACC for solar PV
            battery_WACC= 0.080                                                             # After tax WACC for stationary storge li-ion batteries
            AEP_W = np.sum(wind_t_first_year)
            O_Wfixed = 12600                                                               # Wind fixed O&M cost per year [Euro/MW /year]
            O_Wvar = 1.35                                                                     #[EUR/MWh_e] Danish Energy Agency
            
            C_WIND = hpp.prob.get_val('finance.CAPEX_w')
            O_WIND = hpp.prob.get_val('finance.OPEX_w')
            C_PV = hpp.prob.get_val('finance.CAPEX_s')
            O_PV = hpp.prob.get_val('finance.OPEX_s')
            C_BATT = hpp.prob.get_val('finance.CAPEX_b')
            O_BATT = hpp.prob.get_val('finance.OPEX_b')
            C_EL = hpp.prob.get_val('finance.CAPEX_el')
            O_EL = hpp.prob.get_val('finance.OPEX_el')
            level_costs = hpp.prob.get_val('finance.level_costs')
            
            C_H = C_WIND  + C_PV + C_BATT + C_EL   # Total CAPEX
            O_H = O_WIND  + O_PV + O_BATT + O_EL   # Total OPEX
            
            wind_WACC= 0.052                                                                 # Markup of after tax WACC for onshore WT
            solar_WACC= 0.048                                                               # After tax WACC for solar PV
            battery_WACC= 0.080                                                             # After tax WACC for stationary storge li-ion batteries
            
            WACC_m = (wind_WACC + solar_WACC + battery_WACC ) / 3
            WACC_tx = (C_WIND*wind_WACC + C_PV*solar_WACC + C_BATT*battery_WACC + C_EL*WACC_m)/C_H
            
            # Calculate CL (Capitalized Cost)
            CL =  level_costs
            
            # # Calculate AEPL (Annual Equivalent Power Levelized)
            # 1) Find a boolean mask of “hours where HPP ≥ 0.99·load”
            mask_full = (hpp_t_first_year >= load_profile_ts_first_year)   # array of True/False
            print(f'mask_full = {mask_full.sum()} hours where HPP ≥ load')

            # 2) Multiply that mask *by* the load curve, then sum over time → scalar MWh
            #    (only those hours where mask_full is True will contribute their load value)
            AEPy_scalar = (load_profile_ts_first_year[mask_full]).sum()   # sum of MW‐hours over full‐load hours
            print("AEPy_scalar =", AEPy_scalar, "MWh")
            # 3) Capitalize over 25 years
            # number of years
            n = 25
            # discount rate
            r = WACC_tx
            # annuity factor: sum_{t=1}^n 1/(1+r)^t
            annuity = (1 - (1+r)**(-n)) / r

            # present‐value of energy over n years
            AEPL = AEPy_scalar * annuity

            print("AEPL =", AEPL, "MWh")
            print('WACC_tx =', WACC_tx)
            # Calculate LCoE (Levelized Cost of Electricity)
            LCOE = CL / AEPL
            print("LCOE =", LCOE, "Euro/MWh")

            # PPA VALUATION 
            # 1) slice your first‐year spot price to match your dispatch
            price_first_year = price_t[n_start : n_start + 24*n_days_plot]

            # 2) slice your first‐year HPP dispatch
            dispatch_first_year = hpp_t[n_start : n_start + 24*n_days_plot]

            # 3) CAPTURE PRICE CP = ∑(dispatch * spot_price) / ∑dispatch
            CP = (dispatch_first_year * price_first_year).sum() / dispatch_first_year.sum()

            # 4) BASELOAD PRICE BP = simple average of spot_price
            BP = price_first_year.mean()

            # 5) CAPTURE FACTOR CF = CP / BP
            capture_factor = CP / BP

            # print(f"Capture price (CP) = {CP:.1f} €/MWh")
            # print(f"Baseload price (BP) = {BP:.1f} €/MWh")
            # print(f"Capture factor (CF) = {capture_factor:.3f}")


            # 1) compute under‐load (shortage) for first year
            shortage = np.maximum(load_profile_ts_first_year - hpp_t_first_year, 0)

            shape_cost = (shortage * price_first_year).sum()  
            #print(f"Total shape penalty cost = {shape_cost:,.0f} € (per year)")
            total_annual_demand = load_profile_ts_first_year.sum()    # MWh
            shape_cost_per_MWh = shape_cost / total_annual_demand     # €/MWh of demand
            #print(f"Shape risk premium = {shape_cost_per_MWh:.2f} €/MWh of demand")


            # flf_cplex_list.append(CF_cplex)
            # ulf_cplex_list.append(ULF_cplex)
            # lcoe_list.append(LCOE)
            # component_size_record.append(config_key)

            # extract first‐year dispatch
            hpp_full        = hpp.prob.get_val('ems.hpp_t')
            hpp_1yr         = hpp_full[n_start : n_start + n_hours]
            total_capacity_MW_list.append(nwt * p_rated + solar_MW_val + b_P_val)


            # # store under the **input** battery size
            # hpp_outputs[config_key] = hpp_1yr
            # hpp_outputs_deg[config_key] = hpp_t_25th_year



            results[profile_name]["FLF"].append(CF_cplex)
            results[profile_name]["LCOE"].append(float(LCOE))

            records.append({
                'wind_capacity_MW':   nwt * p_rated,
                'solar_capacity_MW':   solar_MW_val,
                'battery_capacity_MW': b_P_val,
                'CAPEX_€':             CAPEX,
                'OPEX_€_per_yr':       OPEX,
                'LCOE_€_per_MWh':      LCOE,
                'NPV_€':              NPV,
                'CF_cplex':          np.mean(CF_cplex),
            })

            summary_records.append({
                "Profile": profile_name,
                "Category": category,
                "total_capacity_MW": nwt * p_rated + solar_MW_val + b_P_val,
                'wind_capacity_MW':   nwt * p_rated,
                'solar_capacity_MW':   solar_MW_val,
                'battery_capacity_MW': b_P_val,
                "Full Load Factor [%]": round(CF_cplex * 100, 2),
                "Under Load Factor [%]": round(ULF_cplex * 100, 2),
                "LCOE [€/MWh]": round(float(LCOE), 2),
                "Capture price [€/MWh]": round(CP, 1),
                "Baseload price [€/MWh]": round(BP, 1),
                "Capture factor":           round(capture_factor, 3),
                "Shape risk premium [€/MWh]": round(shape_cost_per_MWh, 2),
            })

            flf_cplex_list.append(CF_cplex)
            ulf_cplex_list.append(ULF_cplex)
            lcoe_list.append(LCOE)


            print(f"done → capacity = {nwt*p_rated} MW, LCOE = {LCOE.item():.1f} €/MWh")






# Zet alle resultaten in een DataFrame
df_all = pd.DataFrame(summary_records)


# Voeg afgeleide kolommen toe
df_all["total_capacity_MW"] = (
    df_all["wind_capacity_MW"] +
    df_all["solar_capacity_MW"] +
    df_all["battery_capacity_MW"]
)
df_all["FLF_%"] = df_all["Full Load Factor [%]"]
df_all["ULF_%"] = df_all["Under Load Factor [%]"]
df_all.rename(columns={"LCOE [€/MWh]": "LCOE_€_per_MWh"}, inplace=True)

# Maak map aan om bestanden op te slaan (optioneel)
os.makedirs("excel_per_profile_batt4h_correct_first", exist_ok=True)

# Loop over alle profielen
for profile in df_all["Profile"].unique():
    df = df_all[df_all["Profile"] == profile].copy()
    df_sorted = df.sort_values("total_capacity_MW").reset_index(drop=True)

    excel_df = df_sorted[[
        "total_capacity_MW",
        "wind_capacity_MW",
        "solar_capacity_MW",
        "battery_capacity_MW",
        "LCOE_€_per_MWh",
        "FLF_%",
        "ULF_%"
    ]]

    # Bestandsnaam maken zonder spaties of speciale tekens
    safe_profile_name = profile.replace(" ", "_").replace("/", "-")
    file_path = f"excel_per_profile_batt4h_correct_first/first_{safe_profile_name}.xlsx"

    # Opslaan
    excel_df.to_excel(file_path, index=False)
    print(f"Excel file saved for profile: {profile} → {file_path}")



 #%%   
# Maak map aan om bestanden op te slaan (optioneel)
os.makedirs("excel_per_profile_4h_correct_first", exist_ok=True)



# Voeg afgeleide kolommen toe
df_all["total_capacity_MW"] = (
    df_all["wind_capacity_MW"] +
    df_all["solar_capacity_MW"] +
    df_all["battery_capacity_MW"]
)
df_all["FLF_%"] = df_all["Full Load Factor [%]"]
df_all["ULF_%"] = df_all["Under Load Factor [%]"]
df_all.rename(columns={"LCOE [€/MWh]": "LCOE_€_per_MWh"}, inplace=True)

# … everything up to the Excel export stays the same …

# Loop over all profiles
for profile in df_all["Profile"].unique():
    # 1) pick out this profile’s rows
    df_prof = df_all[df_all["Profile"] == profile].copy()
    df_prof = df_prof.sort_values("total_capacity_MW").reset_index(drop=True)

    # 2) find the baseline *for this profile* (lowest‐LCOE among designs that meet reliability)
    FLF_min = 98.0  # or whatever threshold you use
    candidates = df_prof[df_prof["FLF_%"] >= FLF_min]
    if len(candidates) < 2:
        # Not enough designs to define a delta → set Volume RP to NaN
        df_prof["Volume RP [€/MWh]"] = np.nan
    else:
        baseline = candidates.nsmallest(1, "LCOE_€_per_MWh").iloc[0]
        LCOE_base = baseline["LCOE_€_per_MWh"]
        FLF_base  = baseline["FLF_%"]
        df_prof["Volume RP [€/MWh]"] = (
            df_prof["LCOE_€_per_MWh"] - LCOE_base
        ) / (
            df_prof["FLF_%"] - FLF_base
        )

    # 3) now slice your final export columns (including newly computed Volume RP)
    excel_df = df_prof[[
        "total_capacity_MW",
        "wind_capacity_MW",
        "solar_capacity_MW",
        "battery_capacity_MW",
        "LCOE_€_per_MWh",
        "FLF_%",
        "ULF_%",
        "Capture price [€/MWh]",
        "Baseload price [€/MWh]",
        "Capture factor",
        "Shape risk premium [€/MWh]",
        #"Volume RP [€/MWh]",
    ]]

    # 4) write it out
    safe_profile_name = profile.replace(" ", "_").replace("/", "-")
    file_path = f"excel_per_profile_4h_correct_first/correct_first_{safe_profile_name}.xlsx"
    excel_df.to_excel(file_path, index=False)
    print(f"Excel file saved for profile: {profile} → {file_path}")





# %%

FLF_min = 98.0   # in percent

# find all designs that just meet the reliability requirement
candidates = df_all[df_all["Full Load Factor [%]"] >= FLF_min]

# pick the cheapest (lowest LCOE) among them
baseline = candidates.loc[candidates["LCOE_€_per_MWh"].idxmin()]

LCOE_baseline = baseline["LCOE_€_per_MWh"]
FLF_baseline  = baseline["Full Load Factor [%]"]

df_all["Volume RP [€/MWh]"] = (
      df_all["LCOE_€_per_MWh"] - LCOE_baseline
  ) / (
      df_all["Full Load Factor [%]"] - FLF_baseline
  )
print("\nVolume Risk Premium (RP) [€/MWh]:\n")
# merge back into summary_records if needed
for rec, vol in zip(summary_records, df_all["Volume RP [€/MWh]"]):
    rec["Volume RP [€/MWh]"] = round(vol, 2)


# %% 
# ##-- EXCEL -----------------------------------------------------------------------

# Maak map aan om bestanden op te slaan (optioneel)
os.makedirs("excel_per_profile_4h_second", exist_ok=True)

# Voeg afgeleide kolommen toe
df_all["total_capacity_MW"] = (
    df_all["wind_capacity_MW"] +
    df_all["solar_capacity_MW"] +
    df_all["battery_capacity_MW"]
)
df_all["FLF_%"] = df_all["Full Load Factor [%]"]
df_all["ULF_%"] = df_all["Under Load Factor [%]"]

#df_all.rename(columns={"LCOE [€/MWh]": "LCOE_€_per_MWh"}, inplace=True)


# Loop over alle profielen
for profile in df_all["Profile"].unique():
    df = df_all[df_all["Profile"] == profile].copy()
    df_sorted = df.sort_values("total_capacity_MW").reset_index(drop=True)

    excel_df = df_sorted[[
        "total_capacity_MW",
        "wind_capacity_MW",
        "solar_capacity_MW",
        "battery_capacity_MW",
        "LCOE_€_per_MWh",
        "FLF_%",
        "ULF_%",
        "Capture price [€/MWh]",
        "Baseload price [€/MWh]",
        "Capture factor",
        "Shape risk premium [€/MWh]",
        "Volume RP [€/MWh]",
    ]]

    # Bestandsnaam maken zonder spaties of speciale tekens
    safe_profile_name = profile.replace(" ", "_").replace("/", "-")
    file_path = f"excel_per_profile_4h_second/lcoe_flf_{safe_profile_name}.xlsx"

    # Opslaan
    excel_df.to_excel(file_path, index=False)
    print(f"Excel file saved for profile: {profile} → {file_path}")



