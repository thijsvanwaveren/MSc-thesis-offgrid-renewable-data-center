import os
import sys
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from hydesign.ems.ems import expand_to_lifetime
# Import your specific offgrid assembly
from hydesign.assembly.hpp_assembly_offgrid_thijs_2_2_26 import hpp_model_constant_output_offgrid as hpp_model
from hydesign.examples.Thesis_ThijsvanWaveren.models.datacenter import DataCenterModel 

import importlib.util


# --- 1. DYNAMIC PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

# --- 2. CONFIGURE SITE & PARAMETERS ---
site_name = 'Denmark_good_wind'
examples_dir = os.path.abspath(os.path.join(thesis_dir, '..'))
sites_csv_path = os.path.join(examples_dir, 'examples_sites.csv')
examples_sites = pd.read_csv(sites_csv_path, sep=';')
ex_site = examples_sites.loc[examples_sites.name == site_name]
weather_fn = os.path.join(examples_dir, ex_site['input_ts_fn'].values[0])

# --- DYNAMIC PARAMETER UPDATE ---
# 1. Load the original YAML
original_pars_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
with open(original_pars_fn, 'r') as f:
    sim_pars = yaml.safe_load(f)

# 2. [FIX] Force Grid to Zero (Removes Ghost Cost)
print(f"⚠️ Overwriting G_MW (Old: {sim_pars.get('G_MW')}) to 0 for Off-Grid Simulation")
sim_pars['G_MW'] = 0 

# 3. [FIX] Force Battery Efficiency to 86% Round-Trip (Matches Sizing Script)
eff_one_way = np.sqrt(0.86) # ~0.9273
print(f"⚠️ Overwriting Battery Efficiency (Old: {sim_pars.get('battery_charge_efficiency')}) to {eff_one_way:.4f}")
sim_pars['battery_charge_efficiency'] = float(eff_one_way)

# 4. Save to temporary file
temp_pars_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_offgrid_temp.yml')
with open(temp_pars_fn, 'w') as f:
    yaml.dump(sim_pars, f)

# 5. Use the temporary file for the simulation
sim_pars_fn = temp_pars_fn

# --- 3. SETUP DATA CENTER & PROFILES ---
total_it_capacity = 20 # MW
dc_model = DataCenterModel(total_it_capacity=total_it_capacity, pue=1.15)
scenario = "Not_Flexible" 
print(f"🏢 Generating Workloads for Scenario: {scenario}")

# Generate 1-year profiles (Tier A = MW, Tier B = MWh/day)
tier_a_1yr, tier_b_1yr_daily = dc_model.generate_profile(scenario)

# --- EXPAND TO LIFETIME (25 Years) ---
# Tier A (Hourly): Expand 8760 -> 219,000
tier_a_25y = expand_to_lifetime(tier_a_1yr, life=25*8760)

# Tier B (Daily -> Hourly): 
# 1. Convert Daily MWh bucket to Hourly Average MW (just for shape/total calculation)
#    Note: Tier B is flexible, but for the "Total Target" profile, we just need the volume.
tier_b_1yr_hourly_avg = np.repeat(tier_b_1yr_daily / 24.0, 24) 

# 2. Tile for 25 years
tier_b_25y_hourly = np.tile(tier_b_1yr_hourly_avg, 25)

# 3. Create the Tier B INPUT for EMS (Daily buckets repeated)
#    The EMS expects the daily bucket value repeated 24 times for each hour of that day
tier_b_25y_input = np.tile(np.repeat(tier_b_1yr_daily, 24), 25)

# --- CREATE TOTAL LOAD PROFILE FOR FINANCE ---
# This ensures the Finance module knows the total demand to calculate AEP correctly
total_load_profile_25y = tier_a_25y + tier_b_25y_hourly

print(f"✅ Profiles Ready.")
print(f"   Tier A Shape: {tier_a_25y.shape}")
print(f"   Tier B Shape: {tier_b_25y_input.shape}")
print(f"   Total Load Average: {np.mean(total_load_profile_25y):.2f} MW")


# --- 4. INITIALIZE MODEL ---
print("⚙️ Initializing HPP Model...")

hpp = hpp_model(
    latitude=ex_site['latitude'].values[0],
    longitude=ex_site['longitude'].values[0],
    altitude=ex_site['altitude'].values[0],
    num_batteries=2,
    work_dir=current_dir,
    input_ts_fn=weather_fn,
    sim_pars_fn=sim_pars_fn,
    
    # PASS THE PROFILES FOR EMS
    tier_a_profile=tier_a_25y,
    tier_b_profile=tier_b_25y_input,
    
    # PASS THE TOTAL LOAD FOR FINANCE (Crucial for LCOE!)
    load_profile_ts=total_load_profile_25y,
    
    battery_deg=False,
)

# --- 5. RUN SIZING LOOP --- Can be either 1 result from the sizing constant dc load or several options to find a more suitable result the case of workload flexibility.
bess_sizes = [11] # Hours

print(f"\n🚀 Starting Sizing Loop for {scenario}...")

for b_hours in bess_sizes:
    print(f"\n🔋 Testing BESS Duration: {b_hours} hours")
    # [clearance, sp, p_rated, Nwt, wind_MW_km2, solar_MW, tilt, azimuth, DCAC, b_P, b_E_h, cost_deg]
    design_vector = [35, 300, 5, 10, 7, 112, 39, 180, 1.25, 40, b_hours, 10]

    # Run Evaluation
    results = hpp.evaluate(*design_vector)
    
    # Extract Results
    prob = hpp.prob
    
    
    # 1. Reliability (Based on EMS Slack Variables)
    unserved_a = prob.get_val('ems.Unserved_A') 
    shortfall_b = prob.get_val('ems.Shortfall_B')
    
    total_unserved_MWh = np.sum(unserved_a) + np.sum(shortfall_b)
    total_demand_MWh = np.sum(total_load_profile_25y)
    
    reliability = 1.0 - (total_unserved_MWh / total_demand_MWh)
    
    # 2. LCOE (Based on Finance Module)
    # Since we passed 'load_profile_ts', this should now be correct.
    # It calculates Cost / (Served Energy). 
    lcoe = results[3]

    print(f"   RESULTS:")
    print(f"   📉 Total Unserved: {total_unserved_MWh:.2f} MWh / 25y")
    print(f"   ✅ Reliability:    {reliability*100:.3f}%")
    print(f"   💰 LCOE:           {lcoe:.2f} EUR/MWh")
    
    
# =========================================================
# 🔍 DETAILED ANALYSIS OF THE FINAL RUN (e.g., 48h Battery)
# =========================================================
print("\n" + "="*60)
print(f"📊 DEEP DIVE ANALYSIS (Design: {b_hours}h Battery)")
print("="*60)

# 1. Retrieve Time Series Data
# ----------------------------
# Inputs
wind_ts = prob.get_val('ems.wind_t_ext')
solar_ts = prob.get_val('ems.solar_t_ext')
price_ts = prob.get_val('ems.price_t_ext')

# System State
hpp_power_ts = prob.get_val('ems.hpp_t')           # Total Power Delivered to DC
batt_soc_ts  = prob.get_val('ems.b_E_SOC_t')[:-1]  # State of Charge (remove last index)
batt_flow_ts = prob.get_val('ems.b_t')             # + Discharging / - Charging
curtail_ts   = prob.get_val('ems.hpp_curt_t')      # Wasted energy

# Reliability Metrics (Slack Variables)
unserved_a_ts = prob.get_val('ems.Unserved_A')
shortfall_b_ts = prob.get_val('ems.Shortfall_B')   # This is smeared daily average in current logic

# 2. Calculate Tier Statistics
# ----------------------------
total_hours = len(hpp_power_ts)
total_years = 25

# Tier A (Rigid) Analysis
target_a_total = np.sum(tier_a_25y)
unmet_a_total = np.sum(unserved_a_ts)
avail_a_pct = 100 * (1 - unmet_a_total / target_a_total)
hours_unserved_a = np.sum(unserved_a_ts > 0.01) # Count hours with >10kW failure

# Tier B (Flexible) Analysis
target_b_total = np.sum(tier_b_25y_input)/24 # This input has the daily buckets repeated hourly
unmet_b_total = np.sum(shortfall_b_ts)
avail_b_pct = 100 * (1 - unmet_b_total / target_b_total)

print(f"\n📈 LIFETIME GENERATION STATS:")
print(f"   Total Generation:   {np.sum(hpp_power_ts)/1e6:.2f} TWh")
print(f"   Total Curtailment:  {np.sum(curtail_ts)/1e6:.2f} TWh")
print(f"   Battery Cycling:    {np.sum(np.abs(batt_flow_ts))/2/1e6:.2f} TWh (Throughput)")

print(f"\n📉 RELIABILITY BREAKDOWN:")
print(f"   [Tier A - Baseload]: {avail_a_pct:.4f}% Availability")
print(f"      - Unserved Energy: {unmet_a_total:.2f} MWh")
print(f"      - Outage Hours:    {hours_unserved_a} hours / 25y")
print(f"   [Tier B - Batch]:    {avail_b_pct:.4f}% Availability")
print(f"      - Shortfall Energy:{unmet_b_total:.2f} MWh")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # Optional, makes heatmaps nicer (pip install seaborn)

# %%


# =========================================================
# 📅 YEARLY PERFORMANCE DASHBOARD (Year 1 Snapshot)
# =========================================================
print("\n" + "="*60)
print(f"🌍 YEAR 1 ANALYSIS (Design: {b_hours}h Battery)")
print("="*60)

# --- 1. PREPARE DATA ---
# Extract just the first year (8760 hours)
N_h = 8760
wind_y1 = prob.get_val('ems.wind_t_ext')[:N_h]
solar_y1 = prob.get_val('ems.solar_t_ext')[:N_h]
hpp_gen_y1 = prob.get_val('ems.hpp_t')[:N_h]
# SOC has N+1 points, take first 8760
soc_y1 = prob.get_val('ems.b_E_SOC_t')[:N_h] / (42 * b_hours) * 100 

# Reliability Slack Variables
unserved_a_y1 = prob.get_val('ems.Unserved_A')[:N_h]
shortfall_b_y1 = prob.get_val('ems.Shortfall_B')[:N_h]
total_unserved_y1 = unserved_a_y1 + shortfall_b_y1

# Create a DataFrame for easy resampling
df_year = pd.DataFrame({
    'Wind': wind_y1,
    'Solar': solar_y1,
    'Generation': hpp_gen_y1,
    'SOC': soc_y1,
    'Unserved': total_unserved_y1,
    'Load_Target': total_load_profile_25y[:N_h]
})
df_year.index = pd.date_range(start='2025-01-01', periods=N_h, freq='h')

# --- 2. PLOT A: SEASONAL TRENDS (Weekly Rolling Average) ---
# Resample to 7-day averages to smooth out daily noise and see seasonal patterns
df_weekly = df_year.resample('7D').mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top Panel: Power
ax1.plot(df_weekly.index, df_weekly['Load_Target'], color='red', linestyle='--', label='Target Load (Avg)', linewidth=2)
ax1.plot(df_weekly.index, df_weekly['Generation'], color='navy', label='HPP Output (Avg)', linewidth=2)
ax1.fill_between(df_weekly.index, df_weekly['Generation'], color='navy', alpha=0.1)
ax1.set_ylabel('Power (MW)')
ax1.set_title('Seasonal Performance: Generation vs Load (Weekly Average)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Bottom Panel: Battery State of Health (Seasonal Storage)
ax2.plot(df_weekly.index, df_weekly['SOC'], color='green', label='Avg Weekly SOC %', linewidth=2)
ax2.fill_between(df_weekly.index, 0, df_weekly['SOC'], color='green', alpha=0.1)
ax2.set_ylabel('Battery SOC (%)')
ax2.set_ylim(0, 100)
ax2.set_title('Seasonal Storage Status: How full is the battery on average?')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 3. PLOT B: THE "OUTAGE MAP" (Heatmap) ---
# This shows exactly WHEN the system failed.
# X-axis = Day of Year, Y-axis = Hour of Day
# Color = MW Unserved

# Pivot data: Rows=Hour (0-23), Cols=Day (0-364)
# We handle leap years/length mismatch by truncating to 365 days * 24h
matrix_unserved = total_unserved_y1[:365*24].reshape(365, 24).T 

plt.figure(figsize=(14, 6))
# If you don't have seaborn, use plt.imshow
try:
    import seaborn as sns
    sns.heatmap(matrix_unserved, cmap="Reds", cbar_kws={'label': 'Unserved Power (MW)'}, vmin=0)
    plt.title('Outage Heatmap: Red = Blackout / Failure')
    plt.xlabel('Day of Year (0 = Jan 1st, 365 = Dec 31st)')
    plt.ylabel('Hour of Day')
except ImportError:
    plt.imshow(matrix_unserved, aspect='auto', cmap='Reds', vmin=0, vmax=np.max(matrix_unserved))
    plt.colorbar(label='Unserved Power (MW)')
    plt.title('Outage Heatmap (Red = Failure)')
    plt.xlabel('Day of Year')
    plt.ylabel('Hour of Day')

plt.tight_layout()
plt.show()

# --- 4. TEXT REPORT ---
total_hours_unserved = np.sum(total_unserved_y1 > 0.1)
worst_day = df_year['Unserved'].resample('D').sum().idxmax()
worst_day_val = df_year['Unserved'].resample('D').sum().max()

print(f"\n📝 YEAR 1 SUMMARY:")
print(f"   - Hours with Outages: {total_hours_unserved} hours ({(total_hours_unserved/8760)*100:.2f}%)")
print(f"   - Worst Day:          {worst_day.date()} (Shortfall: {worst_day_val:.2f} MWh)")
print(f"   - Average SOC:        {df_year['SOC'].mean():.1f}%")
print(f"   - Time at 0% SOC:     {np.sum(df_year['SOC'] < 1):.0f} hours")