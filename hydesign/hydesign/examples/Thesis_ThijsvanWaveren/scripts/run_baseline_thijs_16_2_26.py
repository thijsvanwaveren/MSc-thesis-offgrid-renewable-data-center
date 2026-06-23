import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Suppress divide by zero warnings from G_MW=0
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- HYDESIGN IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.ems.ems import expand_to_lifetime
from hydesign.assembly.hpp_assembly_offgrid_thijs_2_2_26 import hpp_model_constant_output_offgrid as hpp_model
from hydesign.examples.Thesis_ThijsvanWaveren.models.datacenter import DataCenterModel

def configure_parameters(thesis_dir):
    """
    Patches the simulation parameters to ensure:
    1. Grid Capacity (G_MW) = 0 (Off-grid, no ghost cost)
    2. Battery Efficiency = 86% Round Trip (~92.7% One-way)
    Returns path to the temporary patched YAML file.
    """
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)

    sim_pars['G_MW'] = 0
    sim_pars['battery_charge_efficiency'] = float(np.sqrt(0.86))

    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_offgrid_sweep_temp.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    
    return temp_fn

def plot_yearly_overview(df, title):
    """
    Plots a Yearly Heatmap of Battery Status.
    X-Axis: Day of Year | Y-Axis: Hour of Day
    Color: SOC % (Green = Full, Red = Empty)
    """
    # 1. Prepare Data (Reshape 8760 hours -> 365 days x 24 hours)
    n_days = 365
    # Take first year of data
    soc = df['SOC'].values[:n_days*24]
    # Reshape: Rows=Hour (0-23), Cols=Day (0-364)
    # We transpose (.T) so X is Day and Y is Hour
    soc_matrix = soc.reshape(n_days, 24).T
    
    # 2. Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Heatmap: RdYlGn (Red=Low SOC, Green=High SOC)
    im = ax.imshow(soc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100, interpolation='nearest', origin='lower')
    
    # 3. Styling
    ax.set_title(f"Yearly Operation Fingerprint: {title}\n(Color = Battery State of Charge)", fontsize=14)
    ax.set_ylabel("Hour of Day (0=Midnight, 12=Noon)", fontsize=12)
    ax.set_xlabel("Day of Year", fontsize=12)
    
    # Format X-Axis (Months)
    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names)
    
    # Format Y-Axis
    ax.set_yticks([0, 6, 12, 18, 23])
    
    # Colorbar
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Battery SOC (%)", rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()
    
def summarize_findings_detailed(df_res):
    """
    Prints a detailed text table comparing the different reliability metrics
    for the optimal design (11-hour battery) in each scenario, including C2 sponge effects.
    """
    print("\n" + "="*105)
    print(f"{'DETAILED RELIABILITY BREAKDOWN (Optimal Design)':^105}")
    print("="*105)
    print(f"{'Scenario':<25} | {'Batt':<5} | {'LCOE':<6} | {'Rel(En)':<8} | {'Rel(A-Time)':<11} | {'Rel(B-Dead)':<11} | {'Curt(GWh)':<9} | {'C2(GWh)':<9}")
    print("-" * 105)
    
    scenarios = df_res['Scenario'].unique()
    
    for scen in scenarios:
        # Filter designs for this scenario
        df_scen = df_res[df_res['Scenario'] == scen]
        
        # Hardcode the summary to only show the 11-hour battery design
        best_df = df_scen[df_scen['Battery_Hours'] == 11]
        
        # Safety check in case the 11-hour simulation failed or wasn't in the sweep
        if best_df.empty:
            print(f"{scen:<25} | 11-hour battery data not found.")
            continue
            
        best = best_df.iloc[0]
        
        # Using .get() to prevent errors if the columns aren't there
        curt_val = best.get('Curtailment_GWh', 0.0)
        c2_val = best.get('C2_GWh', 0.0)
        
        print(f"{scen:<25} | {int(best['Battery_Hours']):>2}h  | {best['LCOE']:>6.2f} | "
              f"{best['Reliability_Energy_Total']:>7.2f}% | "
              f"{best['Reliability_Time_A']:>10.2f}% | "
              f"{best['Reliability_Deadline_B']:>10.2f}% | "
              f"{curt_val:>9.2f} | {c2_val:>9.2f}")
        
    print("="*105)
    print("Metric Definitions:")
    print("  Rel(En):      Total Energy Served / Total Demand (System-wide Efficiency)")
    print("  Rel(A-Time):  % of HOURS where Critical Tier A was fully served (Uptime)")
    print("  Rel(B-Dead):  % of DAYS where Tier B finished its bucket by midnight")
    print("  Curt(GWh):    Energy physically thrown away (Absolute waste)")
    print("  C2(GWh):      Excess energy successfully absorbed by Opportunistic compute")
    print("="*105 + "\n")


def plot_duration_curves(df, title):
    """
    Plots Stacked Duration Curves for both Generation and Consumption.
    Sorted by Total System Power to maintain the physical balance of each hour.
    """
    # 1. Calculate Total Power (Generation = Consumption)
    # Using .get() avoids errors if a column is missing in earlier scenarios
    gen_total = df.get('Solar', 0) + df.get('Wind', 0) + df.get('Batt_Discharge', 0)
    
    # 2. Sort the dataframe by the total generation in descending order
    # This gives us a smooth top envelope, while preserving the actual hourly mix inside
    sort_idx = gen_total.sort_values(ascending=False).index
    df_sorted = df.loc[sort_idx].reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # --- CHART 1: GENERATION DURATION CURVE ---
    ax1.stackplot(df_sorted.index,
                  df_sorted.get('Solar', 0),
                  df_sorted.get('Wind', 0),
                  df_sorted.get('Batt_Discharge', 0),
                  labels=['Solar PV', 'Wind', 'Batt Discharge'],
                  colors=['#FDB813', '#87CEEB', 'purple'], alpha=0.8)
                  
    ax1.set_title(f"Generation Duration Curve\n{title}", fontsize=14)
    ax1.set_xlabel("Hours of the Year (Sorted by Total Power)", fontsize=12)
    ax1.set_ylabel("Power (MW)", fontsize=12)
    ax1.set_xlim(0, 8760)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # --- CHART 2: CONSUMPTION DURATION CURVE ---
    ax2.stackplot(df_sorted.index,
                  df_sorted.get('Load_Fixed_Served', 0),
                  df_sorted.get('Load_Flexible_Served', 0),
                  df_sorted.get('Served_C2', 0),
                  df_sorted.get('Batt_Charge', 0),
                  df_sorted.get('Curtailment', 0),
                  labels=['Tier A (Baseload)', 'Tier B (Schedulable)', 'Tier C2 (Opportunistic)', 'Batt Charge', 'Curtailment'],
                  colors=['silver', 'orange', '#00CC96', 'green', 'gray'], alpha=0.8)
                  
    ax2.set_title(f"Consumption Duration Curve\n{title}", fontsize=14)
    ax2.set_xlabel("Hours of the Year (Sorted by Total Power)", fontsize=12)
    ax2.set_xlim(0, 8760)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
def plot_dispatch_zoom(df, title, window_days=7, start_idx=8000):
    """
    Detailed dispatch plot using Stackplot.
    Now explicitly highlights dropped critical loads (Unserved_A) in bright RED.
    """
    end_idx = start_idx + (24 * window_days)
    d = df.iloc[start_idx:end_idx].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # --- TOP: Supply & Demand ---
    # 1. Supply Stack (Generation)
    ax1.stackplot(d.index, d['Solar'], d['Wind'], labels=['Solar PV', 'Wind'], 
                  colors=['#FDB813', '#87CEEB'], alpha=0.7)
    
    # 2. Battery Discharge (Overlay)
    ax1.bar(d.index, d['Batt_Discharge'], width=0.04, color='purple', alpha=0.6, label='Batt Discharge')

    # 3. Demand Stack (Now with Unserved A in RED)
    ax1.stackplot(d.index, 
                  d['Load_Fixed_Served'],      # Bottom: Gray (Tier A)
                  d['Load_Flexible_Served'],   # Middle: Orange (Tier B)
                  d['Served_C2'],              # Top: Green (Tier C2)
                  d['Unserved_A'],             # OVER THE TOP: Red (CRITICAL FAIL)
                  labels=['Fixed (A)', 'Flex (B)', 'Opportunistic (C2)', 'Unserved A (FAIL)'], 
                  colors=['silver', 'orange', '#00CC96', 'red'], alpha=0.8)

    # Total Target Load Line (Black Line)
    # We add Unserved_A back to the Total Served to show where the line *should* have been
    target_load = d['Load_Total_Served'] + d['Unserved_A']
    ax1.step(d.index, target_load, where='mid', color='black', linewidth=1.5, label='Target Load')

    # Battery Charging (Negative)
    ax1.bar(d.index, -d['Batt_Charge'], width=0.04, color='green', alpha=0.6, label='Batt Charging')
    
    ax1.set_ylabel('Power (MW)')
    ax1.set_title(f"Dispatch Zoom: {title}")
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    ax1.grid(True, alpha=0.3)
    
    # --- BOTTOM: SOC ---
    ax2.plot(d.index, d['SOC'], color='green', linewidth=2)
    ax2.fill_between(d.index, 0, d['SOC'], color='green', alpha=0.1)
    ax2.set_ylabel('SOC (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    
    plt.tight_layout()
    plt.show()
    
def plot_weekly_energy_bar(df, title):
    """
    Plots a weekly aggregated energy balance (MWh) to show seasonal dynamics.
    Above axis: Generation (Solar, Wind) + Battery Discharge
    Below axis: Loads (A, B, C2) + Battery Charge + Curtailment
    """
    # Resample the hourly data to weekly sums (MWh)
    df_w = df.resample('W').sum()
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # --- POSITIVE BARS (Supply / Energy In) ---
    ax.bar(df_w.index, df_w['Solar'], width=5, label='Solar', color='#FDB813')
    ax.bar(df_w.index, df_w['Wind'], width=5, bottom=df_w['Solar'], label='Wind', color='#87CEEB')
    ax.bar(df_w.index, df_w['Batt_Discharge'], width=5, bottom=df_w['Solar']+df_w['Wind'], label='Batt Discharge', color='purple', alpha=0.6)
    
    # --- NEGATIVE BARS (Demand / Energy Out) ---
    # Keep track of the bottom to stack downwards
    bottom_neg = np.zeros(len(df_w))
    
    ax.bar(df_w.index, -df_w['Load_Fixed_Served'], width=5, label='Fixed Load (A)', color='silver')
    bottom_neg -= df_w['Load_Fixed_Served']
    
    if 'Load_Flexible_Served' in df_w.columns:
        ax.bar(df_w.index, -df_w['Load_Flexible_Served'], width=5, bottom=bottom_neg, label='Flexible Load (B)', color='orange')
        bottom_neg -= df_w['Load_Flexible_Served']
        
    if 'Served_C2' in df_w.columns:
        ax.bar(df_w.index, -df_w['Served_C2'], width=5, bottom=bottom_neg, label='Opp. Load (C2)', color='#00CC96')
        bottom_neg -= df_w['Served_C2']
        
    ax.bar(df_w.index, -df_w['Batt_Charge'], width=5, bottom=bottom_neg, label='Batt Charge', color='green', alpha=0.6)
    bottom_neg -= df_w['Batt_Charge']
    
    if 'Curtailment' in df_w.columns:
        ax.bar(df_w.index, -df_w['Curtailment'], width=5, bottom=bottom_neg, label='Curtailment (Waste)', color='gray', alpha=0.3)
        bottom_neg -= df_w['Curtailment']

    # Explicitly highlight failed critical loads pulling down the stack
    if 'Unserved_A' in df_w.columns:
        ax.bar(df_w.index, -df_w['Unserved_A'], width=5, bottom=bottom_neg, label='Unserved A (FAIL)', color='red')
        
    ax.axhline(0, color='black', linewidth=1.5) # Zero line
    ax.set_ylabel("Total Weekly Energy (MWh)", fontsize=12)
    ax.set_title(f"Seasonal Energy Balance: {title}", fontsize=14)
    
    # Put legend outside the plot so it doesn't block data
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')) # Format x-axis as Months
    
    plt.tight_layout()
    plt.show()    

def run_sensitivity_sweep():
    # --- 1. CONFIGURATION ---
    scenarios = [
        "Not_Flexible",           
        "Interactive_Focused",
        "Interactive_Traditional", 
        "Batch_Traditional",
        "Batch_Focused",          
        "Infinitely_Flexible"     
    ]
    
    batt_duration_sweep = [0,2, 4, 6, 8, 11, 16, 20, 24] 
    fixed_design_base = [35, 300, 5, 10, 7, 112, 39, 180, 1.25, 40] 
    batt_cost_factor = 10
    
    # Setup Paths
    site_name = 'Denmark_good_wind'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)
    
    dc_model = DataCenterModel(total_it_capacity=20, pue=1.15)
    results = []
    
    print(f"--- Starting Sensitivity Sweep ---")
    
    # --- 2. MAIN LOOP ---
    for scen in scenarios:
        print(f"\nProcessing Scenario: {scen}")
        
        t_a_1y, t_b_1y = dc_model.generate_profile(scen)
        t_a_25y = expand_to_lifetime(t_a_1y, life=25*8760)
        t_b_input_25y = np.tile(np.repeat(t_b_1y, 24), 25)
        
        # Calculate Tier B Daily Targets for Reliability Math
        # (Needed to count failed days later)
        # We can infer daily target from the first hour of the day in t_b_input
        # t_b_input_25y is hourly, but repeated daily values.
        daily_targets_25y = t_b_input_25y[::24] 

        t_b_hourly = np.tile(np.repeat(t_b_1y / 24.0, 24), 25)
        total_load_25y = t_a_25y + t_b_hourly
        
        # Magic Spike (80% Util)
        total_load_for_ems = total_load_25y.copy()
        total_load_for_ems[0] = 23/0.8
        
        total_demand_MWh = np.sum(total_load_25y) 

        hpp = hpp_model(
            latitude=ex_site['latitude'].values[0],
            longitude=ex_site['longitude'].values[0],
            altitude=ex_site['altitude'].values[0],
            num_batteries=1,
            work_dir=current_dir,
            input_ts_fn=weather_fn,
            sim_pars_fn=sim_pars_fn,
            tier_a_profile=t_a_25y,
            tier_b_profile=t_b_input_25y,
            load_profile_ts=total_load_for_ems, 
            battery_deg=False
        )

        best_design_for_plot = None
        best_rel_diff = 1.0 

        # C. Inner Loop: Sweep Battery
        for b_h in batt_duration_sweep:
            current_design = fixed_design_base + [b_h, batt_cost_factor]
            
            out = hpp.evaluate(*current_design)
            
            prob = hpp.prob
            lcoe = out[3]
            
            # --- EXTRACT METRICS ---
            unserved_a_ts = prob.get_val('ems.Unserved_A')
            shortfall_b_ts = prob.get_val('ems.Shortfall_B') # SLA Violations
            
            # Derive actual served energy from the total HPP output
            hpp_out_ts = prob.get_val('ems.hpp_t')
            served_c2_ts = prob.get_val('ems.Served_C2')
            
            served_a_ts = t_a_25y - unserved_a_ts
            served_b_ts = hpp_out_ts - served_a_ts - served_c2_ts
            
            # 1. Total Energy Reliability (Mathematically Flawless)
            total_served_primary_MWh = np.sum(served_a_ts) + np.sum(served_b_ts)
            rel_energy_total = total_served_primary_MWh / total_demand_MWh
            
            # 2. Tier A: Time-Based Reliability (Uptime)
            # Count hours where unserved > 1e-3
            total_hours = len(unserved_a_ts)
            failed_hours_a = np.sum(unserved_a_ts > 1e-3)
            rel_time_a = 1.0 - (failed_hours_a / total_hours)
            
            # 3. Tier B: Deadline Reliability
            # The EMS output 'Shortfall_B' is spread over the day.
            # If a day has Shortfall > 0, the deadline was missed.
            # Convert hourly series to daily sum
            shortfall_b_daily_sum = np.sum(shortfall_b_ts.reshape(-1, 24), axis=1)
            # Compare against daily targets to see if it was a partial failure or full
            # Simplest metric: If Shortfall > 1e-3, Day Failed.
            failed_days_b = np.sum(shortfall_b_daily_sum > 1e-3)
            total_days = len(shortfall_b_daily_sum)
            
            # --- EXTRACT WASTE & OPPORTUNISTIC SPONGE ---
            total_curt_GWh = np.sum(prob.get_val('ems.hpp_curt_t')) / 1000.0
            total_c2_GWh = np.sum(prob.get_val('ems.Served_C2')) / 1000.0
            
            # Handle edge case if Tier B doesn't exist in this scenario
            if np.sum(daily_targets_25y) > 1:
                rel_deadline_b = 1.0 - (failed_days_b / total_days)
            else:
                rel_deadline_b = 1.0 # 100% success (N/A)

            results.append({
                "Scenario": scen,
                "Battery_Hours": b_h,
                "LCOE": lcoe,
                "Reliability_Energy_Total": rel_energy_total * 100,
                "Reliability_Time_A": rel_time_a * 100,
                "Reliability_Deadline_B": rel_deadline_b * 100,
                "Curtailment_GWh": total_curt_GWh,
                "C2_GWh": total_c2_GWh # <--- NEW
            })
            
            print(f"   -> Batt: {b_h:2d}h | LCOE: {lcoe:.2f} | En: {rel_energy_total*100:5.2f}% | A-Time: {rel_time_a*100:5.2f}% | B-Dead: {rel_deadline_b*100:5.2f}%")
            
            # Save 11 hour battery capacity for plotting (using Total Energy Reliability)
            if b_h == 11:
                
                N_h = 8760
                
                # Re-extract for plot
                hpp_out = prob.get_val('ems.hpp_t')[:N_h]
                unserved_a = prob.get_val('ems.Unserved_A')[:N_h]
                served_c2 = prob.get_val('ems.Served_C2')[:N_h]
                
                load_fixed_target = t_a_25y[:N_h]
                load_fixed_served = load_fixed_target - unserved_a
                
                load_flexible_served = hpp_out - load_fixed_served - served_c2
                load_flexible_served = np.maximum(0, load_flexible_served)

                df_plot = pd.DataFrame({
                    'Wind': prob.get_val('ems.wind_t_ext')[:N_h],
                    'Solar': prob.get_val('ems.solar_t_ext')[:N_h],
                    'HPP_Out': hpp_out,
                    'SOC': prob.get_val('ems.b_E_SOC_t')[:N_h] / (40 * b_h) * 100,
                    'Batt_Power': prob.get_val('ems.b_t')[:N_h],
                    'Unserved_A': unserved_a,
                    'Load_Fixed_Served': load_fixed_served,
                    'Load_Flexible_Served': load_flexible_served,
                    'Served_C2': served_c2,
                    'Load_Total_Served': hpp_out,
                    'Curtailment': prob.get_val('ems.hpp_curt_t')[:N_h] # <--- ADD THIS LINE
                })
                df_plot['Batt_Discharge'] = df_plot['Batt_Power'].clip(lower=0)
                df_plot['Batt_Charge'] = df_plot['Batt_Power'].clip(upper=0).abs()
                df_plot.index = pd.date_range(start='2025-01-01', periods=N_h, freq='h')
                
                best_design_for_plot = df_plot
                best_b_h = b_h

        # D. Plotting Dispatch
        print(f"   [PLOT] Generating plots for {scen} (Battery: {best_b_h}h)")
        plot_yearly_overview(best_design_for_plot, f"{scen} (Batt: {best_b_h}h)")
        plot_weekly_energy_bar(best_design_for_plot, f"{scen} (Batt: {best_b_h}h)") # <--- Add this
        plot_dispatch_zoom(best_design_for_plot, f"{scen} (Batt: {best_b_h}h)", window_days=14, start_idx=3600)
        plot_duration_curves(best_design_for_plot, f"{scen} (Batt: {best_b_h}h)") # <--- Add this

    # --- 3. FINAL ANALYSIS & PLOTS ---
    df_res = pd.DataFrame(results)
    
    # 3A. Print Text Summary
    summarize_findings_detailed(df_res)
    
    # 3B. Plot 1: Total Energy Reliability vs LCOE
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_res, x="Reliability_Energy_Total", y="LCOE", hue="Scenario", style="Scenario", markers=True, dashes=False, linewidth=2, markersize=8)
    plt.axvline(x=99, color='red', linestyle='--', alpha=0.5, label='99% Target')
    plt.title("The Cost of Reliability: Total Energy Delivered", fontsize=14)
    plt.xlabel("Total Energy-Based Reliability (%)", fontsize=12)
    plt.ylabel("LCOE (€/MWh)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(max(0, df_res['Reliability_Energy_Total'].min() - 5), 100.5) 
    plt.tight_layout()
    plt.show()
    
    # 3C. Plot 2: Tier A Time-Based Reliability vs LCOE (NEW)
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_res, x="Reliability_Time_A", y="LCOE", hue="Scenario", style="Scenario", markers=True, dashes=False, linewidth=2, markersize=8)
    plt.axvline(x=99, color='red', linestyle='--', alpha=0.5, label='99% Uptime Target')
    
    # Add Zoom-in text annotation if reliability is very high
    if df_res['Reliability_Time_A'].min() > 90:
        plt.xlim(90, 100.1)
        
    plt.title("The Cost of Uptime: Critical Tier A Reliability", fontsize=14)
    plt.xlabel("Tier A Time-Based Reliability (Uptime %)", fontsize=12)
    plt.ylabel("LCOE (€/MWh)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sensitivity_sweep()