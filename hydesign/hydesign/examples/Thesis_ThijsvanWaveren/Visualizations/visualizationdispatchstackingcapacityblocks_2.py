# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:42:24 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Final Thesis Visualization: Workload Stacking & Queue Dynamics
Demonstrates Valley Filling across Weeks 7, 8, and 9.
Separates Annual Firm Baseload (7 MW) from Weekly Capacity Blocks.
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- HYDESIGN IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.assembly.assembly_rolling_horizon_thijs_10_3_26 import hpp_model_constant_output_offgrid as hpp_model

def configure_parameters(thesis_dir):
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    sim_pars['G_MW'] = 0
    sim_pars['battery_charge_efficiency'] = float(np.sqrt(0.86))
    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_offgrid_sweep_temp.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    return temp_fn

def run_visualization():
    # =========================================================================
    # CONFIGURATION & HYPERPARAMETERS
    # =========================================================================
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    
    MAX_IT_CAPACITY_MW = 55.0  # Strict hardware ceiling
    ANNUAL_FIRM_BASE   = 7.0   # Flat baseload foundation
    TIER_B1_MW         = 1.0   # Target daily batch
    TIER_B2_MW         = 20.0  # Target weekly capacity blocks (Set to 10 MW for clean valley-filling)
    
    N_life_1yr = 8760
    N_life_total = 25 * 8760
    
    # Environment Setup
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    # Disable C2 Opportunistic Load entirely to prevent math loopholes
    os.environ['REWARD_C2'] = '1000.0'  

    # =========================================================================
    # 1. LOAD CSV AND BUILD STRICT PROFILES
    # =========================================================================
    csv_fn = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\Results\Weekly\annual simulation weekly result\Weekly_Capacity_Blocks_yearlysimulation.csv"
    try:
        df_blocks = pd.read_csv(csv_fn)
        weekly_firm_mw = df_blocks['Total_Firm_Load_MW'].values
    except FileNotFoundError:
        print(f"❌ Error: Could not find CSV at {csv_fn}")
        sys.exit(1)

    # Build 25-yr Tier A profile (Firm Load)
    dynamic_tier_a_1yr = np.zeros(N_life_1yr)
    for w in range(52):
        dynamic_tier_a_1yr[w*168 : (w+1)*168] = weekly_firm_mw[w]
    dynamic_tier_a_1yr[52*168:] = weekly_firm_mw[-1]
    
    # Optional: If you scaled it in the iterative script, scale it here too
    dynamic_tier_a_25yr = 0.95 * np.tile(dynamic_tier_a_1yr, 25)
    
    # Build Tier B1 and B2 constant profiles
    tier_b1_25yr = np.full(N_life_total, TIER_B1_MW * 24.0)
    tier_b2_25yr = np.full(N_life_total, TIER_B2_MW * 168.0)
    
    # IT Capacity limit array (Forces CPLEX to respect the 55 MW hardware cap)
    fixed_capacity_profile = np.full(N_life_total, MAX_IT_CAPACITY_MW)

    # =========================================================================
    # 2. RUN SIMULATION
    # =========================================================================
    print(f"\n--- Running HPP Model for Visualization ({MAX_IT_CAPACITY_MW} MW Cap) ---")
    hpp_yearly = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=dynamic_tier_a_25yr, tier_b_profile=tier_b1_25yr, tier_b2_profile=tier_b2_25yr, 
        load_profile_ts=fixed_capacity_profile, battery_deg=False,
        run_mode='yearly' 
    )
    hpp_yearly.evaluate(*fixed_design)

    # =========================================================================
    # 3. EXTRACT AND RECONSTRUCT RESULTS
    # =========================================================================
    unserved_a   = hpp_yearly.prob.get_val('ems.Unserved_A')[:8760]
    hpp_t        = hpp_yearly.prob.get_val('ems.hpp_t')[:8760]
    b_t          = hpp_yearly.prob.get_val('ems.b_t')[:8760]
    wind_t       = hpp_yearly.prob.get_val('ems.wind_t_ext')[:8760]
    solar_t      = hpp_yearly.prob.get_val('ems.solar_t_ext')[:8760]
    shortfall_b1 = hpp_yearly.prob.get_val('ems.Shortfall_B')[:8760]
    shortfall_b2 = hpp_yearly.prob.get_val('ems.Shortfall_B2')[:8760]

    # Calculate actual Firm Load Served
    served_a = (0.95 * dynamic_tier_a_1yr) - unserved_a
    
    # --- SPLIT TIER A INTO BASELOAD AND CAPACITY BLOCKS ---
    served_a_base = np.minimum(served_a, ANNUAL_FIRM_BASE)
    served_a_blocks = np.maximum(served_a - ANNUAL_FIRM_BASE, 0.0)
    
    # Calculate Total Batch Power available for B1 and B2
    served_total_b = np.maximum(hpp_t - served_a, 0.0)

    # --- Heuristic Allocation & Queue Tracking ---
    # Because VoLL_Tier_B > VoLL_Tier_B2, CPLEX mathematically prioritizes B1.
    served_b1 = np.zeros(8760)
    served_b2 = np.zeros(8760)
    queue_b1  = np.zeros(8760)
    queue_b2  = np.zeros(8760)
    
    q1_len = 0.0
    q2_len = 0.0
    
    for i in range(8760):
        # 1. New jobs arrive
        q1_len += TIER_B1_MW
        q2_len += TIER_B2_MW
        
        power_avail = served_total_b[i]
        
        # 2. Serve B1 (Highest Priority)
        serve_1 = min(power_avail, q1_len)
        served_b1[i] = serve_1
        q1_len -= serve_1
        power_avail -= serve_1
        
        # 3. Serve B2 (Remaining Power)
        serve_2 = min(power_avail, q2_len)
        served_b2[i] = serve_2
        q2_len -= serve_2
        
        # 4. Record Queues
        queue_b1[i] = q1_len
        queue_b2[i] = q2_len

    bess_discharge = np.maximum(b_t, 0.0)
    bess_charge = np.minimum(b_t, 0.0)
    available_gen = wind_t + solar_t

    # =========================================================================
    # 4. PLOTTING WEEKS 7, 8, AND 9
    # =========================================================================
    start_hr = 6 * 168  # Hour 1008
    end_hr   = 9 * 168  # Hour 1512
    
    time_arr = np.arange(start_hr, end_hr)
    hours_plot = np.arange(len(time_arr))

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.subplots_adjust(hspace=0.08)
    
    # -------------------------------------------------------------------------
    # SUBPLOT 1: POWER DISPATCH STACK
    # -------------------------------------------------------------------------
    ax1.plot(hours_plot, available_gen[start_hr:end_hr], 'k--', linewidth=1.5, alpha=0.7, label='Available Gen (Wind+Solar)')
    
    # Stacked Area: Baseload -> Blocks -> Tier B1 -> Tier B2
    ax1.fill_between(hours_plot, 0, served_a_base[start_hr:end_hr], 
                     color='#00549F', alpha=0.9, label='Annual Firm Load (7 MW)')
                     
    ax1.fill_between(hours_plot, served_a_base[start_hr:end_hr], 
                     served_a_base[start_hr:end_hr] + served_a_blocks[start_hr:end_hr], 
                     color='#73A4FF', alpha=0.9, label='Weekly Capacity Blocks (Firm)')
    
    y_stack = served_a_base[start_hr:end_hr] + served_a_blocks[start_hr:end_hr]
    
    ax1.fill_between(hours_plot, y_stack, y_stack + served_b1[start_hr:end_hr], 
                     color='#E07A5F', alpha=0.9, label='Tier B1 (Daily)')
                     
    y_stack += served_b1[start_hr:end_hr]
    
    ax1.fill_between(hours_plot, y_stack, y_stack + served_b2[start_hr:end_hr], 
                     color='#8B5A2B', alpha=0.8, label='Tier B2 (Weekly)')
    
    # Battery Dynamics (Plotted behind/at baseline to show power injection)
    ax1.fill_between(hours_plot, 0, bess_discharge[start_hr:end_hr], color='#B82E12', alpha=0.9, label='BESS Discharge')
    ax1.fill_between(hours_plot, bess_charge[start_hr:end_hr], 0, color='#2CA02C', alpha=0.8, label='BESS Charge')
    
    ax1.axhline(MAX_IT_CAPACITY_MW, color='darkred', linestyle=':', linewidth=2, label=f'IT Capacity Limit ({MAX_IT_CAPACITY_MW} MW)')
    ax1.axhline(0, color='black', linewidth=1)
    
    ax1.set_ylabel('Power (MW)', fontweight='bold')
    ax1.set_title(f'Workload stacking with capacity blocks (week 7-9 highlight)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', fontsize=9, ncol=2)

    # -------------------------------------------------------------------------
    # SUBPLOT 2: QUEUE DYNAMICS (Stacked Backlogs)
    # -------------------------------------------------------------------------
    # Stack B1 and B2 backlogs visually
    ax3.fill_between(hours_plot, 0, queue_b1[start_hr:end_hr], 
                     color='#E07A5F', alpha=0.7, label='Tier B1 Backlog (MWh)')
    
    ax3.fill_between(hours_plot, queue_b1[start_hr:end_hr], queue_b1[start_hr:end_hr] + queue_b2[start_hr:end_hr], 
                     color='#C5B0D5', alpha=0.7, label='Tier B2 Backlog (MWh)')
    
    # Violations (If any exist, CPLEX calculates them directly and they will spike up here)
    ax3.bar(hours_plot, shortfall_b1[start_hr:end_hr], width=1.0, color='#D62728', alpha=0.9, label='B1 SLA Violations (Past 24h)')
    ax3.bar(hours_plot, shortfall_b2[start_hr:end_hr], width=1.0, color='darkred', alpha=0.9, label='B2 SLA Violations (Past 168h)')
    
    ax3.set_ylabel('Backlog (MWh)', fontweight='bold')
    ax3.set_xlabel('Hours (Relative to start of Week 7)', fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='upper right', fontsize=9)
    
    # Highlight the different weeks on the X-axis for clarity
    for w in [0, 168, 336, 504]:
        if w < len(hours_plot):
            ax1.axvline(w, color='gray', linestyle='-', alpha=0.3)
            ax3.axvline(w, color='gray', linestyle='-', alpha=0.3)
            
            if w < 504:
                week_num = 7 + int(w/168)
                ax3.text(w + 84, ax3.get_ylim()[1]*0.9, f'Week {week_num}', ha='center', va='top', 
                         fontsize=10, fontweight='bold', color='gray', alpha=0.7)

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'Thesis_B1_B2_Queue_Dynamics_Weeks7to9.png')
    plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {plot_fn}")
    plt.show()

if __name__ == "__main__":
    run_visualization()