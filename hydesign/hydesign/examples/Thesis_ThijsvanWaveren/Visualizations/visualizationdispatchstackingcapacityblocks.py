# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:21:34 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Visualization of Workload Stacking & Queue Dynamics for Tier B2
Specifically zooming in on Weeks 7, 8, and 9 to demonstrate Valley Filling.
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
    # Fixed Data
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    MAX_IT_CAPACITY_MW = 55.0  # Strict hardware ceiling
    TIER_B2_MW = 19.0          # Target weekly capacity blocks
    N_life_1yr = 8760
    N_life_total = 25 * 8760
    
    b_E_capacity = 25.0 * 8.0  # b_P * b_E_h = 200 MWh
    
    # Environment Setup
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    # Disable C2 Opportunistic Load entirely
    os.environ['REWARD_C2'] = '1.0'  

    # =========================================================================
    # 1. LOAD CSV AND BUILD PROFILES
    # =========================================================================
    csv_fn = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\Results\Weekly\annual simulation weekly result\Weekly_Capacity_Blocks_yearlysimulation.csv"
    try:
        df_blocks = pd.read_csv(csv_fn)
        weekly_firm_mw = df_blocks['Total_Firm_Load_MW'].values
    except FileNotFoundError:
        print(f"❌ Error: Could not find CSV at {csv_fn}")
        sys.exit(1)

    # Build 25-yr Tier A profile (Raw Firm Load, No Scaling)
    dynamic_tier_a_1yr = np.zeros(N_life_1yr)
    for w in range(52):
        dynamic_tier_a_1yr[w*168 : (w+1)*168] = weekly_firm_mw[w]
    dynamic_tier_a_1yr[52*168:] = weekly_firm_mw[-1]
    
    dynamic_tier_a_25yr = np.tile(dynamic_tier_a_1yr, 25)
    t_zero_25yr = np.zeros(N_life_total)

    # Build Tier B2 profile (10 MW constant arrival)
    tier_b2_25yr = np.full(N_life_total, TIER_B2_MW * 168.0)
    
    # IT Capacity limit array
    fixed_capacity_profile = np.full(N_life_total, MAX_IT_CAPACITY_MW)

    # =========================================================================
    # 2. RUN SIMULATION
    # =========================================================================
    print(f"\n--- Running HPP Model for Visualization ---")
    hpp_yearly = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=dynamic_tier_a_25yr, tier_b_profile=t_zero_25yr, tier_b2_profile=tier_b2_25yr, 
        load_profile_ts=fixed_capacity_profile, battery_deg=False,
        run_mode='yearly' 
    )
    hpp_yearly.evaluate(*fixed_design)

    # =========================================================================
    # 3. EXTRACT RESULTS (First 8760 Hours)
    # =========================================================================
    unserved_a = hpp_yearly.prob.get_val('ems.Unserved_A')[:8760]
    hpp_t      = hpp_yearly.prob.get_val('ems.hpp_t')[:8760]
    b_t        = hpp_yearly.prob.get_val('ems.b_t')[:8760]       # Battery power (positive=discharge)
    soc_t      = hpp_yearly.prob.get_val('ems.b_E_SOC_t')[:8760] # Battery SOC
    wind_t     = hpp_yearly.prob.get_val('ems.wind_t_ext')[:8760]
    solar_t    = hpp_yearly.prob.get_val('ems.solar_t_ext')[:8760]
    shortfall_b2 = hpp_yearly.prob.get_val('ems.Shortfall_B2')[:8760] # Missed deadlines

    # Calculate exactly what was served
    served_a = dynamic_tier_a_1yr - unserved_a
    served_b2 = hpp_t - served_a
    served_b2 = np.maximum(served_b2, 0.0) # Clean up floating point dust

    bess_discharge = np.maximum(b_t, 0.0)
    bess_charge = np.minimum(b_t, 0.0)
    available_gen = wind_t + solar_t

    # Reconstruct the Tier B2 Queue dynamically
    queue_b2 = np.zeros(8760)
    q_len = 0.0
    for i in range(8760):
        q_len = q_len + TIER_B2_MW - served_b2[i]
        q_len = max(0.0, q_len)
        queue_b2[i] = q_len

    # =========================================================================
    # 4. PLOTTING WEEKS 7, 8, AND 9
    # =========================================================================
    # Week indices: Week 1 is 0-167. Weeks 7, 8, 9 range from 6*168 to 9*168
    start_hr = 6 * 168  # Hour 1008
    end_hr   = 9 * 168  # Hour 1512
    
    time_arr = np.arange(start_hr, end_hr)
    hours_plot = np.arange(len(time_arr)) # X-axis relative to start of plot

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.subplots_adjust(hspace=0.08)
    
    # -------------------------------------------------------------------------
    # SUBPLOT 1: POWER DISPATCH STACK
    # -------------------------------------------------------------------------
    # Available Gen
    ax1.plot(hours_plot, available_gen[start_hr:end_hr], 'k--', linewidth=1.5, alpha=0.7, label='Available Gen (Wind+Solar)')
    
    # Firm Tier A & Tier B2 Dispatch (Stacked Area)
    ax1.fill_between(hours_plot, 0, served_a[start_hr:end_hr], color='#00549F', alpha=0.8, label='Tier A Dispatch (Firm)')
    ax1.fill_between(hours_plot, served_a[start_hr:end_hr], served_a[start_hr:end_hr] + served_b2[start_hr:end_hr], 
                     color='#8B5A2B', alpha=0.8, label='Tier B2 Dispatch (Weekly Batch)')
    
    # Battery Dynamics
    ax1.fill_between(hours_plot, 0, bess_discharge[start_hr:end_hr], color='#FF7F0E', alpha=0.9, label='BESS Discharge')
    ax1.fill_between(hours_plot, bess_charge[start_hr:end_hr], 0, color='#2CA02C', alpha=0.8, label='BESS Charge')
    
    # Thresholds
    ax1.axhline(MAX_IT_CAPACITY_MW, color='darkred', linestyle=':', linewidth=2, label=f'IT Capacity Limit ({MAX_IT_CAPACITY_MW} MW)')
    ax1.axhline(TIER_B2_MW, color='green', linestyle=':', linewidth=2, label=f'Tier B2 Arrival Target ({TIER_B2_MW} MW)')
    ax1.axhline(0, color='black', linewidth=1)
    
    ax1.set_ylabel('Power (MW)', fontweight='bold')
    ax1.set_title(f'Tier B2 Queue Dynamics - Valley Filling in Weeks 7-9 ({MAX_IT_CAPACITY_MW} MW IT Cap)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', fontsize=9, ncol=2)

    # -------------------------------------------------------------------------
    # SUBPLOT 2: BATTERY SOC
    # -------------------------------------------------------------------------
    soc_percent = (soc_t[start_hr:end_hr] / b_E_capacity) * 100.0
    ax2.plot(hours_plot, soc_percent, 'b-', linewidth=2, label='Battery SOC')
    ax2.set_ylabel('SOC (%)', fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', fontsize=9)

    # -------------------------------------------------------------------------
    # SUBPLOT 3: QUEUE DYNAMICS
    # -------------------------------------------------------------------------
    ax3.fill_between(hours_plot, 0, queue_b2[start_hr:end_hr], color='#C5B0D5', alpha=0.7, label='Tier B2 Queue Backlog (MWh)')
    ax3.bar(hours_plot, shortfall_b2[start_hr:end_hr], width=1.0, color='#D62728', alpha=0.9, label='SLA Violations (Jobs past 168h deadline)')
    
    ax3.set_ylabel('Backlog (MWh)', fontweight='bold')
    ax3.set_xlabel('Hours (Relative to start of Week 7)', fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='upper right', fontsize=9)
    
    # Highlight the different weeks on the X-axis for clarity
    for w in [0, 168, 336, 504]:
        if w < len(hours_plot):
            ax1.axvline(w, color='gray', linestyle='-', alpha=0.3)
            ax2.axvline(w, color='gray', linestyle='-', alpha=0.3)
            ax3.axvline(w, color='gray', linestyle='-', alpha=0.3)
            
            # Add text labels for the weeks
            if w < 504:
                week_num = 7 + int(w/168)
                ax3.text(w + 84, ax3.get_ylim()[1]*0.9, f'Week {week_num}', ha='center', va='top', 
                         fontsize=10, fontweight='bold', color='gray', alpha=0.7)

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'Thesis_B2_Queue_Dynamics_Weeks7to9.png')
    plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {plot_fn}")
    plt.show()

if __name__ == "__main__":
    run_visualization()