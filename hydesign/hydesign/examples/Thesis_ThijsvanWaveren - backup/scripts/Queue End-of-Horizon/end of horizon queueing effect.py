# -*- coding: utf-8 -*-
"""
Visualizes the queueing end-of-horizon effect for highly flexible workloads.

Simulates a hybrid power plant paired with a dedicated Tier B2 (weekly flexible) 
data center. Extracts the final 500 hours of the operational year to demonstrate 
how the Energy Management System (EMS) exploits the finite simulation horizon 
by intentionally accumulating unserved workloads up to the Service Level Agreement 
(SLA) boundary limit in the final weeks of December.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# SETUP & PATHS
# =============================================================================
scripts_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(scripts_dir, '..', '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..', '..'))
vis_dir = os.path.join(thesis_dir, "Figures")

os.makedirs(vis_dir, exist_ok=True)
sys.path.append(root_dir)

from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model

# =============================================================================
# INPUTS & PLOT CONFIGURATION
# =============================================================================
FACILITY_MW = 16.0
TIER_B2_MW = 16.0   
PLOT_LAST_N_HOURS = 500  

SAVE_PLOTS = True 

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9          
plt.rcParams['axes.labelsize'] = 9      
plt.rcParams['xtick.labelsize'] = 8     
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8    

C_B2    = '#8e44ad'  
C_GEN   = '#2c3e50'  
C_DEF2  = '#c0392b'  

# Generate the full year to ensure the final 500 hours correctly map to December
full_time_index = pd.date_range(start="2026-01-01 00:00", periods=8760, freq='h')

def evaluate_16mw_anomaly():
    """Configures and runs the CPLEX evaluation for the anomaly scenario."""
    os.environ['REWARD_C2'] = '1000.0' 
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f: 
        sim_pars = yaml.safe_load(f)
        
    sim_pars['G_MW'] = 0
    sim_pars['battery_charge_efficiency'] = float(np.sqrt(0.86))
    
    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_anomaly.yml')
    with open(temp_fn, 'w') as f: 
        yaml.dump(sim_pars, f)

    N_life = 25 * 8760
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10]
    site_name = 'Denmark_good_solar'
    
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])

    t_a_ts = np.zeros(N_life)
    t_b1_ts = np.zeros(N_life)
    t_b2_ts = np.full(N_life, TIER_B2_MW * 168.0)  
    total_load = np.full(N_life, FACILITY_MW)

    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=scripts_dir, input_ts_fn=weather_fn, sim_pars_fn=temp_fn,
        tier_a_profile=t_a_ts, tier_b_profile=t_b1_ts, tier_b2_profile=t_b2_ts, load_profile_ts=total_load, battery_deg=False
    )
    hpp.evaluate(*fixed_design)

    idx = slice(0, 8760)
    return {
        'gen': hpp.prob.get_val('ems.wind_t_ext')[idx] + hpp.prob.get_val('ems.solar_t_ext')[idx],
        's_b2': hpp.prob.get_val('ems.Served_B2')[idx]
    }

res = evaluate_16mw_anomaly()

# =============================================================================
# QUEUE TRACKING & DATA SLICING
# =============================================================================
def track_raw_queue(served, target_mw):
    """Calculates the chronological queue accumulation over the year."""
    q = np.zeros(8760)
    for t in range(1, 8760): 
        q[t] = max(0, q[t-1] + target_mw - served[t])
    return q

q_b2_full = track_raw_queue(res['s_b2'], TIER_B2_MW)
SLA_LIMIT_MWH = TIER_B2_MW * 168.0

plot_idx = slice(8760 - PLOT_LAST_N_HOURS, 8760)
time_plot = full_time_index[plot_idx]
gen_plot = res['gen'][plot_idx]
s_b2_plot = res['s_b2'][plot_idx]
q_b2_plot = q_b2_full[plot_idx]
def_b2_plot = np.maximum(0, TIER_B2_MW - s_b2_plot)

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================
def plot_end_of_horizon_effect():
    """Generates the dual-panel time-series visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True, gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.4})

    # --- TOP PLOT: POWER DISPATCH ---
    ax1.plot(time_plot, gen_plot, color=C_GEN, linewidth=1.2, linestyle='--', label='Generation', zorder=5)
    ax1.fill_between(time_plot, 0, s_b2_plot, color=C_B2, alpha=0.8, label=f'Tier B2 ({int(TIER_B2_MW)} MW)', zorder=3)
    ax1.fill_between(time_plot, s_b2_plot, TIER_B2_MW, where=(def_b2_plot > 0.05), 
                     facecolor=C_DEF2, alpha=0.3, edgecolor=C_DEF2, hatch='\\\\\\', label='Deferred B2 Workload', zorder=4)
    
    ax1.axhline(FACILITY_MW, color='#c0392b', linestyle='-.', linewidth=1.2, label='Hardware Limit')

    ax1.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    ax1.set_ylim(0, max(FACILITY_MW + 5, gen_plot.max() * 1.05))
    
    for ax in [ax1, ax2]:
        ax.grid(axis='y', linestyle='-', alpha=0.3, color='#b0b0b0')
        ax.grid(axis='x', visible=False)
        for spine in ['top', 'right', 'left']: ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['bottom'].set_color('#2d3748')
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(time_plot[0], time_plot[-1])

    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)

    # --- BOTTOM PLOT: QUEUE ACCUMULATION ---
    ax2.plot(time_plot, q_b2_plot, color=C_B2, linewidth=2.0, label='Tier B2 Queue Volume')
    ax2.fill_between(time_plot, 0, q_b2_plot, color=C_B2, alpha=0.15)
    ax2.axhline(SLA_LIMIT_MWH, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'SLA limit ({int(SLA_LIMIT_MWH)} MWh)')

    ax2.set_ylabel("Queue\n(MWh)", fontweight='bold', color='#2c3e50')
    ax2.set_ylim(0, SLA_LIMIT_MWH * 1.15)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, PLOT_LAST_N_HOURS//120)))

    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)

    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    
    file_name = "EMSplot_EndOfHorizon.svg"
    if SAVE_PLOTS: 
        save_path = os.path.join(vis_dir, file_name)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        
    plt.show()
    
if __name__ == "__main__":
    plot_end_of_horizon_effect()
