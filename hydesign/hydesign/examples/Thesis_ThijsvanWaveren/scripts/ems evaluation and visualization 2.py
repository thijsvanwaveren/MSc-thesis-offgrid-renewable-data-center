# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:26:52 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.1.1 - Step-by-Step EMS Dispatch Visualization (Final & Unified)
Recalibrated for LaTeX PDF Export (7.5-inch width).
Fonts, line weights, and spacing have been adjusted for perfect document proportions.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- HYDESIGN IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model

# =============================================================================
# 1. INPUTS & EDITORIAL STYLING
# =============================================================================
MAX_IT_MW = 20.0
TIER_A_MW = 5.0
TIER_B1_MW = 10.0   
START_HOUR = 450  
WINDOW_HOURS = 72

SAVE_PLOTS = True 

# --- RECALIBRATED TYPOGRAPHY FOR 7.5" WIDTH ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9          # Base font size dropped for smaller canvas
plt.rcParams['axes.labelsize'] = 9     # Axis titles
plt.rcParams['xtick.labelsize'] = 8    # Tick numbers
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8    # Legends scaled down

# Unified Editorial Color Palette
C_A         = '#1a365d'  
C_B1        = '#2b6cb0'  
C_CATCHUP   = '#27ae60'  
C_GEN       = '#2c3e50'  
C_WIND      = '#4da6ff'  
C_SOLAR     = '#ffd166'  
C_SOC       = '#8e44ad'  
C_DISCHARGE = '#e74c3c'  
C_CHARGE    = '#00cc66'  
C_DEF       = '#e67e22'  

time_index = pd.date_range(start="2026-01-01 00:00", periods=8760, freq='h')[START_HOUR:START_HOUR+WINDOW_HOURS]

# =============================================================================
# 2. SIMULATION SETUP & RAW DATA EXTRACTION (Truncated for brevity, kept exactly same)
# =============================================================================
def configure_parameters(thesis_dir):
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    sim_pars['G_MW'] = 0
    sim_pars['battery_charge_efficiency'] = float(np.sqrt(0.86))
    sim_pars['Reward_C2'] = -0.5 
    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_clean_logic.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    return temp_fn

N_life = 25 * 8760
fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10]
site_name = 'Denmark_good_solar'
examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
ex_site = examples_sites.loc[examples_sites.name == site_name]
weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
sim_pars_fn = configure_parameters(thesis_dir)

t_a_ts = np.full(N_life, TIER_A_MW)
t_b1_ts = np.full(N_life, TIER_B1_MW * 24.0) 
t_b2_ts = np.full(N_life, 0.0) 
total_load = np.full(N_life, MAX_IT_MW)

hpp = hpp_model(
    latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
    num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
    tier_a_profile=t_a_ts, tier_b_profile=t_b1_ts, tier_b2_profile=t_b2_ts, load_profile_ts=total_load, battery_deg=False
)
hpp.evaluate(*fixed_design)

idx = slice(START_HOUR, START_HOUR + WINDOW_HOURS)
wind = hpp.prob.get_val('ems.wind_t_ext')[idx]
solar = hpp.prob.get_val('ems.solar_t_ext')[idx]
generation = wind + solar
soc = hpp.prob.get_val('ems.b_E_SOC_t')[idx] 
served_a = hpp.prob.get_val('ems.Served_A')[idx]

# =============================================================================
# 3. DETERMINISTIC QUEUE ENGINE
# =============================================================================
def simulate_deterministic_queue(avail_gen, base_load, max_facility_mw, prior_base_load=0):
    logical, catchup, deferred, queue = np.zeros(WINDOW_HOURS), np.zeros(WINDOW_HOURS), np.zeros(WINDOW_HOURS), np.zeros(WINDOW_HOURS)
    current_q = 0.0
    hw_headroom = max_facility_mw - (prior_base_load + base_load)
    
    for t in range(WINDOW_HOURS):
        avail = avail_gen[t]
        if avail < base_load:
            logical[t], deferred[t], catchup[t] = avail, base_load - avail, 0.0
            current_q += deferred[t]
        else:
            logical[t], deferred[t] = base_load, 0.0
            actual_catchup = min(min(avail - base_load, hw_headroom), current_q)
            catchup[t] = actual_catchup
            current_q -= actual_catchup
        queue[t] = current_q
    return logical, catchup, deferred, queue

log_b3, catch_b3, def_b3, q_b3 = simulate_deterministic_queue(generation, TIER_B1_MW, MAX_IT_MW, 0)
avail_s4 = np.maximum(0, generation - served_a)
log_b4, catch_b4, def_b4, q_b4 = simulate_deterministic_queue(avail_s4, TIER_B1_MW, MAX_IT_MW, TIER_A_MW)

# =============================================================================
# 4. STANDARD FORMATTING
# =============================================================================
def format_axes(ax):
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='#b0b0b0', zorder=0)
    ax.grid(axis='x', visible=False)
    for spine in ['top', 'right', 'left']: ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0) # Thinner baseline
    ax.spines['bottom'].set_color('#2d3748')
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(time_index[0], time_index[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

# =============================================================================
# 5. PLOTTING FUNCTIONS
# =============================================================================
def plot_step_1_baseline():
    fig, ax = plt.subplots(figsize=(7.5, 3), facecolor='white')
    ax.stackplot(time_index, wind, solar, colors=[C_WIND, C_SOLAR], labels=['Wind Generation', 'Solar Generation'], alpha=0.8)
    ax.plot(time_index, generation, color=C_GEN, linewidth=1.2, linestyle='--', label='Total Generation')
    ax.set_ylabel("Power (MW)", fontweight='bold', color='#2c3e50')
    format_axes(ax)
    ax.legend(loc='upper right', frameon=False, ncol=3)
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig(os.path.join(current_dir, "EMSplot_generation.svg"), bbox_inches='tight')

def plot_step_2_firm_storage():
    # Increased hspace to 0.4 to give legend extreme breathing room
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})
    
    ax1.stackplot(time_index, served_a, colors=[C_A], labels=[f'Tier A ({int(TIER_A_MW)} MW)'], alpha=0.9)
    ax1.plot(time_index, generation, color=C_GEN, linewidth=1.2, linestyle='--', label='Total Generation')
    
    bat_discharge = np.maximum(0, served_a - generation)
    ax1.fill_between(time_index, generation, generation + bat_discharge, where=(bat_discharge > 0.05), 
                     facecolor=C_DISCHARGE, alpha=0.8, hatch='//', label='Battery Discharging')
                     
    soc_diff = np.append(np.diff(soc), 0)
    is_charging = ((soc_diff > 0.05) & (generation > served_a)) | np.insert(((soc_diff > 0.05) & (generation > served_a))[:-1], 0, False)
    ax1.fill_between(time_index, served_a, generation, where=is_charging, facecolor=C_CHARGE, alpha=0.5, hatch='\\\\', label='Battery Charging')
    
    ax1.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    ax1.set_ylim(0, max(MAX_IT_MW, generation.max() * 1.05))
    format_axes(ax1)
    
    # Placed legend slightly higher to avoid crunching
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    
    ax2.plot(time_index, soc/2, color=C_SOC, linewidth=1.5, label='State of Charge (%)')
    ax2.fill_between(time_index, 0, soc/2, color=C_SOC, alpha=0.15)
    ax2.set_ylabel("SoC (%)", fontweight='bold', color='#2c3e50')
    ax2.set_ylim(0, 105)
    format_axes(ax2)
    ax2.legend(loc='upper left', frameon=False)
    
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig(os.path.join(current_dir, "EMSplot_storage.svg"), bbox_inches='tight')

def plot_step_3_flexible_queue():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})
    
    ax1.plot(time_index, generation, color=C_GEN, linewidth=1.2, linestyle='--', zorder=6, label='Total Generation')
    ax1.stackplot(time_index, log_b3, catch_b3, colors=[C_B1, C_CATCHUP], labels=[f'B1 Base ({int(TIER_B1_MW)} MW)', 'Catch-up'], alpha=0.85, zorder=3)
    
    ax1.axhline(TIER_B1_MW, color='#333333', linestyle=':', linewidth=1.0, zorder=5, label='Target Load')
    ax1.axhline(MAX_IT_MW, color='#c0392b', linestyle='-.', linewidth=1.0, zorder=5, label='Hardware Limit')
    
    ax1.fill_between(time_index, log_b3, TIER_B1_MW, where=(def_b3 > 0), facecolor=C_DEF, alpha=0.4, edgecolor=C_DEF, hatch='///', zorder=4, label='Deferred')
    
    ax1.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    ax1.set_ylim(0, max(MAX_IT_MW + 5, generation.max() * 1.05)) 
    format_axes(ax1)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
    
    ax2.plot(time_index, q_b3, color=C_B1, linewidth=1.5, zorder=5, label='Queue Volume')
    ax2.fill_between(time_index, 0, q_b3, color=C_B1, alpha=0.15, zorder=3)
    ax2.set_ylabel("Queue\n(MWh)", fontweight='bold', color='#2c3e50')
    ax2.set_ylim(0, max(10, q_b3.max() * 1.15)) 
    format_axes(ax2)
    ax2.legend(loc='upper left', frameon=False)
    
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig(os.path.join(current_dir, "EMSplot_b1.svg"), bbox_inches='tight')

def plot_step_4_full_hierarchy():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.4})
    
    target_total = TIER_A_MW + TIER_B1_MW
    actual_base = served_a + log_b4
    
    ax1.stackplot(time_index, served_a, log_b4, catch_b4, colors=[C_A, C_B1, C_CATCHUP], labels=['Tier A', 'B1 Base', 'B1 Catch-up'], alpha=0.9)
    ax1.plot(time_index, generation, color=C_GEN, linewidth=1.2, linestyle='--', label='Generation')
    ax1.axhline(target_total, color='#333333', linestyle=':', linewidth=1.0, label='Target Demand')
    ax1.axhline(MAX_IT_MW, color='#c0392b', linestyle='-.', linewidth=1.0, alpha=0.8, label='Hardware Limit')
    
    ax1.fill_between(time_index, actual_base, target_total, where=(def_b4 > 0), facecolor=C_DEF, alpha=0.4, edgecolor=C_DEF, hatch='///', label='Deferred B1')
    
    bat_discharge = np.maximum(0, served_a - generation)
    ax1.fill_between(time_index, generation, generation + bat_discharge, where=(bat_discharge > 0.05), facecolor=C_DISCHARGE, alpha=0.8, hatch='//', label='Battery Discharge')
    
    ax1.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    ax1.set_ylim(0, max(MAX_IT_MW + 5, generation.max() * 1.05))
    format_axes(ax1)
    
    # 4 columns for legend so it spans nicely across the top
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False, columnspacing=1.0)
    
    ax2.plot(time_index, soc/2, color=C_SOC, linewidth=1.5, label='State of Charge (%)')
    ax2.fill_between(time_index, 0, soc/2, color=C_SOC, alpha=0.15)
    ax2.set_ylabel("SoC (%)", fontweight='bold', color='#2c3e50')
    ax2.set_ylim(0, 105)
    format_axes(ax2)
    ax2.legend(loc='upper left', frameon=False)
    
    ax3.plot(time_index, q_b4, color=C_B1, linewidth=1.5, label='Tier B1 Queue')
    ax3.fill_between(time_index, 0, q_b4, color=C_B1, alpha=0.15)
    ax3.set_ylabel("Queue\n(MWh)", fontweight='bold', color='#2c3e50')
    ax3.set_ylim(0, max(10, q_b4.max() * 1.15))
    format_axes(ax3)
    ax3.legend(loc='upper left', frameon=False)
    
    fig.align_ylabels([ax1, ax2, ax3])
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig(os.path.join(current_dir, "EMSplot_total.svg"), bbox_inches='tight')

if __name__ == "__main__":
    plot_step_1_baseline()
    plot_step_2_firm_storage()
    plot_step_3_flexible_queue()
    plot_step_4_full_hierarchy()