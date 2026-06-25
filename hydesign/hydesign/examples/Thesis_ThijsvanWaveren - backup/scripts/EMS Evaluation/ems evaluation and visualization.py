# -*- coding: utf-8 -*-
"""
Section 3.1 & 3.2 - Complete EMS Dispatch Visualization
Plots 100% raw CPLEX outputs. 
Tier C is completely disabled (via os.environ) for Steps 1-4 and 6 to isolate core mechanics,
and enabled for Step 5 to expose advanced co-optimization.
Includes accurate physical battery charging bounds for ALL plots.
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
thesis_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model

# =============================================================================
# 1. INPUTS & EDITORIAL STYLING ("Academic Consulting" Style)
# =============================================================================
MAX_IT_MW = 20.0
START_HOUR = 450  
WINDOW_HOURS = 72

SAVE_PLOTS = True 

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 9          
plt.rcParams['axes.labelsize'] = 9      
plt.rcParams['xtick.labelsize'] = 8    
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8    

# --- LOGICAL COLOR PALETTE ---
C_A         = '#1a365d'  # Tier A (Dark Blue)

C_B1        = '#2b6cb0'  # Tier B1 Base (Medium Blue)
C_B1_CATCH  = '#7cb3f1'  # Tier B1 Catch-up (Lighter Blue)

C_B2        = '#8e44ad'  # Tier B2 Base (Purple)
C_B2_CATCH  = '#c39bd3'  # Tier B2 Catch-up (Lighter Purple)

C_C         = '#00cec9'  # Tier C (Cyan)

C_GEN       = '#2c3e50'  # Total Gen (Slate)
C_WIND      = '#4da6ff'  
C_SOLAR     = '#ffd166'  
C_SOC       = '#7f8c8d'  # SoC Line (Gray)
C_DISCHARGE = '#e74c3c'  # Battery Discharge
C_CHARGE    = '#00cc66'  # Battery Charge
C_LIMIT     = '#c0392b'  # Hardware Limit

time_index = pd.date_range(start="2026-01-01 00:00", periods=8760, freq='h')[START_HOUR:START_HOUR+WINDOW_HOURS]

# =============================================================================
# 2. MASTER CPLEX EVALUATION ENGINE
# =============================================================================
def evaluate_cplex_scenario(tier_a, tier_b1, tier_b2, enable_tier_c, file_suffix):
    """Runs CPLEX. Hard-forces Tier C state using the os.environ override."""
    
    if enable_tier_c:
        os.environ['REWARD_C2'] = '-0.5'   
    else:
        os.environ['REWARD_C2'] = '1.0' 

    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
        
    sim_pars['G_MW'] = 0
    charge_eff = float(np.sqrt(0.86))
    sim_pars['battery_charge_efficiency'] = charge_eff
    
    temp_fn = os.path.join(thesis_dir, 'inputs', f'hpp_pars_{file_suffix}.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)

    N_life = 25 * 8760
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10]
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])

    t_a_ts = np.full(N_life, tier_a)
    t_b1_ts = np.full(N_life, tier_b1 * 24.0) 
    t_b2_ts = np.full(N_life, tier_b2 * 168.0) 
    total_load = np.full(N_life, MAX_IT_MW)

    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=temp_fn,
        tier_a_profile=t_a_ts, tier_b_profile=t_b1_ts, tier_b2_profile=t_b2_ts, load_profile_ts=total_load, battery_deg=False
    )
    hpp.evaluate(*fixed_design)

    idx = slice(START_HOUR, START_HOUR + WINDOW_HOURS)
    return {
        'wind': hpp.prob.get_val('ems.wind_t_ext')[idx],
        'solar': hpp.prob.get_val('ems.solar_t_ext')[idx],
        'gen': hpp.prob.get_val('ems.wind_t_ext')[idx] + hpp.prob.get_val('ems.solar_t_ext')[idx],
        'soc': hpp.prob.get_val('ems.b_E_SOC_t')[idx],
        's_a': hpp.prob.get_val('ems.Served_A')[idx],
        's_b1': hpp.prob.get_val('ems.Served_B')[idx],
        's_b2': hpp.prob.get_val('ems.Served_B2')[idx],
        's_c': hpp.prob.get_val('ems.Served_C2')[idx] if enable_tier_c else np.zeros(WINDOW_HOURS),
        'charge_eff': charge_eff
    }

print("Running CPLEX Scenario: Tier A Only (Step 2)...")
res_a_only = evaluate_cplex_scenario(8.0, 0.0, 0.0, False, "a_only")

print("Running CPLEX Scenario: Base (Step 4)...")
res_base = evaluate_cplex_scenario(5.0, 10.0, 0, False, "base")

print("Running CPLEX Scenario: B1 Only (Step 3)...")
res_b1_only = evaluate_cplex_scenario(0.0, 10.0, 0, False, "b1")

print("Running CPLEX Scenario: Tier C Enabled (Step 5)...")
res_c = evaluate_cplex_scenario(5.0, 10.0, 0, True, "tier_c")

print("Running CPLEX Scenario: Dual Queues B1 & B2 (Step 6)...")
res_dual = evaluate_cplex_scenario(0.0, 8.0, 8.0, False, "dual")

print("Running CPLEX Scenario: Over-allocated Firm Load (Step 7)...")
res_firm_fail = evaluate_cplex_scenario(20.0, 0.0, 0.0, False, "firm_fail")
# =============================================================================
# 3. DATA PARSING & QUEUE TRACKING
# =============================================================================
def track_raw_queue(served, target_mw):
    q = np.zeros(WINDOW_HOURS)
    for t in range(1, WINDOW_HOURS): 
        q[t] = max(0, q[t-1] + target_mw - served[t])
    return q

def parse_base_catchup(served_array, target_mw):
    base = np.minimum(target_mw, served_array)
    catchup = np.maximum(0, served_array - target_mw)
    return base, catchup

# =============================================================================
# 4. STANDARD AXIS FORMATTING
# =============================================================================
def format_axes(ax):
    # Minimalist, clean grid suitable for consulting reports
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='#b0b0b0', zorder=0)
    ax.grid(axis='x', visible=False)
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0) 
    ax.spines['bottom'].set_color('#2d3748')
    ax.spines['left'].set_linewidth(1.0) 
    ax.spines['left'].set_color('#2d3748')
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(time_index[0], time_index[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

# =============================================================================
# 5. PLOTTING FUNCTIONS
# =============================================================================
def plot_step_1_baseline():
    fig, ax = plt.subplots(figsize=(8.5, 3.5), facecolor='white')
    ax.stackplot(time_index, res_base['wind'], res_base['solar'], colors=[C_WIND, C_SOLAR], labels=['Wind Power', 'Solar Power'], alpha=0.8, zorder=3)
    ax.plot(time_index, res_base['gen'], color=C_GEN, linewidth=1.5, linestyle='--', zorder=6, label='Total Power Output')
    ax.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    format_axes(ax)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncol=3)
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig(os.path.join(current_dir, "EMSplot_generation.svg"), bbox_inches='tight')

def plot_step_2_firm_storage():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.5), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})
    
    ax1.stackplot(time_index, res_a_only['s_a'], colors=[C_A], labels=['Tier A (8 MW)'], alpha=0.9, zorder=3)
    ax1.plot(time_index, res_a_only['gen'], color=C_GEN, linewidth=1.5, linestyle='--', zorder=6, label='Total Power Output')
    
    total_served = res_a_only['s_a']
    bat_discharge = np.maximum(0, total_served - res_a_only['gen'])
    ax1.fill_between(time_index, res_a_only['gen'], res_a_only['gen'] + bat_discharge, where=(bat_discharge > 0.05), 
                     facecolor=C_DISCHARGE, alpha=0.85, hatch='//', zorder=5, label='Battery Discharging')
                     
    charge_eff = res_a_only['charge_eff']
    actual_charge_mw = np.maximum(0, np.append(np.diff(res_a_only['soc']), 0)) / charge_eff
    ax1.fill_between(time_index, total_served, total_served + actual_charge_mw, where=(actual_charge_mw > 0.05), 
                     facecolor=C_CHARGE, alpha=0.6, hatch='\\\\', zorder=5, label='Battery Charging')
    
    ax1.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    ax1.set_ylim(0, max(MAX_IT_MW, res_a_only['gen'].max() * 1.05))
    format_axes(ax1)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    
    ax2.plot(time_index, res_a_only['soc']/2, color=C_SOC, linewidth=1.5, label='State of Charge (%)', zorder=5)
    ax2.fill_between(time_index, 0, res_a_only['soc']/2, color=C_SOC, alpha=0.15, zorder=3)
    ax2.set_ylabel("SoC (%)", fontweight='bold', color='#2c3e50')
    ax2.set_ylim(0, 105)
    format_axes(ax2)
    ax2.legend(loc='upper left', frameon=False)
    
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig(os.path.join(current_dir, "EMSplot_storage.svg"), bbox_inches='tight')

def plot_step_3_flexible_queue():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.5), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})
    
    b_base, b_catch = parse_base_catchup(res_b1_only['s_b1'], 10.0)
    def_b = np.maximum(0, 10.0 - b_base)
    q_b = track_raw_queue(res_b1_only['s_b1'], 10.0)

    ax1.plot(time_index, res_b1_only['gen'], color=C_GEN, linewidth=1.5, linestyle='--', zorder=6, label='Total Power Output')
    ax1.stackplot(time_index, b_base, b_catch, colors=[C_B1, C_B1_CATCH], labels=['Tier B1 Served (10 MW)', 'Tier B1 Catch-up'], alpha=0.9, zorder=3)
    
    ax1.axhline(10.0, color='#333333', linestyle=':', linewidth=1.0, zorder=5, label='Target Load')
    ax1.axhline(MAX_IT_MW, color=C_LIMIT, linestyle='-.', linewidth=1.0, zorder=5, label='Hardware Limit')
    
    # Deferred plotted identically to base color but strictly transparent
    ax1.fill_between(time_index, b_base, 10.0, where=(def_b > 0), facecolor=C_B1, alpha=0.35, edgecolor='none', zorder=4, label='Deferred Tier B1')
    
    total_served = res_b1_only['s_b1']
    bat_discharge = np.maximum(0, total_served - res_b1_only['gen'])
    ax1.fill_between(time_index, res_b1_only['gen'], res_b1_only['gen'] + bat_discharge, where=(bat_discharge > 0.05), 
                     facecolor=C_DISCHARGE, alpha=0.85, hatch='//', zorder=5) 
    
    charge_eff = res_b1_only['charge_eff']
    actual_charge_mw = np.maximum(0, np.append(np.diff(res_b1_only['soc']), 0)) / charge_eff
    ax1.fill_between(time_index, total_served, total_served + actual_charge_mw, where=(actual_charge_mw > 0.05), 
                     facecolor=C_CHARGE, alpha=0.6, hatch='\\\\', zorder=5)

    ax1.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    ax1.set_ylim(0, max(MAX_IT_MW + 5, res_b1_only['gen'].max() * 1.05)) 
    format_axes(ax1)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)
    
    ax2.plot(time_index, q_b, color=C_B1, linewidth=1.5, zorder=5, label='Tier B1 Queue')
    ax2.fill_between(time_index, 0, q_b, color=C_B1_CATCH, alpha=0.45, zorder=3)
    ax2.set_ylabel("Queue\n(MWh)", fontweight='bold', color='#2c3e50')
    ax2.set_ylim(0, max(10, q_b.max() * 1.15)) 
    format_axes(ax2)
    ax2.legend(loc='upper left', frameon=False)
    
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig(os.path.join(current_dir, "EMSplot_b1.svg"), bbox_inches='tight')

def plot_step_4_full_hierarchy():
    # FIXED: Removed the SoC chart. Now exactly 2 subplots (Power and Queue).
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.5), sharex=True, gridspec_kw={'height_ratios': [2.2, 1], 'hspace': 0.3})
    
    b_base, b_catch = parse_base_catchup(res_base['s_b1'], 10.0)
    def_b = np.maximum(0, 10.0 - b_base)
    q_b = track_raw_queue(res_base['s_b1'], 10.0)
    
    ax1.stackplot(time_index, res_base['s_a'], b_base, b_catch, colors=[C_A, C_B1, C_B1_CATCH], labels=['Tier A', 'Tier B1 Served', 'Tier B1 Catch-up'], alpha=0.9, zorder=3)
    ax1.plot(time_index, res_base['gen'], color=C_GEN, linewidth=1.5, linestyle='--', zorder=6, label='Total Power Output')
    #ax1.axhline(15.0, color='#333333', linestyle=':', linewidth=1.0, zorder=5, label='Target Demand')
    ax1.axhline(MAX_IT_MW, color=C_LIMIT, linestyle='-.', linewidth=1.0, alpha=0.8, zorder=5, label='Hardware Limit')
    
    ax1.fill_between(time_index, res_base['s_a'] + b_base, res_base['s_a'] + 10.0, where=(def_b > 0), facecolor=C_B1, alpha=0.35, edgecolor='none', zorder=4, label='Deferred Tier B1')
    
    total_served = res_base['s_a'] + res_base['s_b1']
    bat_discharge = np.maximum(0, total_served - res_base['gen'])
    ax1.fill_between(time_index, res_base['gen'], res_base['gen'] + bat_discharge, where=(bat_discharge > 0.05), facecolor=C_DISCHARGE, alpha=0.85, hatch='//', zorder=5, label='Battery Discharging')
    
    charge_eff = res_base['charge_eff']
    actual_charge_mw = np.maximum(0, np.append(np.diff(res_base['soc']), 0)) / charge_eff
    ax1.fill_between(time_index, total_served, total_served + actual_charge_mw, where=(actual_charge_mw > 0.05), 
                     facecolor=C_CHARGE, alpha=0.6, hatch='\\\\', zorder=5, label='Battery Charging')

    ax1.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    ax1.set_ylim(0, max(MAX_IT_MW + 5, res_base['gen'].max() * 1.05))
    format_axes(ax1)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False, columnspacing=1.0)
    
    ax2.plot(time_index, q_b, color=C_B1, linewidth=1.5, zorder=5, label='Tier B1 Queue')
    ax2.fill_between(time_index, 0, q_b, color=C_B1_CATCH, alpha=0.45, zorder=3)
    ax2.set_ylabel("Queue\n(MWh)", fontweight='bold', color='#2c3e50')
    ax2.set_ylim(0, max(10, q_b.max() * 1.15))
    format_axes(ax2)
    ax2.legend(loc='upper left', frameon=False)
    
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig(os.path.join(current_dir, "EMSplot_total.svg"), bbox_inches='tight')
    
def plot_step_5_b1battery():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.5), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})
    
    b_base, b_catch = parse_base_catchup(res_c['s_b1'], 10.0)
    def_b = np.maximum(0, 10.0 - b_base)
    q_b_c = track_raw_queue(res_c['s_b1'], 10.0)

    ax1.stackplot(time_index, res_c['s_a'], b_base, b_catch, res_c['s_c'], 
                  colors=[C_A, C_B1, C_B1_CATCH, C_C], labels=['Tier A', 'Tier B1 Served', 'Tier B1 Catch-up', 'Tier C Served'], alpha=0.9, zorder=3)
    ax1.plot(time_index, res_c['gen'], color=C_GEN, linewidth=1.5, linestyle='--', zorder=6, label='Total Power Output')
    ax1.axhline(MAX_IT_MW, color=C_LIMIT, linestyle='-.', linewidth=1.0, alpha=0.8, zorder=5, label='Hardware Limit')
    
    # Restored explicitly to show deferred area behind components
    ax1.fill_between(time_index, res_c['s_a'] + b_base, res_c['s_a'] + 10.0, where=(def_b > 0), facecolor=C_B1, alpha=0.35, edgecolor='none', zorder=4, label='Deferred Tier B1')

    total_served = res_c['s_a'] + res_c['s_b1'] + res_c['s_c']
    bat_discharge = np.maximum(0, total_served - res_c['gen'])
    ax1.fill_between(time_index, res_c['gen'], res_c['gen'] + bat_discharge, where=(bat_discharge > 0.05), 
                     facecolor=C_DISCHARGE, alpha=0.85, hatch='//', zorder=5, label='Battery Discharging')

    charge_eff = res_c['charge_eff']
    actual_charge_mw = np.maximum(0, np.append(np.diff(res_c['soc']), 0)) / charge_eff
    ax1.fill_between(time_index, total_served, total_served + actual_charge_mw, where=(actual_charge_mw > 0.05), 
                     facecolor=C_CHARGE, alpha=0.6, hatch='\\\\', zorder=5, label='Battery Charging')

    ax1.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    ax1.set_ylim(0, max(MAX_IT_MW + 5, res_c['gen'].max() * 1.05))
    format_axes(ax1)
    
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, columnspacing=1.0)
    
    ax2.plot(time_index, q_b_c, color=C_B1, linewidth=1.5, zorder=5, label='Tier B1 Queue')
    ax2.fill_between(time_index, 0, q_b_c, color=C_B1_CATCH, alpha=0.45, zorder=3)
    ax2.set_ylabel("Queue\n(MWh)", fontweight='bold', color='#2c3e50')
    ax2.set_ylim(0, max(10, q_b_c.max() * 1.15))
    format_axes(ax2)
    ax2.legend(loc='upper left', frameon=False)
    
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig(os.path.join(current_dir, "EMSplot_b1battery.svg"), bbox_inches='tight')

def plot_step_6_b1b2():
    """
    Step 6: Visualizes dual flexible queues (B1 and B2).
    Upgraded to a 2-panel layout, overlaying both queues on a single bottom axis 
    for direct comparative volume analysis.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True, 
                                   gridspec_kw={'height_ratios': [2.2, 1.2], 'hspace': 0.3})

    b1_b, b1_c = parse_base_catchup(res_dual['s_b1'], 8.0)
    b2_b, b2_c = parse_base_catchup(res_dual['s_b2'], 8.0)
    def_b1 = np.maximum(0, 8.0 - b1_b)
    def_b2 = np.maximum(0, 8.0 - b2_b)
    
    # -------------------------------------------------------------------------
    # TOP PANEL: Power Allocation
    # -------------------------------------------------------------------------
    ax1.stackplot(time_index, b1_b, b2_b, b1_c, b2_c, 
                  colors=[C_B1, C_B2, C_B1_CATCH, C_B2_CATCH], 
                  labels=['Tier B1 Served', 'Tier B2 Served', 'B1 Catch-up', 'B2 Catch-up'], 
                  alpha=0.9, zorder=3)
    
    ax1.plot(time_index, res_dual['gen'], color=C_GEN, linewidth=1.5, linestyle='--', zorder=6, label='Total Power Output')
    ax1.axhline(MAX_IT_MW, color=C_LIMIT, linestyle='-.', linewidth=1.2, alpha=0.9, zorder=5, label='Hardware Limit')

    # Deferred mapped exactly to respective tier colors, distinguishable purely by lowered alpha
    ax1.fill_between(time_index, b1_b, 8.0, where=(def_b1 > 0), facecolor=C_B1, alpha=0.35, edgecolor='none', zorder=4, label='Deferred B1')
    ax1.fill_between(time_index, 8.0 + b2_b, 16.0, where=(def_b2 > 0), facecolor=C_B2, alpha=0.35, edgecolor='none', zorder=4, label='Deferred B2')

    total_served = res_dual['s_b1'] + res_dual['s_b2']
    bat_discharge = np.maximum(0, total_served - res_dual['gen'])
    ax1.fill_between(time_index, res_dual['gen'], res_dual['gen'] + bat_discharge, where=(bat_discharge > 0.05), 
                     facecolor=C_DISCHARGE, alpha=0.85, hatch='//', zorder=5, label='Battery Discharging')

    charge_eff = res_dual['charge_eff']
    actual_charge_mw = np.maximum(0, np.append(np.diff(res_dual['soc']), 0)) / charge_eff
    ax1.fill_between(time_index, total_served, total_served + actual_charge_mw, where=(actual_charge_mw > 0.05), 
                     facecolor=C_CHARGE, alpha=0.6, hatch='\\\\', zorder=5, label='Battery Charging')

    ax1.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    ax1.set_ylim(0, max(MAX_IT_MW + 5, res_dual['gen'].max() * 1.05))
    format_axes(ax1)
    
    # Legend formatting for Top Panel
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False, columnspacing=1.5)

    # -------------------------------------------------------------------------
    # BOTTOM PANEL: Combined Queue Volumes
    # -------------------------------------------------------------------------
    q_b1_raw = track_raw_queue(res_dual['s_b1'], 8.0)
    q_b2_raw = track_raw_queue(res_dual['s_b2'], 8.0)

    # Plot B2 (Purple) first so it sits in the background (since it's usually larger)
    ax2.plot(time_index, q_b2_raw, color=C_B2, linewidth=1.5, zorder=4, label='Tier B2 Queue')
    ax2.fill_between(time_index, 0, q_b2_raw, color=C_B2_CATCH, alpha=0.40, zorder=3)
    
    # Plot B1 (Blue) on top of B2
    ax2.plot(time_index, q_b1_raw, color=C_B1, linewidth=1.5, zorder=6, label='Tier B1 Queue')
    ax2.fill_between(time_index, 0, q_b1_raw, color=C_B1_CATCH, alpha=0.60, zorder=5)

    ax2.set_ylabel("Queue Volume\n(MWh)", fontweight='bold', color='#2c3e50')
    
    # Dynamically scale Y-axis to whichever queue is largest (typically B2)
    max_queue = max(q_b1_raw.max(), q_b2_raw.max())
    ax2.set_ylim(0, max(10, max_queue * 1.15))
    format_axes(ax2)
    
    # Use a 2-column legend for the bottom panel so they sit neatly side-by-side
    ax2.legend(loc='upper left', frameon=False, ncol=2)
    
    # -------------------------------------------------------------------------
    # GLOBAL FORMATTING
    # -------------------------------------------------------------------------
    fig.align_ylabels([ax1, ax2])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Leave room for the top legend
    
    if SAVE_PLOTS: 
        plt.savefig(os.path.join(current_dir, "EMSplot_b1b2.svg"), bbox_inches='tight')
    
def plot_step_7_firm_fail():
    """
    Step 7: Visualizes a Firm Load (Tier A) blackout. 
    Target line removed. Target capacity updated to 20.0 MW to match CPLEX run.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, 
                                   gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.3})

    s_a = res_firm_fail['s_a']
    gen = res_firm_fail['gen']
    soc = res_firm_fail['soc']
    
    target_mw = 20.0 # Updated to match your CPLEX input
    
    # Distinct colors for clarity
    C_DROPPED = '#8b0000'    # Deep Crimson for Blackout
    C_DISCHARGE = '#e74c3c'  # Bright Red for Battery

    # -------------------------------------------------------------------------
    # TOP PANEL: Power Allocation & Blackout
    # -------------------------------------------------------------------------
    ax1.plot(time_index, gen, color=C_GEN, linewidth=1.5, linestyle='--', zorder=6, label='Total Power Output')
    ax1.fill_between(time_index, 0, s_a, facecolor=C_A, alpha=0.9, zorder=3, label='Tier A Served')

    bat_discharge = np.maximum(0, s_a - gen)
    ax1.fill_between(time_index, gen, s_a, where=(bat_discharge > 0.01), interpolate=True,
                     facecolor='none', edgecolor=C_DISCHARGE, linewidth=0.0, hatch='////', 
                     zorder=4, label='Battery Discharging')

    dropped_load = np.maximum(0, target_mw - s_a)
    ax1.fill_between(time_index, s_a, target_mw, where=(dropped_load > 0.01), interpolate=True,
                     facecolor=C_DROPPED, alpha=0.85, edgecolor='#4a0000', linewidth=1.2, 
                     hatch='\\\\\\', zorder=4, label='Dropped Load')

    # FIXED: The ax.axhline(target_mw...) target line has been completely removed.

    ax1.set_ylabel("Power\n(MW)", fontweight='bold', color='#2c3e50')
    ax1.set_ylim(0, max(target_mw + 5.0, gen.max() * 1.05))
    format_axes(ax1)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)

    # -------------------------------------------------------------------------
    # BOTTOM PANEL: State of Charge (SoC)
    # -------------------------------------------------------------------------
    max_soc = np.max(soc) if np.max(soc) > 0 else 1
    soc_pct = (soc / max_soc) * 100

    ax2.plot(time_index, soc_pct, color=C_SOC, linewidth=1.5, label='State of Charge (%)', zorder=5)
    ax2.fill_between(time_index, 0, soc_pct, color=C_SOC, alpha=0.15, zorder=3)

    dead_battery = (soc_pct < 0.5)
    ax2.fill_between(time_index, 0, 100, where=dead_battery, interpolate=True, 
                     facecolor=C_DROPPED, alpha=0.1, zorder=1)
    
    dead_idx = np.where(dead_battery)[0]
    if len(dead_idx) > 0:
        ax2.text(time_index[dead_idx[len(dead_idx)//2]], 10, 'BESS Depleted', 
                 color=C_DROPPED, fontsize=8, fontweight='bold', ha='center', zorder=6)

    ax2.set_ylabel("SoC (%)", fontweight='bold', color='#2c3e50')
    ax2.set_ylim(0, 105)
    format_axes(ax2)
    ax2.legend(loc='upper left', frameon=False)

    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    if SAVE_PLOTS: 
        plt.savefig(os.path.join(current_dir, "EMSplot_firm_fail.svg"), bbox_inches='tight')
        
        
if __name__ == "__main__":
    plot_step_1_baseline()
    plot_step_2_firm_storage()
    plot_step_3_flexible_queue()
    plot_step_4_full_hierarchy()
    plot_step_5_b1battery()
    plot_step_6_b1b2()
    plot_step_7_firm_fail() 