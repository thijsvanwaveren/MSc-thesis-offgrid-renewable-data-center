# -*- coding: utf-8 -*-
"""
Section 3.2 - Final 2-Panel EMS Dispatch Visualization
Features Logical Stacking to eliminate CPLEX zigzag artifacts,
vibrant colors for Tier C, and integrated catch-up/deferral dynamics.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- HYDESIGN IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..'))
sys.path.append(root_dir)

from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model

# =============================================================================
# 1. INPUTS
# =============================================================================
MAX_IT_MW = 16.0
TIER_A_MW = 5
TIER_B1_MW = 6
TIER_B2_MW = 4
START_HOUR = 450  
WINDOW_HOURS = 72

# Upgraded Colors
C_A  = '#08306b'  
C_B1 = '#2879b9'  
C_B2 = '#73b3d8'  
C_C  = '#00b4d8'  # Vibrant Cyan to make the 1 MW sliver pop
C_CATCHUP = '#2ecc71' # Bright Green
C_GEN = '#ff9f1c' 
C_SOC = '#e71d36' 

# =============================================================================
# 2. SIMULATION SETUP & EXECUTION
# =============================================================================
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

print(f"Running CPLEX Solver for Mix: {TIER_A_MW}A / {TIER_B1_MW}B1 / {TIER_B2_MW}B2")
N_life = 25 * 8760
fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10]
site_name = 'Denmark_good_solar'
examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
ex_site = examples_sites.loc[examples_sites.name == site_name]
weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
sim_pars_fn = configure_parameters(thesis_dir)

t_a_ts = np.full(N_life, TIER_A_MW)
t_b1_ts = np.full(N_life, TIER_B1_MW * 24.0) 
t_b2_ts = np.full(N_life, TIER_B2_MW * 168.0) 
total_load = np.full(N_life, MAX_IT_MW)

hpp = hpp_model(
    latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
    num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
    tier_a_profile=t_a_ts, tier_b_profile=t_b1_ts, tier_b2_profile=t_b2_ts, load_profile_ts=total_load, battery_deg=False
)
hpp.evaluate(*fixed_design)



# =============================================================================
# 3. DATA EXTRACTION & LOGICAL STACKING
# =============================================================================
idx = slice(START_HOUR, START_HOUR + WINDOW_HOURS)
hours = np.arange(WINDOW_HOURS)

wind = hpp.prob.get_val('ems.wind_t_ext')[idx]
solar = hpp.prob.get_val('ems.solar_t_ext')[idx]
generation = wind + solar
soc = hpp.prob.get_val('ems.b_E_SOC_t')[idx] 

served_a = hpp.prob.get_val('ems.Served_A')[idx]
served_b1_raw = hpp.prob.get_val('ems.Served_B')[idx]
served_b2_raw = hpp.prob.get_val('ems.Served_B2')[idx]
served_c = hpp.prob.get_val('ems.Served_C2')[idx]

# --- LOGICAL STACKING (Flattens the CPLEX Bang-Bang Artifacts) ---
actual_b_total = served_b1_raw + served_b2_raw

logical_b1 = np.minimum(TIER_B1_MW, actual_b_total)
logical_b2 = np.minimum(TIER_B2_MW, actual_b_total - logical_b1)
catchup_mw = actual_b_total - logical_b1 - logical_b2

target_base = TIER_A_MW + TIER_B1_MW + TIER_B2_MW
actual_base = served_a + logical_b1 + logical_b2
deferred_mw = np.maximum(0, target_base - actual_base)
total_served = actual_base + catchup_mw + served_c

# --- RECONSTRUCT QUEUES ---
arrival_b1, arrival_b2 = TIER_B1_MW, TIER_B2_MW
queue_b1 = np.zeros(WINDOW_HOURS)
queue_b2 = np.zeros(WINDOW_HOURS)

start_queue_b1 = max(0, (START_HOUR * arrival_b1) - np.sum(hpp.prob.get_val('ems.Served_B')[:START_HOUR]))
start_queue_b2 = max(0, (START_HOUR * arrival_b2) - np.sum(hpp.prob.get_val('ems.Served_B2')[:START_HOUR]))

queue_b1[0] = max(0, start_queue_b1 + arrival_b1 - served_b1_raw[0])
queue_b2[0] = max(0, start_queue_b2 + arrival_b2 - served_b2_raw[0])

for t in range(1, WINDOW_HOURS):
    queue_b1[t] = max(0, queue_b1[t-1] + arrival_b1 - served_b1_raw[t])
    queue_b2[t] = max(0, queue_b2[t-1] + arrival_b2 - served_b2_raw[t])

# =============================================================================
# 4. CONDENSED 2-PANEL PLOT
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, facecolor='white', gridspec_kw={'height_ratios': [1.8, 1]})

# --- Panel 1: Instantaneous Power Balance (MW) ---
# 1. Plot the Logically Stacked Workloads (Native stacking prevents overlaps)
ax1.stackplot(hours, served_a, logical_b1, logical_b2, catchup_mw, served_c, 
              colors=[C_A, C_B1, C_B2, C_CATCHUP, C_C], 
              labels=['Tier A', 'Tier B1 (Base)', 'Tier B2 (Base)', 'Catch-up Processing', 'Tier C (Opportunistic)'], 
              alpha=0.9)

# 2. Target Demand Line
ax1.plot(hours, np.full_like(hours, target_base), color='#333333', linewidth=1.5, linestyle=':', label=f'Target Demand ({target_base} MW)')

# 3. Highlight Queue Accumulation (Ghost Workload)
ax1.fill_between(hours, actual_base, target_base, where=(deferred_mw > 0), 
                 facecolor='none', edgecolor='#555555', hatch='///', linewidth=0, label='Workload Deferred (Enters Queue)')

# 4. Renewable Generation & Battery Discharge
ax1.plot(hours, generation, color=C_GEN, linewidth=3, linestyle='--', label='Renewable Generation')
battery_discharge_gap = np.maximum(0, total_served - generation)
ax1.fill_between(hours, generation, generation + battery_discharge_gap, 
                 where=(battery_discharge_gap > 0.1), color='#e74c3c', alpha=0.4, hatch='\\\\', label='Battery Discharge')

#ax1.set_title(f"3-Day Power & Buffer Dynamics | Mix: {TIER_A_MW}A / {TIER_B1_MW}B1 / {TIER_B2_MW}B2", fontsize=15, fontweight='bold', pad=15)
ax1.set_ylabel("Power (MW)", fontweight='bold', fontsize=11)
ax1.axhline(MAX_IT_MW, color='black', linestyle='-', alpha=0.5, label='16 MW Hardware Limit')
ax1.set_ylim(0, max(MAX_IT_MW + 5, generation.max() * 1.1))

# --- Panel 2: Energy Buffers (MWh) ---
ax2_twin = ax2.twinx()

line_soc = ax2.plot(hours, soc/2, color=C_SOC, linewidth=2.5, label='Battery State of Charge (%)')
ax2.fill_between(hours, 0, soc/2, color=C_SOC, alpha=0.1)

line_q1 = ax2_twin.plot(hours, queue_b1, color=C_B1, linewidth=2, linestyle='-', label='Tier B1 Queue Size (MWh)')
line_q2 = ax2_twin.plot(hours, queue_b2, color=C_B2, linewidth=3, linestyle='-.', label='Tier B2 Queue Size (MWh)')

ax2.set_ylabel("Battery State of Charge (%)", fontweight='bold', color=C_SOC, fontsize=11)
ax2_twin.set_ylabel("Workload Queues (MWh)", fontweight='bold', color='#1f618d', fontsize=11)
ax2.set_xlabel("Time (Hours within 3-Day Window)", fontsize=12, fontweight='bold')
ax2.set_xlim(0, WINDOW_HOURS - 1)

ax2.set_ylim(0, max(soc.max()/2 * 1.15, 10))
ax2_twin.set_ylim(0, max(queue_b1.max(), queue_b2.max(), 5) * 1.15)

# --- Global Formatting ---
for ax in [ax1, ax2]:
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.grid(axis='y', linestyle='--', alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
ax2_twin.spines['top'].set_visible(False)

# Custom Unified Legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles3, labels3 = ax2_twin.get_legend_handles_labels()
fig.legend(handles1 + handles2 + handles3, labels1 + labels2 + labels3, 
           loc='lower center', bbox_to_anchor=(0.5, 1.07), ncol=4, frameon=False, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(top=0.82, hspace=0.15) 

plot_fn = os.path.join(current_dir, f"EMS_Dispatch_{TIER_A_MW}A_{TIER_B1_MW}B1_{TIER_B2_MW}B2.svg")
plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
print(f"✅ Plot saved successfully to {plot_fn}")
plt.show()