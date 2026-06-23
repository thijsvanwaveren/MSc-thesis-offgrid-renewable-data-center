# -*- coding: utf-8 -*-
"""
Section 3.x / Appendix - End-of-Horizon Sensitivity Analysis
Evaluates the optimal workload portfolios across various facility sizes with Tier C enabled.
Visualizes the final Tier B2 queue against the theoretical 1.92% (168-hour) SLA loophole limit.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- HYDESIGN IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = os.path.abspath(os.path.join(current_dir, '..','..'))
root_dir = os.path.abspath(os.path.join(thesis_dir, '..', '..','..'))
sys.path.append(root_dir)

from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model

# =============================================================================
# 1. PORTFOLIO MIXES
# =============================================================================
portfolios = [
    {"name": "16 MW \nFacility", "A": 8.0, "B1": 0.0, "B2": 7.0,  "Total": 16.0},
    {"name": "20 MW \nFacility", "A": 8.0, "B1": 0.0, "B2": 11.0, "Total": 20.0},
    {"name": "30 MW \nFacility", "A": 8.0, "B1": 0.0, "B2": 17.0, "Total": 30.0},
    {"name": "40 MW \nFacility", "A": 8.0, "B1": 0.0, "B2": 20.0, "Total": 40.0}, 
    {"name": "50 MW \nFacility", "A": 8.0, "B1": 0.0, "B2": 23.0, "Total": 50.0},
    {"name": "75 MW \nFacility", "A": 8.0, "B1": 0.0, "B2": 28.0, "Total": 75.0},
    {"name": "100 MW \nFacility", "A": 8.0, "B1": 0.0, "B2": 29.0, "Total": 100.0},
]

SAVE_PLOTS = True 

# --- STYLING ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10          
plt.rcParams['axes.labelsize'] = 11     
plt.rcParams['xtick.labelsize'] = 10    
plt.rcParams['ytick.labelsize'] = 10 

C_B2 = '#8e44ad'      # Deep Purple for Tier B2
C_LIMIT = '#c0392b'   # Dark Red for the Theoretical Limit

# =============================================================================
# 2. EVALUATION FUNCTION
# =============================================================================
def evaluate_portfolio(tier_a, tier_b1, tier_b2, dc_size, file_suffix):
    """Runs CPLEX for a specific mix and exact DC hardware limit."""
    
    # Enable Tier C opportunistic load (Cost -> Reward)
    os.environ['REWARD_C2'] = '-0.0000001' 

    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
        
    sim_pars['G_MW'] = 0
    sim_pars['battery_charge_efficiency'] = float(np.sqrt(0.86))
    
    temp_fn = os.path.join(thesis_dir, 'inputs', f'hpp_pars_sens_{file_suffix}.yml')
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
    total_load = np.full(N_life, dc_size)

    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=temp_fn,
        tier_a_profile=t_a_ts, tier_b_profile=t_b1_ts, tier_b2_profile=t_b2_ts, load_profile_ts=total_load, battery_deg=False
    )
    
    hpp.evaluate(*fixed_design)

    idx = slice(0, 8760)
    return {
        's_b2': hpp.prob.get_val('ems.Served_B2')[idx]
    }

def track_raw_queue(served, target_mw):
    """Tracks queue accumulation across the full year."""
    q = np.zeros(8760)
    for t in range(1, 8760): 
        q[t] = max(0, q[t-1] + target_mw - served[t])
    return q

# =============================================================================
# 3. RUN SIMULATIONS & COLLECT RESULTS
# =============================================================================
results_actual_queue = []
results_limit_mwh = []
results_pct_hidden = []
labels = []

print("=" * 70)
print(f"{'Portfolio':<10} | {'DC Size':<8} | {'Final Queue':<12} | {'SLA Limit':<12} | {'% Hidden'}")
print("-" * 70)

for idx, port in enumerate(portfolios):
    print(f"Running {port['name']}...", end="\r")
    res = evaluate_portfolio(port['A'], port['B1'], port['B2'], port['Total'], f"mix{idx+1}")
    
    q_b2 = track_raw_queue(res['s_b2'], port['B2'])
    final_q_b2 = q_b2[-1]
    
    # Calculate Limits & Percentages
    annual_b2_load = port['B2'] * 8760.0
    sla_limit_mwh = port['B2'] * 168.0
    pct_hidden = (final_q_b2 / annual_b2_load) * 100 if annual_b2_load > 0 else 0
    
    results_actual_queue.append(final_q_b2)
    results_limit_mwh.append(sla_limit_mwh)
    results_pct_hidden.append(pct_hidden)
    
    labels.append(f"{port['name']}")
    
    print(f"{port['name']:<10} | {port['Total']:<5} MW | {final_q_b2:>8.1f} MWh | {sla_limit_mwh:>8.1f} MWh | {pct_hidden:>6.2f}%")

print("=" * 70)

# =============================================================================
# 4. PLOT BAR CHART 
# =============================================================================
x = np.arange(len(portfolios))
width = 0.55

fig, ax = plt.subplots(figsize=(8.5, 5), facecolor='white')

# Plot 1: The Hollow "Theoretical Limit" Outline (1.92% of Annual Load)
# rects_limit = ax.bar(x, results_limit_mwh, width, color='none', edgecolor=C_LIMIT, 
#                      linewidth=1.5, linestyle='--', label='Theoretical Maximum Loophole (1.92% Annual Load)')

# Plot 2: The Solid "Actual Final Queue"
rects_actual = ax.bar(x, results_actual_queue, width, color=C_B2, alpha=0.85, 
                      edgecolor='#222222', linewidth=1.0, label='Final Tier B2 Queue Volume')

ax.set_ylabel('Final Tier B2 Queue Volume (MWh)', fontweight='bold', color='#222222')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Clean, minimalist axes formatting
ax.grid(axis='y', linestyle='-', alpha=0.2, color='#b0b0b0')
ax.grid(axis='x', visible=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('#222222')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('#222222')
ax.tick_params(axis='x', length=0) 

# Ensure Y-axis starts exactly at 0 and adds padding on top for text
ax.set_ylim(0, max(results_limit_mwh) * 1.2)

# Annotate the exact hidden percentage on top of the bars
for i, rect in enumerate(rects_actual):
    height = rect.get_height()
    pct = results_pct_hidden[i]
    if height > 0.01:
        # Display the % value
        ax.annotate(f'{pct:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#222222')

# Place the legend neatly above the plot
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False)

plt.tight_layout()
if SAVE_PLOTS: 
    plt.savefig(os.path.join(current_dir, "EMSplot_Sensitivity_FinalQueues_exclC.svg"), bbox_inches='tight')
print("\n✅ Saved plot as EMSplot_Sensitivity_FinalQueues.svg")
plt.show()