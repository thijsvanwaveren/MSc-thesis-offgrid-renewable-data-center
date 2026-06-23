# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:11:56 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.7 - Full Yearly Operation Extraction & LDC Plotting
Extracts 8760-hour chronological data for each capacity, saves it to CSV,
and generates high-contrast Load Duration Curves in the IDE.
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# --- HYDESIGN EXPLICIT IMPORTS & PATHS ---
# =============================================================================
current_dir = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))
hydesign_sys_path = r"C:\Users\thijs\Downloads\hydesign"

if hydesign_sys_path not in sys.path:
    sys.path.insert(0, hydesign_sys_path)

from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model

# =============================================================================
# 1. TARGET OPTIMAL MIXES
# =============================================================================
# Format: (IT_Capacity, Tier_A, Tier_B1, Tier_B2)
target_mixes = [
    (16.0, 8.0, 0.0, 7.0),
    (20.0, 8.0, 0.0, 11.0),
    (30.0, 8.0, 0.0, 17.0),
    (40.0, 8.0, 0.0, 20.0),
    (50.0, 8.0, 0.0, 23.0),
    (75.0, 8.0, 0.0, 28.0),
    (100.0, 8.0, 0.0, 29.0)
]

# High-Contrast Color Palette for easy distinction
C_A = '#08306b'         # Deep Navy (Firm)
C_B1 = '#17becf'        # Teal/Cyan (Daily)
C_B2 = '#d62728'        # Crimson Red (Weekly - Highlights the massive bursts)
C_C = '#7f7f7f'         # Neutral Grey (Opportunistic)
C_GRID = '#e0e0e0'

def configure_parameters(thesis_dir):
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)
    sim_pars['G_MW'] = 0
    sim_pars['battery_charge_efficiency'] = float(np.sqrt(0.86))
    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_offgrid_ldc_temp.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    return temp_fn

# =============================================================================
# 2. MAIN LOOP: SIMULATE, SAVE CSV, & PLOT
# =============================================================================
fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10]
N_life = 25 * 8760
site_name = 'Denmark_good_solar'

examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
ex_site = examples_sites.loc[examples_sites.name == site_name]
weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
sim_pars_fn = configure_parameters(thesis_dir)

os.environ['REWARD_C2'] = '-0.5'

print("\n" + "=" * 80)
print(" GENERATING YEARLY OPERATION CSVs & DURATION CURVES ".center(80))
print("=" * 80)

for cap_mw, a_mw, b1_mw, b2_mw in target_mixes:
    print(f"\n🚀 Simulating {cap_mw} MW Facility (A:{a_mw}, B1:{b1_mw}, B2:{b2_mw})...")
    
    t_a_ts = np.full(N_life, a_mw)
    t_b1_ts = np.full(N_life, b1_mw * 24.0)
    t_b2_ts = np.full(N_life, b2_mw * 168.0)
    total_load_for_ems = np.full(N_life, cap_mw)

    hpp = hpp_model(
        latitude=ex_site['latitude'].values[0],
        longitude=ex_site['longitude'].values[0],
        altitude=ex_site['altitude'].values[0],
        num_batteries=1,
        work_dir=current_dir,
        input_ts_fn=weather_fn,
        sim_pars_fn=sim_pars_fn,
        tier_a_profile=t_a_ts,
        tier_b_profile=t_b1_ts,
        tier_b2_profile=t_b2_ts,
        load_profile_ts=total_load_for_ems,
        battery_deg=False
    )

    out = hpp.evaluate(*fixed_design)
    prob = hpp.prob

    # Extract Year 1 (8760 hours)
    try:
        served_a = prob.get_val('ems.Served_A')[:8760]
        served_b1 = prob.get_val('ems.Served_B')[:8760]
        served_b2 = prob.get_val('ems.Served_B2')[:8760]
        served_c = prob.get_val('ems.Served_C2')[:8760]
    except KeyError:
        print(f"❌ ERROR: Missing OpenMDAO variable for {cap_mw} MW. Skipping.")
        continue
    
    curtailment = prob.get_val('ems.hpp_curt_t')[:8760]
    # -------------------------------------------------------------------------
    # 3. EXPORT 8760-HOUR CHRONOLOGICAL DATA TO CSV
    # -------------------------------------------------------------------------
    df_chrono = pd.DataFrame({
        'Hour_of_Year': np.arange(1, 8761),
        'Tier_A_MW': served_a,
        'Tier_B1_MW': served_b1,
        'Tier_B2_MW': served_b2,
        'Tier_C_MW': served_c,
        'Total_Hardware_Used_MW': served_a + served_b1 + served_b2 + served_c,
        'Curtailment_MW': curtailment
    })
    
    csv_filename = os.path.join(current_dir, f'Yearly_Operation_{cap_mw:.0f}MW.csv')
    df_chrono.to_csv(csv_filename, index=False)
    print(f"   ✅ Saved chronological operation to: Yearly_Operation_{cap_mw:.0f}MW.csv")

    # -------------------------------------------------------------------------
    # 4. PLOTTING THE LDC (High Contrast, No Fills)
    # -------------------------------------------------------------------------
    # Sort arrays descending for Load Duration
    ldc_a = np.sort(served_a)[::-1]
    ldc_b1 = np.sort(served_b1)[::-1]
    ldc_b2 = np.sort(served_b2)[::-1]
    ldc_c = np.sort(served_c)[::-1]

    avg_a = np.mean(served_a)
    avg_b1 = np.mean(served_b1)
    avg_b2 = np.mean(served_b2)
    avg_c = np.mean(served_c)

    x_pct = np.linspace(0, 100, 8760)

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='white')

    # Plot crisp, distinct lines (no confusing fills)
    ax.plot(x_pct, ldc_b2, color=C_B2, linewidth=2.5, zorder=4, label=f'Tier B2 (Weekly) - Avg: {avg_b2:.1f} MW')
    ax.plot(x_pct, ldc_c, color=C_C, linewidth=2, linestyle='-', zorder=2, label=f'Tier C (Opportunistic) - Avg: {avg_c:.1f} MW')
    if avg_b1 > 0.1:
        ax.plot(x_pct, ldc_b1, color=C_B1, linewidth=2.5, zorder=3, label=f'Tier B1 (Daily) - Avg: {avg_b1:.1f} MW')
    ax.plot(x_pct, ldc_a, color=C_A, linewidth=3, zorder=5, label=f'Tier A (Firm) - Avg: {avg_a:.1f} MW')

    # Add the Average Horizontal Lines matching the exact tier colors
    ax.axhline(avg_a, color=C_A, linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
    if avg_b1 > 0.1: 
        ax.axhline(avg_b1, color=C_B1, linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
    if avg_b2 > 0.1: 
        ax.axhline(avg_b2, color=C_B2, linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
    if avg_c > 0.1: 
        ax.axhline(avg_c, color=C_C, linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)

    # Chart Formatting
    ax.set_title(f"Load Duration Curve: {cap_mw} MW Facility", fontsize=14, fontweight='bold', pad=15, color='#333333')
    ax.set_ylabel("Instantaneous Hardware Capacity (MW)", fontsize=11, fontweight='bold', color='#444444')
    ax.set_xlabel("Percentage of the Year (%)", fontsize=11, fontweight='bold', color='#444444')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, cap_mw * 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    ax.grid(axis='both', linestyle='--', alpha=0.4, color=C_GRID, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#444444')

    # Legend placement
    ax.legend(loc='upper right', frameon=True, edgecolor='#e0e0e0', fontsize=10, facecolor='white', framealpha=0.95)

    plt.tight_layout()
    plt.show() # Pops up directly in Spyder

print("\n" + "=" * 80)
print(" All extractions and plots completed! ")
print("=" * 80)