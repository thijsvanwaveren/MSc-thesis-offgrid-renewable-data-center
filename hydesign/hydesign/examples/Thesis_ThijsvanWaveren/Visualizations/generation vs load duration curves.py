# -*- coding: utf-8 -*-
"""
Section 3.5.2 - Load vs. Generation Duration Curves (BESS Adjusted)

Updates:
- Added a pure Raw Generation duration curve without workloads or BESS.
- Customized the 8 MW Baseline plot to explicitly show Raw vs. Adjusted generation,
  revealing statistical BESS charging (left) and discharging (right).
- Shifts the 'Average served load' box up and left to avoid the generation curve.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# =============================================================================
# 1. SETUP & PATHS
# =============================================================================

base_dir = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren"
scripts_dir = os.path.join(base_dir, "scripts")
vis_dir = os.path.join(base_dir, "Visualizations")

# Split lists: Process ALL for individual, but only MASTER for the combined plot
ALL_IT_CAPACITIES_MW = [16.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0]
MASTER_IT_CAPACITIES_MW = [16.0, 30.0, 50.0, 75.0, 100.0]

PUE = 1.15
FIRM_TIER_A_IT = 8.0
FIRM_TIER_A_FACILITY = FIRM_TIER_A_IT * PUE

C_GEN       = '#d35400'
C_CURTAIL   = '#fdf2e9'
C_FIRM      = '#04234b'       # Dark Navy for Firm Tier A Baseline
C_FLEX      = '#2b83ba'       # Mid-tone Cerulean for Flexible Upside
C_DISCHARGE = '#e74c3c'       # Red for BESS Discharge
C_CHARGE    = '#00cc66'       # Green for BESS Charge

# 5 Colors matching the 5 Master Capacities
COLORS_IT_MASTER = [
    '#04234b', # 16 MW
    '#0b559f', # 30 MW
    '#2b83ba', # 50 MW
    '#63b1d3', # 75 MW
    '#a1d9e7'  # 100 MW
]

print("\n" + "=" * 80)
print(" GENERATING BESS-ADJUSTED DURATION PLOTS ".center(80))
print("=" * 80)

# =============================================================================
# 2. LOAD DATA
# =============================================================================

gen_csv_path = os.path.join(vis_dir, 'Generation_Data.csv')
if not os.path.exists(gen_csv_path):
    raise FileNotFoundError(f"Missing Generation file here: {gen_csv_path}")

df_gen = pd.read_csv(gen_csv_path)

available_re = df_gen['Wind'] + df_gen['Solar']
raw_gen_sorted = np.sort(available_re.values)[::-1]
x_pct = np.linspace(0, 100, len(raw_gen_sorted))

all_loads_sorted = {}
all_net_gen_sorted = {} 

for cap_mw in ALL_IT_CAPACITIES_MW:
    op_csv_path = os.path.join(scripts_dir, f'Yearly_Operation_{cap_mw:.0f}MW.csv')

    if os.path.exists(op_csv_path):
        df_op = pd.read_csv(op_csv_path)

        # IT-side load converted to facility-side electrical demand
        load_it = df_op['Total_Hardware_Used_MW'].values
        load_facility = load_it * PUE
        all_loads_sorted[cap_mw] = np.sort(load_facility)[::-1]
        
        # Calculate Net HPP Output
        if 'Curtailment_MW' in df_op.columns:
            curt = df_op['Curtailment_MW'].values
            net_hpp_chronological = load_facility + curt
            all_net_gen_sorted[cap_mw] = np.sort(net_hpp_chronological)[::-1]
        else:
            all_net_gen_sorted[cap_mw] = raw_gen_sorted
    else:
        print(f" ⚠️ Missing operation file for {cap_mw:.0f} MW_IT: {op_csv_path}")


# =============================================================================
# 3. PLOTTING FUNCTIONS
# =============================================================================

def plot_raw_generation(raw_gen_sorted, x_pct):
    """Generates the introductory pure raw generation duration curve."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor='white')

    ax.plot(x_pct, raw_gen_sorted, color=C_GEN, linewidth=2.5, linestyle='-', zorder=10)
    ax.fill_between(x_pct, 0, raw_gen_sorted, facecolor=C_GEN, alpha=0.1, zorder=1)

    ax.set_ylabel("HPP generation / Facility power (MW$_{el}$)", fontsize=12, fontweight='bold', color='#222222')
    ax.set_xlabel("Percentage of the Year (%)", fontsize=12, fontweight='bold', color='#222222')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, raw_gen_sorted.max() * 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    secax = ax.secondary_yaxis('right', functions=(lambda y: y / PUE, lambda y: y * PUE))
    secax.set_ylabel("Workload power (MW$_{IT}$)", fontsize=11, fontweight='bold', color='#222222')
    secax.tick_params(axis='y', colors='#222222', labelsize=10)

    ax.grid(axis='y', linestyle='-', alpha=0.2, color='#b0b0b0', zorder=0)
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['left'].set_color('#222222')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#222222')

    custom_legend = [
        Line2D([0], [0], color=C_GEN, lw=2.5, linestyle='-', label='Wind + Solar generation')
    ]
    ax.legend(handles=custom_legend, loc='upper right', frameon=True, edgecolor='#e0e0e0', fontsize=10, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    out_fn_svg = os.path.join(vis_dir, 'LoadvsGenerationDuration_RawGenOnly.svg')
    plt.savefig(out_fn_svg, dpi=300, bbox_inches='tight')
    #plt.close(fig)
    print(" ✅ Saved Raw Generation plot.")


def plot_8mw_baseline(raw_gen_sorted, adj_gen_sorted, x_pct):
    """Bespoke plot for 8 MW to explicitly show BESS statistical transformation."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor='white')
    cap_facility = FIRM_TIER_A_FACILITY
    
    # 1. Dark Blue Firm Baseline
    firm_load_actual = np.minimum(adj_gen_sorted, cap_facility)
    ax.fill_between(x_pct, 0, firm_load_actual, color=C_FIRM, alpha=0.9, zorder=3)

    # 2. Curtailment Area (Above load, up to adjusted gen)
    ax.fill_between(x_pct, cap_facility, adj_gen_sorted, 
                    where=(adj_gen_sorted > cap_facility),
                    facecolor=C_CURTAIL, edgecolor='#e67e22', hatch='\\\\', alpha=0.65, zorder=2)
                    
    # 3. BESS Shifting Visualization
    # Charge (Left side: Raw > Adj)
    ax.fill_between(x_pct, adj_gen_sorted, raw_gen_sorted, 
                    where=(raw_gen_sorted > adj_gen_sorted), 
                    facecolor=C_CHARGE, alpha=0.6, hatch='\\\\', zorder=4)
                    
    # Discharge (Right side: Adj > Raw)
    ax.fill_between(x_pct, raw_gen_sorted, adj_gen_sorted, 
                    where=(adj_gen_sorted > raw_gen_sorted), 
                    facecolor=C_DISCHARGE, alpha=0.8, hatch='//', zorder=4)

    # 4. Trajectory Lines
    ax.plot(x_pct, raw_gen_sorted, color='#7f8c8d', linewidth=1.5, linestyle='--', zorder=5) 
    ax.plot(x_pct, adj_gen_sorted, color=C_GEN, linewidth=2.0, zorder=6)

    # Load Reference Line (Text removed per request)
    ax.axhline(y=cap_facility, color=C_FIRM, linestyle=':', linewidth=1.6, alpha=0.95, zorder=6)

    ax.set_ylabel("HPP generation / Facility power (MW$_{el}$)", fontsize=12, fontweight='bold', color='#222222')
    ax.set_xlabel("Percentage of the Year (%)", fontsize=12, fontweight='bold', color='#222222')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(adj_gen_sorted.max(), raw_gen_sorted.max()) * 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    secax = ax.secondary_yaxis('right', functions=(lambda y: y / PUE, lambda y: y * PUE))
    secax.set_ylabel("Workload power (MW$_{IT}$)", fontsize=11, fontweight='bold', color='#222222')
    secax.tick_params(axis='y', colors='#222222', labelsize=10)

    ax.grid(axis='y', linestyle='-', alpha=0.2, color='#b0b0b0', zorder=0)
    ax.grid(axis='x', visible=False)
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.2); ax.spines['left'].set_color('#222222')
    ax.spines['bottom'].set_linewidth(1.2); ax.spines['bottom'].set_color('#222222')

    # FIX: Use mcolors.to_rgba for facecolor and remove the alpha kwarg to make stripes visible
    custom_legend = [
        Line2D([0], [0], color='#7f8c8d', lw=1.5, linestyle='--', label='Wind + Solar generation'),
        Line2D([0], [0], color=C_GEN, lw=2.0, linestyle='-', label='Dispatched generation'),
        Patch(facecolor=mcolors.to_rgba(C_CHARGE, 0.6), edgecolor=C_CHARGE, hatch='\\\\', label='BESS Charging'),
        Patch(facecolor=mcolors.to_rgba(C_DISCHARGE, 0.8), edgecolor=C_DISCHARGE, hatch='//', label='BESS Discharging'),
        Patch(facecolor=C_CURTAIL, edgecolor='#e67e22', hatch='\\\\', label='Curtailed Power'),
        Patch(facecolor=C_FIRM, alpha=0.9, label='Firm Tier A Load')
    ]
    ax.legend(handles=custom_legend, loc='upper right', frameon=True, edgecolor='#e0e0e0', fontsize=9, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    out_fn_svg = os.path.join(vis_dir, f'LoadvsGenerationDuration_{FIRM_TIER_A_IT:.0f}MWIT.svg')
    plt.savefig(out_fn_svg, dpi=300, bbox_inches='tight')
    #plt.close(fig)
    print(f" ✅ Saved pure Baseline Net HPP plot for {FIRM_TIER_A_IT:.0f} MW_IT.")


def plot_individual_capacity(cap_mw, load_facility, net_gen_sorted, x_pct):
    """Standard Flexible Duration Plot for >= 16 MW."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor='white')
    cap_facility = cap_mw * PUE
    
    avg_load_facility = np.mean(load_facility)
    avg_load_it = avg_load_facility / PUE

    ax.axhline(y=avg_load_facility, color='black', linestyle='-.', linewidth=1.5, alpha=0.85, zorder=7)
    ax.text(75, avg_load_facility + 20, f"Average served load\n{avg_load_it:.1f} MW$_{{IT}}$",
            fontsize=9, fontweight='bold', color='black', ha='left', va='bottom', zorder=9,
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.75, pad=3))

    ax.fill_between(x_pct, load_facility, net_gen_sorted, where=(net_gen_sorted > load_facility),
                    facecolor=C_CURTAIL, edgecolor='#e67e22', hatch='\\\\', alpha=0.65, zorder=1)

    firm_load_actual = np.minimum(load_facility, FIRM_TIER_A_FACILITY)
    ax.fill_between(x_pct, 0, firm_load_actual, color=C_FIRM, alpha=0.9, zorder=4)

    if cap_mw > FIRM_TIER_A_IT:
        ax.fill_between(x_pct, firm_load_actual, load_facility, color=C_FLEX, alpha=0.62, zorder=3)
        ax.plot(x_pct, load_facility, color=C_FLEX, linewidth=2.0, zorder=5)

    ax.plot(x_pct, net_gen_sorted, color=C_GEN, linewidth=2.0, linestyle='--', zorder=10)

    ax.axhline(y=cap_facility, color=C_FLEX, linestyle=':', linewidth=1.6, alpha=0.95, zorder=6)
    label_text = f"{cap_mw:.0f} MW$_{{IT}}$\n{cap_facility:.1f} MW$_{{el}}$ facility"
    y_pos = cap_facility * 0.5
        
    ax.text(2, y_pos, label_text, color='white', fontweight='bold', fontsize=10, 
            ha='left', va='center', zorder=8)

    ax.set_ylabel("HPP generation / Facility power (MW$_{el}$)", fontsize=12, fontweight='bold', color='#222222')
    ax.set_xlabel("Percentage of the Year (%)", fontsize=12, fontweight='bold', color='#222222')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(net_gen_sorted.max(), raw_gen_sorted.max()) * 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    secax = ax.secondary_yaxis('right', functions=(lambda y: y / PUE, lambda y: y * PUE))
    secax.set_ylabel("Workload power (MW$_{IT}$)", fontsize=11, fontweight='bold', color='#222222')
    secax.tick_params(axis='y', colors='#222222', labelsize=10)

    ax.grid(axis='y', linestyle='-', alpha=0.2, color='#b0b0b0', zorder=0)
    ax.grid(axis='x', visible=False)
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.2); ax.spines['left'].set_color('#222222')
    ax.spines['bottom'].set_linewidth(1.2); ax.spines['bottom'].set_color('#222222')

    custom_legend = [
        Line2D([0], [0], color=C_GEN, lw=2.5, linestyle='--', label='Dispatched generation'),
        Patch(facecolor=C_CURTAIL, edgecolor='#e67e22', hatch='\\\\', label='Curtailed Power'),
        Patch(facecolor=C_FLEX, alpha=0.65, label=f'Flexible demand (Tiers B & C)'),
        Patch(facecolor=C_FIRM, alpha=0.9, label='Firm Tier A Load')
    ]
    ax.legend(handles=custom_legend, loc='upper right', frameon=True, edgecolor='#e0e0e0', fontsize=9, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    out_fn_svg = os.path.join(vis_dir, f'LoadvsGenerationDuration_{cap_mw:.0f}MWIT.svg')
    plt.savefig(out_fn_svg, dpi=300, bbox_inches='tight')
    #plt.close(fig)
    print(f" ✅ Saved individual Net HPP plot for {cap_mw:.0f} MW_IT.")


# =============================================================================
# 4. MASTER PLOT (Excludes 20 & 40 MW, uses Raw Generation Envelope)
# =============================================================================

if all_loads_sorted:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

    for idx, cap in reversed(list(enumerate(MASTER_IT_CAPACITIES_MW))):
        if cap in all_loads_sorted:
            load_facility = all_loads_sorted[cap]
            color = COLORS_IT_MASTER[idx]

            ax.fill_between(x_pct, 0, load_facility, color=color, alpha=0.55, zorder=3)
            ax.plot(x_pct, load_facility, color=color, linewidth=1.2, zorder=4, alpha=1.0)

            lower_cap = MASTER_IT_CAPACITIES_MW[idx - 1] if idx > 0 else 0.0
            label_y = ((lower_cap * PUE) + (cap * PUE)) / 2
            ax.text(
                4, label_y, f"{cap:.0f} MW$_{{IT}}$", color='white', fontweight='bold',
                fontsize=10, ha='left', va='center', zorder=8,
                path_effects=[path_effects.withStroke(linewidth=1, foreground='black', alpha=0.35)]
            )

    ax.plot(x_pct, raw_gen_sorted, color=C_GEN, linewidth=1.8, linestyle='--', zorder=10)

    ax.set_ylabel("HPP generation / Facility power (MW$_{el}$)", fontsize=12, fontweight='bold', color='#222222')
    ax.set_xlabel("Percentage of the Year (%)", fontsize=12, fontweight='bold', color='#222222')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, raw_gen_sorted.max() * 1.02)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    secax = ax.secondary_yaxis('right', functions=(lambda y: y / PUE, lambda y: y * PUE))
    secax.set_ylabel("Workload power (MW$_{IT}$)", fontsize=11, fontweight='bold', color='#222222')
    secax.tick_params(axis='y', colors='#222222', labelsize=10)

    ax.grid(axis='y', linestyle='-', alpha=0.2, color='#b0b0b0', zorder=0)
    ax.grid(axis='x', visible=False)
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.2); ax.spines['left'].set_color('#222222')
    ax.spines['bottom'].set_linewidth(1.2); ax.spines['bottom'].set_color('#222222')

    custom_legend = [
        Line2D([0], [0], color=C_GEN, lw=2.5, linestyle='--', label='Wind + Solar generation'),
        Patch(facecolor=COLORS_IT_MASTER[-1], alpha=0.65, label=f'Data center facility demand (IT load × {PUE:.2f})')
    ]
    ax.legend(handles=custom_legend, loc='upper right', frameon=True, edgecolor='#e0e0e0', fontsize=10, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    master_svg = os.path.join(vis_dir, 'Master_LoadvsGenerationDuration_RawGen.svg')
    plt.savefig(master_svg, dpi=300, bbox_inches='tight')
    #plt.close(fig)
    print(" ✅ Saved Master Plot (Excluding 20 & 40 MW).")


# =========================================================================
# 5. GENERATE THE SPECIFIC PLOTS
# =========================================================================

# 0. Generate the pure Raw Generation introductory plot
print(" 🚀 Generating Intro Raw Generation Scenario...")
plot_raw_generation(raw_gen_sorted=raw_gen_sorted, x_pct=x_pct)

# A. Generate ALL individual plots for flexible loads
for cap_mw in ALL_IT_CAPACITIES_MW:
    if cap_mw in all_loads_sorted and cap_mw in all_net_gen_sorted:
        plot_individual_capacity(
            cap_mw=cap_mw,
            load_facility=all_loads_sorted[cap_mw],
            net_gen_sorted=all_net_gen_sorted[cap_mw],
            x_pct=x_pct
        )

# B. Generate the pure "8 MW Baseline" Scenario with precise BESS Chronological Math
print(" 🚀 Generating Baseline 8 MW Tier A Scenario (with Chronological BESS)...")

# Simulate basic chronological 300MWh battery acting on an 8 MW load
soc = 150.0
adj_gen = np.zeros(len(available_re))
charge_eff = np.sqrt(0.86)

for i in range(len(available_re)):
    gen = available_re.values[i]
    if gen > FIRM_TIER_A_FACILITY:
        surplus = gen - FIRM_TIER_A_FACILITY
        ch_actual = min(surplus, 35.0, (300.0 - soc) / charge_eff) # 35MW is assumed batt limit
        soc += ch_actual * charge_eff
        adj_gen[i] = gen - ch_actual
    else:
        deficit = FIRM_TIER_A_FACILITY - gen
        dis_actual = min(deficit, 35.0, (soc - 30.0) * charge_eff) # 30MWh min SoC limit
        soc -= dis_actual / charge_eff
        adj_gen[i] = gen + dis_actual
        
# Sort the Adjusted Generation independently to show statistical transformation
adj_gen_sorted = np.sort(adj_gen)[::-1]

plot_8mw_baseline(
    raw_gen_sorted=raw_gen_sorted,
    adj_gen_sorted=adj_gen_sorted,
    x_pct=x_pct
)

print("\n" + "=" * 80)
print(" All visualization completed successfully! ")
print("=" * 80)