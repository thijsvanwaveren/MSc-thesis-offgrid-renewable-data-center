# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:29:36 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Evaluate Value of Capacity Blocks vs Pure Baseload
Calculates exact curtailment reduction and generates Stacked Bar Chart.
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

# Ensure this points to your correct assembly file
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

def plot_baseload_vs_blocks(avg_base_served, avg_base_curt, avg_total_served, avg_blocks_curt):
    """Generates the stacked bar chart comparing the two runs."""
    
    # Calculate the extra capacity unlocked by the blocks
    avg_blocks_served = avg_total_served - avg_base_served
    
    labels = ['Annual Firm Load\n(Stage 1)', '+Capacity Blocks\n(Stage 2)']
    
    # Bar Data Arrays
    tier_a_mw = np.array([avg_base_served, avg_base_served])
    tier_b2_mw = np.array([0.0, avg_blocks_served])
    curtailed_mw = np.array([avg_base_curt, avg_blocks_curt])

    fig, ax = plt.subplots(figsize=(8, 7))

    # TU Delft Colors
    colors = {
        'A': '#00549F',      # Tier A Baseload (Blue)
        'B2': '#73A4FF',     # Tier B2 Weekly Extra (Brown)
        'Curtail': '#E63946' # Wasted / Curtailed (Red)
    }

    width = 0.55

    # Stack the bars
    bars_a = ax.bar(labels, tier_a_mw, width, 
                    label='Annual Firm Load', color=colors['A'], edgecolor='black')
    
    bars_b2 = ax.bar(labels, tier_b2_mw, width, bottom=tier_a_mw, 
                     label='Weekly Capacity Blocks', color=colors['B2'], edgecolor='black')
    
    bars_curt = ax.bar(labels, curtailed_mw, width, bottom=tier_a_mw + tier_b2_mw, 
                       label='Curtailed Power', color=colors['Curtail'], edgecolor='black', hatch='//')

    ax.set_ylabel('Average Power Served (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Value of Capacity Blocks vs Pure Annual Firm Load', fontsize=14, fontweight='bold', pad=20)

    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, framealpha=1, edgecolor='black')

    # Add text annotations inside the bars
    for i in range(len(labels)):
        # Baseload
        if tier_a_mw[i] > 1.0:
            ax.text(i, tier_a_mw[i]/2, f'{tier_a_mw[i]:.1f} MW', 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        # Blocks
        if tier_b2_mw[i] > 1.0:
            y_pos = tier_a_mw[i] + (tier_b2_mw[i] / 2)
            ax.text(i, y_pos, f'{tier_b2_mw[i]:.1f} MW', 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        # Curtailment
        if curtailed_mw[i] > 1.0:
            y_pos = tier_a_mw[i] + tier_b2_mw[i] + (curtailed_mw[i] / 2)
            ax.text(i, y_pos, f'{curtailed_mw[i]:.1f} MW', 
                    ha='center', va='center', color='black', fontweight='bold', fontsize=11, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.tight_layout()
    plot_fn = os.path.join(current_dir, 'Thesis_Baseload_vs_Blocks_Bar.svg')
    plt.savefig(plot_fn, format='svg', dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot successfully saved to: {plot_fn}")
    plt.show()

def run_evaluation():
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10] 
    MAX_IT_CAPACITY_MW = 55.0 
    N_life_1yr = 8760
    N_life_total = 25 * 8760
    
    site_name = 'Denmark_good_solar'
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    # STRICTLY DISABLE TIER C2
    # Setting to 1.0 creates a positive cost penalty in CPLEX. 
    # Because CPLEX minimizes cost, it will force Tier C2 to 0 MW.
    os.environ['REWARD_C2'] = '1.0'  

    # =========================================================================
    # 1. LOAD THE CSV AND CREATE PROFILES
    # =========================================================================
    csv_fn = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\Results\Weekly\annual simulation weekly result\Weekly_Capacity_Blocks_yearlysimulation.csv"
    
    try:
        df_blocks = pd.read_csv(csv_fn)
        weekly_firm_mw = df_blocks['Total_Firm_Load_MW'].values
        print(f"✅ Loaded {len(weekly_firm_mw)} weekly capacity blocks from CSV.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find CSV at {csv_fn}")
        sys.exit(1)

    # Create the Capacity Blocks 1-year profile
    dynamic_blocks_1yr = np.zeros(N_life_1yr)
    for w in range(52):
        dynamic_blocks_1yr[w*168 : (w+1)*168] = weekly_firm_mw[w]
    dynamic_blocks_1yr[52*168:] = weekly_firm_mw[-1]
    
    # Identify the base tier (Minimum of the blocks, usually 7.0 MW)
    baseload_mw = np.min(weekly_firm_mw)
    pure_baseload_1yr = np.full(N_life_1yr, baseload_mw)

    # Empty profiles for unused tiers
    t_zero_25yr = np.zeros(N_life_total)
    
    # Prepare full 25-year profiles for OpenMDAO
    pure_baseload_25yr = np.tile(pure_baseload_1yr, 25)
    dynamic_blocks_25yr = np.tile(dynamic_blocks_1yr, 25)

    # =========================================================================
    # 2. RUN SCENARIO 1: PURE BASELOAD
    # =========================================================================
    print(f"\n--- Running SCENARIO 1: Pure Baseload ({baseload_mw} MW) ---")
    
    load_ts_base = pure_baseload_25yr.copy()
    load_ts_base[0] = MAX_IT_CAPACITY_MW
    
    hpp_base = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=pure_baseload_25yr, tier_b_profile=t_zero_25yr, tier_b2_profile=t_zero_25yr, 
        load_profile_ts=load_ts_base, battery_deg=False,
        run_mode='yearly' 
    )
    hpp_base.evaluate(*fixed_design)
    
    # Extract Baseload Metrics
    curt_base_ts = hpp_base.prob.get_val('ems.hpp_curt_t')[:8760]
    unserved_base_ts = hpp_base.prob.get_val('ems.Unserved_A')[:8760]
    served_base_ts = pure_baseload_1yr - unserved_base_ts
    
    avg_base_served = np.mean(served_base_ts)
    avg_base_curt = np.mean(curt_base_ts)

    # =========================================================================
    # 3. RUN SCENARIO 2: CAPACITY BLOCKS
    # =========================================================================
    print(f"\n--- Running SCENARIO 2: Capacity Blocks ---")
    
    load_ts_blocks = dynamic_blocks_25yr.copy()
    load_ts_blocks[0] = MAX_IT_CAPACITY_MW
    
    hpp_blocks = hpp_model(
        latitude=ex_site['latitude'].values[0], longitude=ex_site['longitude'].values[0], altitude=ex_site['altitude'].values[0],
        num_batteries=1, work_dir=current_dir, input_ts_fn=weather_fn, sim_pars_fn=sim_pars_fn,
        tier_a_profile=dynamic_blocks_25yr, tier_b_profile=t_zero_25yr, tier_b2_profile=t_zero_25yr, 
        load_profile_ts=load_ts_blocks, battery_deg=False,
        run_mode='yearly' 
    )
    hpp_blocks.evaluate(*fixed_design)
    
    # Extract Capacity Blocks Metrics
    curt_blocks_ts = hpp_blocks.prob.get_val('ems.hpp_curt_t')[:8760]
    unserved_blocks_ts = hpp_blocks.prob.get_val('ems.Unserved_A')[:8760]
    served_blocks_ts = dynamic_blocks_1yr - unserved_blocks_ts
    
    avg_total_served = np.mean(served_blocks_ts)
    avg_blocks_curt = np.mean(curt_blocks_ts)
    
    # Calculate the extra MW unlocked strictly by the capacity blocks
    avg_extra_blocks_served = avg_total_served - avg_base_served

    # =========================================================================
    # 4. PRINT CONSOLE REPORT
    # =========================================================================
    print("\n" + "="*60)
    print(" FIRM ENERGY SERVED & CURTAILMENT REPORT (YEAR 1) ".center(60))
    print("="*60)
    
    print("SCENARIO 1: PURE BASELOAD")
    print(f"  Firm Baseload Served : {avg_base_served:>8.2f} MW Average")
    print(f"  Curtailed (Wasted)   : {avg_base_curt:>8.2f} MW Average")
    
    print("\nSCENARIO 2: CAPACITY BLOCKS")
    print(f"  Firm Baseload Served : {avg_base_served:>8.2f} MW Average")
    print(f"  Capacity Blks Served : {avg_extra_blocks_served:>8.2f} MW Average")
    print(f"  Total Firm Served    : {avg_total_served:>8.2f} MW Average")
    print(f"  Curtailed (Wasted)   : {avg_blocks_curt:>8.2f} MW Average")
    
    print("\nIMPACT OF WORKLOAD STACKING:")
    print(f"  Extra Firm Capacity Unlocked : +{avg_extra_blocks_served:.2f} MW")
    print(f"  Reduction in Curtailment     : -{(avg_base_curt - avg_blocks_curt):.2f} MW")
    print("="*60)

    # =========================================================================
    # 5. GENERATE PLOT
    # =========================================================================
    plot_baseload_vs_blocks(avg_base_served, avg_base_curt, avg_total_served, avg_blocks_curt)

if __name__ == "__main__":
    run_evaluation()