# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:31:05 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 3.7 - Surgical Hardware Extraction
Runs specified optimal workload combinations through HyDesign to extract the 
true physical hardware requirements (Min, Max, Avg) for each tier.
"""

"""
Section 3.7 - Surgical Hardware Extraction
Runs specified optimal workload combinations through HyDesign to extract the 
true physical hardware requirements (Min, Max, Avg) for each tier.
"""

import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# --- HYDESIGN EXPLICIT IMPORTS & PATHS ---
# =============================================================================
# 1. The exact paths to your script folder and the HyDesign package root
current_dir = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
thesis_dir = os.path.abspath(os.path.join(current_dir, '..'))

# 2. The critical fix: The folder that CONTAINS the inner 'hydesign' package
hydesign_sys_path = r"C:\Users\thijs\Downloads\hydesign"

# 3. Force Python to check this folder first when looking for modules
if hydesign_sys_path not in sys.path:
    sys.path.insert(0, hydesign_sys_path)

# 4. Now the import will resolve perfectly
from hydesign.assembly.hpp_assembly_tierb2_thijs_3_3_26 import hpp_model_constant_output_offgrid as hpp_model

# 1. SPECIFIED WORKLOAD COMBINATIONS
# =============================================================================
# Format: (IT_Capacity, Tier_A, Tier_B1, Tier_B2)
target_mixes = [
    (16.0, 5.0, 6.0, 4.0),
    (20.0, 8.0, 0.0, 11.0),
    (30.0, 8.0, 0.0, 17.0),
    (40.0, 5.0, 6.0, 17.0),
    (50.0, 5.0, 6.0, 20.0),
    (75.0, 5.0, 6.0, 25.0),
    (100.0, 5.0, 6.0, 26.0)
]

# =============================================================================
# 2. CONFIGURATION HELPERS
# =============================================================================
def configure_parameters(thesis_dir):
    par_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars.yml')
    with open(par_fn, 'r') as f:
        sim_pars = yaml.safe_load(f)

    sim_pars['G_MW'] = 0
    sim_pars['battery_charge_efficiency'] = float(np.sqrt(0.86))

    temp_fn = os.path.join(thesis_dir, 'inputs', 'hpp_pars_offgrid_surgical_temp.yml')
    with open(temp_fn, 'w') as f:
        yaml.dump(sim_pars, f)
    return temp_fn

# =============================================================================
# 3. MAIN SURGICAL RUN
# =============================================================================
def run_surgical_extraction():
    # Base configuration
    fixed_design = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10]
    N_life = 25 * 8760
    site_name = 'Denmark_good_solar'
    
    examples_sites = pd.read_csv(os.path.join(thesis_dir, '..', 'examples_sites.csv'), sep=';')
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    weather_fn = os.path.join(thesis_dir, '..', ex_site['input_ts_fn'].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    os.environ['REWARD_C2'] = '-0.5' 
    
    results = []

    print("\n" + "=" * 80)
    print("🚀 STARTING SURGICAL HARDWARE EXTRACTION".center(80))
    print("=" * 80)

    for cap_mw, a_mw, b1_mw, b2_mw in target_mixes:
        print(f"\n--- Simulating {cap_mw} MW Facility (A:{a_mw}, B1:{b1_mw}, B2:{b2_mw}) ---")
        
        # Build workload signals for HyDesign
        t_a_ts = np.full(N_life, a_mw)
        t_b1_ts = np.full(N_life, b1_mw * 24.0)  # Daily target
        t_b2_ts = np.full(N_life, b2_mw * 168.0) # Weekly target
        
        total_load_for_ems = np.full(N_life, cap_mw)

        # Initialize Model
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

        # --- EXTRACT TIME SERIES (Year 1: 8760 hours) ---
        try:
            served_a = prob.get_val('ems.Served_A')[:8760]  
            served_b1 = prob.get_val('ems.Served_B')[:8760]
            served_b2 = prob.get_val('ems.Served_B2')[:8760]
            served_c = prob.get_val('ems.Served_C2')[:8760]
        except KeyError as e:
            print(f"❌ ERROR: Missing OpenMDAO variable: {e}. Check your ems outputs!")
            return

        # --- CALCULATE STATISTICS ---
        stats = {
            "Facility_MW": cap_mw,
            "Contracted_A_MW": a_mw,
            "Contracted_B1_MW": b1_mw,
            "Contracted_B2_MW": b2_mw,
            
            "Avg_A_MW": np.mean(served_a),
            "Max_A_MW": np.max(served_a),
            "Min_A_MW": np.min(served_a),
            
            "Avg_B1_MW": np.mean(served_b1),
            "Max_B1_MW": np.max(served_b1),
            "Min_B1_MW": np.min(served_b1),
            
            "Avg_B2_MW": np.mean(served_b2),
            "Max_B2_MW": np.max(served_b2),
            "Min_B2_MW": np.min(served_b2),
            
            "Avg_C_MW": np.mean(served_c),
            "Max_C_MW": np.max(served_c),
            "Min_C_MW": np.min(served_c),
        }
        
        stats["True_Physical_Peak_Required"] = stats["Max_A_MW"] + stats["Max_B1_MW"] + stats["Max_B2_MW"] + stats["Max_C_MW"]
        results.append(stats)
        
        # --- PRINT RESULTS TO CONSOLE ---
        print(f"   [Tier A]  Avg: {stats['Avg_A_MW']:05.2f} MW | Peak: {stats['Max_A_MW']:05.2f} MW | Min: {stats['Min_A_MW']:05.2f} MW")
        print(f"   [Tier B1] Avg: {stats['Avg_B1_MW']:05.2f} MW | Peak: {stats['Max_B1_MW']:05.2f} MW | Min: {stats['Min_B1_MW']:05.2f} MW")
        print(f"   [Tier B2] Avg: {stats['Avg_B2_MW']:05.2f} MW | Peak: {stats['Max_B2_MW']:05.2f} MW | Min: {stats['Min_B2_MW']:05.2f} MW")
        print(f"   [Tier C]  Avg: {stats['Avg_C_MW']:05.2f} MW | Peak: {stats['Max_C_MW']:05.2f} MW | Min: {stats['Min_C_MW']:05.2f} MW")
        print("-" * 60)
        print(f"   ⚠️ SUM OF PEAKS: {stats['True_Physical_Peak_Required']:.1f} MW (Facility physical constraint: {cap_mw} MW)")

    # =============================================================================
    # 4. EXPORT TO CSV
    # =============================================================================
    df_results = pd.DataFrame(results)
    
    # Reorder columns logically
    cols = [
        'Facility_MW', 'Contracted_A_MW', 'Contracted_B1_MW', 'Contracted_B2_MW', 
        'True_Physical_Peak_Required',
        'Avg_A_MW', 'Max_A_MW', 'Min_A_MW',
        'Avg_B1_MW', 'Max_B1_MW', 'Min_B1_MW',
        'Avg_B2_MW', 'Max_B2_MW', 'Min_B2_MW',
        'Avg_C_MW', 'Max_C_MW', 'Min_C_MW'
    ]
    df_results = df_results[cols]
    
    csv_fn = os.path.join(current_dir, 'Surgical_Hardware_Peaks.csv')
    df_results.to_csv(csv_fn, index=False)
    
    print("\n" + "=" * 80)
    print(f"✅ Extraction Complete! CSV saved to: {csv_fn}")
    print("=" * 80)

if __name__ == "__main__":
    run_surgical_extraction()