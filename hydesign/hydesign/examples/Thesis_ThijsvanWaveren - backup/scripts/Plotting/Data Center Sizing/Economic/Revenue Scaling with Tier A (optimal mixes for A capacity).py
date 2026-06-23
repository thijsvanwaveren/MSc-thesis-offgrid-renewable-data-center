# -*- coding: utf-8 -*-
"""
Section 3.3 / 3.4 - Economic Optimization (Twin-Bar Trade-Off Plot)
Calculates Tier C capacity directly from annual energy (Energy / 8.76).
Includes dynamic Y-axis scaling and solid number badges.

Updates:
- Swapped visual hierarchy: Workload Capacity (left) is now wide and solid.
- Revenue (right) is now narrow, hatched, and semi-transparent.
- Updated labels to "Workload Capacity (MW)".
- FIXED OVERLAP: Added mathematical scaling so the Revenue text always sits visually higher than both bars.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# =============================================================================
# 1. SETUP & PARAMETERS
# =============================================================================
BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
# os.chdir(BASE_FOLDER) # Uncomment if needed

CAPACITY = 16.0 

# Prices (Millions of € per GWh)
PRICE_A  = 4.0  
PRICE_B1 = 2.8  
PRICE_B2 = 1.6  
PRICE_C  = 0.4  

# Thesis Palette
C_A = '#08306b'      # Deep Navy
C_B1 = '#2879b9'     # Strong Blue
C_B2 = '#73b3d8'     # Soft Blue
C_C = '#c8ddf0'      # Pale Blue

HATCH_REVENUE = '//' 
OPACITY_REVENUE = 0.6 

# =============================================================================
# 2. DATA EXTRACTION & OPTIMIZATION
# =============================================================================
file_path_1 = os.path.join(BASE_FOLDER, f"Feasible_3D_Sweep_Results_99.9pct_IT{CAPACITY}.csv")
file_path_2 = os.path.join(BASE_FOLDER, f"Feasible_3D_Sweep_Results_99.9pct_IT{CAPACITY:.1f}.csv")
file_path = file_path_1 if os.path.exists(file_path_1) else file_path_2

if not os.path.exists(file_path):
    print("⚠️ CSV not found. Generating synthetic data...")
    rows = []
    for a in range(9):
        for b1 in range(15):
            for b2 in range(16):
                if a*1.5 + b1*1.1 + b2*1.3 <= CAPACITY:
                    rows.append({
                        'Tier_A_MW': a, 'Tier_B1_MW': b1, 'Tier_B2_MW': b2, 
                        'Energy_A_Annual_GWh': a*8.76*0.9, 
                        'Energy_B1_Annual_GWh': b1*8.76*0.5, 
                        'Energy_B2_Annual_GWh': b2*8.76*0.3
                    })
    df = pd.DataFrame(rows)
    df['Total_Delivered_Annual_GWh'] = df['Energy_A_Annual_GWh'] + df['Energy_B1_Annual_GWh'] + df['Energy_B2_Annual_GWh'] + (CAPACITY*0.1*8.76)
else:
    df = pd.read_csv(file_path)

# Calculate Tier C Energy (Slack Capacity)
if 'Energy_C_Annual_GWh' not in df.columns:
    e_a = df.get('Energy_A_Annual_GWh', 0)
    e_b1 = df.get('Energy_B1_Annual_GWh', 0)
    e_b2 = df.get('Energy_B2_Annual_GWh', 0)
    e_total_del = df.get('Total_Delivered_Annual_GWh', e_a + e_b1 + e_b2)
    df['Energy_C_Annual_GWh'] = np.maximum(0, e_total_del - (e_a + e_b1 + e_b2))

# Calculate Revenues
df['Rev_A'] = df['Energy_A_Annual_GWh'] * PRICE_A
df['Rev_B1'] = df['Energy_B1_Annual_GWh'] * PRICE_B1
df['Rev_B2'] = df['Energy_B2_Annual_GWh'] * PRICE_B2
df['Rev_C'] = df['Energy_C_Annual_GWh'] * PRICE_C
df['Total_Rev'] = df['Rev_A'] + df['Rev_B1'] + df['Rev_B2'] + df['Rev_C']

# Extract the Optimal Row for each discrete step of Tier A
optima_data = []
for a_step in range(9): 
    subset = df[df['Tier_A_MW'] == a_step]
    if not subset.empty:
        best_row = subset.loc[subset['Total_Rev'].idxmax()]
        
        # Calculate continuous equivalent capacity for Tier C
        tier_c_mw = best_row['Energy_C_Annual_GWh'] / 8.76 if 'Energy_C_Annual_GWh' in best_row else 0
        
        optima_data.append({
            'Tier_A': a_step,
            'MW_A': best_row['Tier_A_MW'], 'MW_B1': best_row['Tier_B1_MW'], 'MW_B2': best_row['Tier_B2_MW'], 'MW_C': tier_c_mw,
            'Rev_A': best_row['Rev_A'], 'Rev_B1': best_row['Rev_B1'], 'Rev_B2': best_row['Rev_B2'], 'Rev_C': best_row['Rev_C'],
            'Total_Rev': best_row['Total_Rev']
        })

df_opt = pd.DataFrame(optima_data)

# =============================================================================
# 3. VISUALIZATION (TWIN-BAR PLOT)
# =============================================================================
fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='white')
ax2 = ax1.twinx()

x = np.array(df_opt['Tier_A'])
mw_width = 0.4     # Wider, solid bars for Workload Capacity
rev_width = 0.25   # Narrower, hatched bars for Revenue
offset = 0.33 / 2

# --- Plot Left Bars: Physical Capacity (MW) on ax1 (SOLID) ---
bottom_mw = np.zeros(len(x))
mw_cols = ['MW_A', 'MW_B1', 'MW_B2', 'MW_C']
colors = [C_A, C_B1, C_B2, C_C]

for col, color in zip(mw_cols, colors):
    # Plot the fully filled, opaque bar
    ax1.bar(x - offset, df_opt[col], mw_width, color=color, bottom=bottom_mw, 
            edgecolor='white', linewidth=0.5, alpha=1.0)
    
    # Text contrast: Use crisp white text for Navy (A) and Strong Blue (B1) since they are now solid
    text_color = 'white' if color in [C_A, C_B1] else '#333333'

    # Add exact MW integer labels inside the bars
    for i, val in enumerate(df_opt[col]):
        if val >= 1.0: 
            ax1.text(x[i] - offset, bottom_mw[i] + (val / 2), f"{val:.0f}", 
                     ha='center', va='center', color=text_color, fontsize=9, 
                     fontweight='bold', zorder=5)
    
    bottom_mw += df_opt[col]

# --- Plot Right Bars: Financial Revenue (€) on ax2 (HATCHED & TRANSPARENT) ---
bottom_rev = np.zeros(len(x))
rev_cols = ['Rev_A', 'Rev_B1', 'Rev_B2', 'Rev_C']

for col, color in zip(rev_cols, colors):
    # Plot the hatched, semi-transparent bar
    ax2.bar(x + offset, df_opt[col], rev_width, color=color, bottom=bottom_rev, 
            edgecolor='white', hatch=HATCH_REVENUE, linewidth=0.5, alpha=OPACITY_REVENUE)
    bottom_rev += df_opt[col]

# --- Set Axis Limits Early (Required for Dynamic Label Placement) ---
# Give a generous top buffer (1.2 multiplier) for text clarity
max_physical_mw = df_opt[['MW_A', 'MW_B1', 'MW_B2', 'MW_C']].sum(axis=1).max()
ax1_ymax = max_physical_mw * 1.2
ax1.set_ylim(0, ax1_ymax) 

max_rev = df_opt['Total_Rev'].max()
ax2_ymax = max_rev * 1.2
ax2.set_ylim(0, ax2_ymax) 

# Add Total Revenue Labels dynamically so they clear BOTH bars
for i, val in enumerate(df_opt['Total_Rev']):
    # Get the height of the adjacent left bar
    left_bar_height = df_opt.loc[i, ['MW_A', 'MW_B1', 'MW_B2', 'MW_C']].sum()
    
    # Map the left bar's physical height to ax2's financial scale
    left_bar_visual_y = (left_bar_height / ax1_ymax) * ax2_ymax
    
    # Anchor the text above whichever bar is visually taller
    text_y = max(val, left_bar_visual_y) + (ax2_ymax * 0.02)
    
    ax2.text(x[i] + offset, text_y, f"€{val:.0f}M", 
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

# --- Formatting & Aesthetics ---
ax1.set_xlabel("Tier A Capacity (MW)", fontsize=12, fontweight='bold', color='#444444', labelpad=10)
ax1.set_ylabel("Optimal Workload Mix Composition (MW)", fontsize=12, fontweight='bold', color='#333333')
ax2.set_ylabel("Annual Gross Revenue (Millions €)", fontsize=12, fontweight='bold', color='#333333')

ax1.set_xticks(x)
ax1.set_xticklabels([f"{int(val)}" for val in x], fontsize=11)

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.grid(axis='y', linestyle='-', alpha=0.3, color='#e0e0e0', zorder=0)

# --- Custom Dual-Legend ---
legend_elements = [
    mpatches.Patch(facecolor=C_A, label='Tier A'),
    mpatches.Patch(facecolor=C_B1, label='Tier B1'),
    mpatches.Patch(facecolor=C_B2, label='Tier B2'),
    mpatches.Patch(facecolor=C_C, label='Tier C'),
    mpatches.Patch(facecolor='#555555', edgecolor='white', label='Left Bar: Workload Capacity (MW)'),
    mpatches.Patch(facecolor='#a6acaf', edgecolor='white', hatch=HATCH_REVENUE, alpha=OPACITY_REVENUE, label='Right Bar: Gross Revenue (€)'),
]

ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, 
           frameon=False, fontsize=11)

plt.tight_layout()

# Save & Show
save_path = os.path.join(BASE_FOLDER, 'TwinBar_Economic_Optimum_16MW.svg')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print(f"✅ Plot calculating Tier C from Energy saved to: {save_path}")
print("=" * 60)
plt.show()