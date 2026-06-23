# -*- coding: utf-8 -*-
"""
Section 3.2 - Isolated Workload Reliability Limits
(Upgraded with Academic-Consulting Formatting & Clean Legend)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# =============================================================================
# 1. SETUP & FILE PATHS
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

file_a = os.path.join(current_dir, 'data_tier_a_rel_1MW.csv')
file_b1 = os.path.join(current_dir, 'data_tier_b1_rel_1MW.csv')
file_b2 = os.path.join(current_dir, 'data_tier_b2_rel_1MW.csv')

# Academic-Consulting Color Palette
C_FIRM = '#113b5e'      # Deep Navy (Tier A)
C_DAILY = '#e67e22'     # Strong Orange (Tier B1)
C_WEEKLY = '#27ae60'    # Strong Green (Tier B2)
C_BREACH = '#c0392b'    # Deep Red (SLA Failure)

# =============================================================================
# 2. DATA PROCESSING FUNCTIONS
# =============================================================================
def pad_to_zero_mw(df, mw_col, rel_col):
    min_simulated_mw = int(df[mw_col].min())
    if min_simulated_mw > 0:
        pad_mws = np.arange(0, min_simulated_mw)
        pad_df = pd.DataFrame({mw_col: pad_mws, rel_col: 100.0})
        return pd.concat([pad_df, df], ignore_index=True)
    return df

def find_actual_optimum(df, mw_col, rel_col, target=99.9):
    reliable_rows = df[df[rel_col] >= target]
    if not reliable_rows.empty:
        return reliable_rows[mw_col].max()
    return 0

# =============================================================================
# 3. LOAD AND PREPARE DATA
# =============================================================================
if not (os.path.exists(file_a) and os.path.exists(file_b1) and os.path.exists(file_b2)):
    print("Warning: CSV files not found. Using synthetic data for plotting demonstration.")
    df_a = pd.DataFrame({'Tier_A_MW': np.arange(0, 51), 'Reliability_Time_A': [100]*9 + [99.5 - (x-8)*1.1 for x in range(9, 51)]})
    df_b1 = pd.DataFrame({'Tier_B_MW': np.arange(0, 51), 'Reliability_Deadline_B': [100]*16 + [99.5 - (x-15)*1.0 for x in range(16, 51)]})
    df_b2 = pd.DataFrame({'Tier_B2_MW': np.arange(0, 51), 'Reliability_Deadline_B2': [100]*40 + [99.5 - (x-39)*0.9 for x in range(40, 51)]})
else:
    df_a = pd.read_csv(file_a)
    df_b1 = pd.read_csv(file_b1)
    df_b2 = pd.read_csv(file_b2)

df_a = pad_to_zero_mw(df_a, 'Tier_A_MW', 'Reliability_Time_A')
df_b1 = pad_to_zero_mw(df_b1, 'Tier_B_MW', 'Reliability_Deadline_B')
df_b2 = pad_to_zero_mw(df_b2, 'Tier_B2_MW', 'Reliability_Deadline_B2')

max_a = find_actual_optimum(df_a, 'Tier_A_MW', 'Reliability_Time_A')
max_b1 = find_actual_optimum(df_b1, 'Tier_B_MW', 'Reliability_Deadline_B')
max_b2 = find_actual_optimum(df_b2, 'Tier_B2_MW', 'Reliability_Deadline_B2')

# =============================================================================
# 4. GENERATE VISUALIZATION
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6.5), facecolor='white')
halo = [path_effects.withStroke(linewidth=3, foreground="white", alpha=0.9)]

# 1. Plot the "SLA Breach Zone" (Visual storytelling foundation)
ax.axhline(99.9, color=C_BREACH, linestyle='--', linewidth=1.5, zorder=1)
ax.fill_between([-5, 60], 0, 99.9, color=C_BREACH, alpha=0.04, zorder=0)
ax.text(39.3, 99.9 + 0.5, "Reliability Target (99.9%)", color=C_BREACH, fontsize=12.8, fontweight='bold', va='bottom', zorder=5)

# 2. Plot the Curves
ax.plot(df_a['Tier_A_MW'], df_a['Reliability_Time_A'], color=C_FIRM, linewidth=2.5, label='Tier A (Firm Load)', zorder=3)
ax.plot(df_b1['Tier_B_MW'], df_b1['Reliability_Deadline_B'], color=C_DAILY, linewidth=2.5, label='Tier B1 (Daily Flexible)', zorder=3)
ax.plot(df_b2['Tier_B2_MW'], df_b2['Reliability_Deadline_B2'], color=C_WEEKLY, linewidth=2.5, label='Tier B2 (Weekly Flexible)', zorder=3)

# 3. Add Explicit "Failure Point" Anchors and Drop Lines
drop_points = [
    (max_a, C_FIRM, f'{int(max_a)} MW\n(Firm Only)'),
    (max_b1, C_DAILY, f'{int(max_b1)} MW\n(Daily Only)'),
    (max_b2, C_WEEKLY, f'{int(max_b2)} MW\n(Weekly Only)')
]

for x_val, color, text in drop_points:
    # Anchor Point
    ax.scatter(x_val, 99.9, facecolors='white', edgecolors=color, s=80, linewidth=2, zorder=5)
    
    # Sleek dashed drop line
    ax.vlines(x=x_val, ymin=50, ymax=99.9, color=color, linestyle=':', linewidth=1.5, alpha=0.8, zorder=2)
    
    # Label shifted slightly to the right (x_val + 0.8) to prevent line overlap
    ax.text(x_val + 0.8, 52, text, color=color, fontsize=10, fontweight='bold', ha='left', va='bottom', path_effects=halo, zorder=5)

# 4. Formatting & Intuitive Axes Names
#ax.set_title('SLA Fulfillment vs. Workload Capacity', fontsize=15, fontweight='bold', pad=20, color='#333333')
ax.set_ylabel('Reliability Level (%)', fontsize=12, fontweight='bold', color='#444444')
ax.set_xlabel('Workload Capacity (MW)', fontsize=12, fontweight='bold', color='#444444')

# Limits
ax.set_ylim(50, 102) 
ax.set_xlim(0, 52) 
ax.set_yticks(np.arange(50, 101, 10))
ax.tick_params(axis='both', colors='#333333', labelsize=11)

# Minimalist Grid and Structural Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('#444444')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('#444444')
ax.grid(axis='y', linestyle='-', alpha=0.3, color='#b0b0b0', zorder=0)
ax.grid(axis='x', visible=False)

# 5. External Legend (Moved below the chart to free up the plot area)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, 
          fontsize=11, frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9, borderpad=1)

# Adjust layout to make room for the legend below
plt.subplots_adjust(bottom=0.2)

# Save Plot
plot_fn = os.path.join(current_dir, 'Isolated_Capacity_Reliability_Academic.svg')
plt.savefig(plot_fn, dpi=300, bbox_inches='tight')
print(f"✅ Final Plot saved to: {plot_fn}")

plt.show()