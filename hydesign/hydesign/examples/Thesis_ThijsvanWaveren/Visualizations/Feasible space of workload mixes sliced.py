# -*- coding: utf-8 -*-
"""
Feasible workload combinations for selected Tier A capacities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D

# =============================================================================
# 1. SETUP & DATA
# =============================================================================

BASE_FOLDER = r"C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren\scripts"
IT_CAPACITY = 16.0
RELIABILITY_TARGET = 99.9

FILE_NAME = f"Feasible_3D_Sweep_Results_99.9pct_IT{IT_CAPACITY:.1f}.csv"
file_path = os.path.join(BASE_FOLDER, FILE_NAME)

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    if "Reliability" in df.columns:
        df = df[df["Reliability"] >= RELIABILITY_TARGET].copy()

    # For each B1/B2 combination, store the maximum Tier A that remains feasible.
    heatmap_df = (
        df.groupby(["Tier_B1_MW", "Tier_B2_MW"])["Tier_A_MW"]
        .max()
        .reset_index()
    )

    x = heatmap_df["Tier_B1_MW"].to_numpy()
    y = heatmap_df["Tier_B2_MW"].to_numpy()
    z = heatmap_df["Tier_A_MW"].to_numpy()

else:
    print("CSV not found. Generating synthetic fallback data.")

    x, y, z = [], [], []
    for b1 in np.arange(0, 16.5, 0.5):
        for b2 in np.arange(0, 16.5, 0.5):
            if b1 + b2 <= IT_CAPACITY:
                effective_use = 1.45 * b1 + 1.00 * b2
                max_a = max(0, IT_CAPACITY - effective_use)
                x.append(b1)
                y.append(b2)
                z.append(max_a)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

# =============================================================================
# 2. STYLE
# =============================================================================

# Distinct, professional Purple theme
C_FRONTIER = "#8e44ad"   # Strong Amethyst
C_LIMIT = "#8c8c8c"      # Subtle Gray for the physical limit

A_SLICES = [0, 4, 8]

# Create a reusable subtle white halo for text overlay readability
halo_effect = [path_effects.withStroke(linewidth=3.5, foreground='white', alpha=0.9)]

# =============================================================================
# 3. HELPER FUNCTION
# =============================================================================

def get_frontier(a_val):
    """
    Return feasible points and frontier for a fixed Tier A capacity.
    Frontier is defined as maximum feasible B2 for each B1.
    """
    mask = z >= a_val
    x_feas = x[mask]
    y_feas = y[mask]

    if len(x_feas) == 0:
        return x_feas, y_feas, np.array([]), np.array([])

    frontier = (
        pd.DataFrame({"B1": x_feas, "B2": y_feas})
        .groupby("B1")["B2"]
        .max()
        .reset_index()
        .sort_values("B1")
    )

    fx = frontier["B1"].to_numpy()
    fy = frontier["B2"].to_numpy()

    return x_feas, y_feas, fx, fy

# =============================================================================
# 4. PLOT
# =============================================================================

# Adjusted figsize for standard LaTeX text width (1x3 grid)
fig, axes = plt.subplots(
    1, 3,
    figsize=(12, 4),
    sharey=True,
    facecolor="white"
)

for ax, a_val in zip(axes, A_SLICES):

    remaining_it = IT_CAPACITY - a_val
    x_feas, y_feas, fx, fy = get_frontier(a_val)

    # -------------------------------------------------------------------------
    # Physical IT capacity limit
    # -------------------------------------------------------------------------
    ax.plot(
        [0, remaining_it],
        [remaining_it, 0],
        color=C_LIMIT,
        linestyle='--',
        linewidth=1.8,
        zorder=3
    )

    # -------------------------------------------------------------------------
    # Feasible combinations and frontier
    # -------------------------------------------------------------------------
    if len(fx) > 0:
        # Case 1: 2D feasible region (Draws polygon and text)
        if len(fx) > 1:
            # Fill feasible region under frontier
            fx_fill = np.r_[fx[0], fx, fx[-1]]
            fy_fill = np.r_[0, fy, 0]

            ax.fill_between(
                fx_fill,
                0,
                fy_fill,
                color=C_FRONTIER,
                alpha=0.2, 
                zorder=1
            )

            # Extend the frontier line to drop vertically down to the x-axis
            fx_line = np.append(fx, fx[-1])
            fy_line = np.append(fy, 0)

            # Frontier line
            ax.plot(
                fx_line,
                fy_line,
                color=C_FRONTIER,
                linewidth=3,
                zorder=5
            )

            # Centroid geometry for text placement (1/3 of max bounds fits the triangular space perfectly)
            cx = np.max(fx) * 0.33
            cy = np.max(fy) * 0.33

            # Split text onto two lines to keep it horizontally compact and add white halo
            ax.text(
                cx, cy,
                "FEASIBLE\nREGION",
                fontsize=9.5,
                fontweight='bold',
                color=C_FRONTIER,
                ha='center',
                va='center',
                zorder=6,
                path_effects=halo_effect
            )

        # Case 2: 1D feasible set (typically Tier A = 8 MW, just a vertical line)
        else:
            b1_val = fx[0]
            b2_max = fy[0]

            # Thick vertical frontier to show B2 remains feasible at B1 = 0
            ax.vlines(
                b1_val,
                0,
                b2_max,
                color=C_FRONTIER,
                linewidth=3,
                zorder=5
            )

    # -------------------------------------------------------------------------
    # Axes styling & Titles
    # -------------------------------------------------------------------------
    ax.set_title(f"Tier A = {a_val:.0f} MW", fontsize=12, fontweight='bold', color='#333333', pad=10)
    
    ax.set_xlim(-0.2, 16.5)
    ax.set_ylim(-0.2, 16.5)
    ax.set_xticks(np.arange(0, 17, 4))
    ax.set_yticks(np.arange(0, 17, 4))

    ax.grid(True, linestyle='--', alpha=0.3, color='#b0b0b0', zorder=0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color('#444444')
    ax.spines["bottom"].set_color('#444444')
    ax.tick_params(axis="both", colors="#333333")

    # Consistent Axis Labels
    ax.set_xlabel("Tier B1 Capacity (MW)", fontsize=11, fontweight='bold', color='#444444')

# Set Y-label only on the far left plot due to sharey=True
axes[0].set_ylabel("Tier B2 Capacity (MW)", fontsize=11, fontweight='bold', color='#444444')

# =============================================================================
# 5. LEGEND
# =============================================================================

# Simplified to match the clean line style of the cannibalization plots
legend_handles = [
    Line2D(
        [0], [0],
        color=C_FRONTIER,
        lw=3,
        label="Max Feasible Tier B1 / B2 Combination"
    ),
    Line2D(
        [0], [0],
        color=C_LIMIT,
        lw=1.8,
        linestyle='--',
        label="Remaining Hardware Limit (16 MW - Tier A)"
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=2,
    frameon=False,
    fontsize=11
)

plt.tight_layout()

# =============================================================================
# 6. EXPORT
# =============================================================================

save_svg = os.path.join(BASE_FOLDER, "Thesis_Feasible_Workload_Combinations_Clean.svg")
plt.savefig(save_svg, bbox_inches="tight")
print(f"Saved SVG: {save_svg}")


plt.show()