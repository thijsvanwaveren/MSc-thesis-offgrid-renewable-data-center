# -*- coding: utf-8 -*-
"""
Feasible workload combinations for selected Tier A capacities.
Clean thesis-style version:
- no hatched infeasible region
- subplot textboxes indicate fixed Tier A capacity
- feasible data points shown clearly
- thicker vertical frontier for Tier A = 8 MW
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "svg.fonttype": "path",
})

# Muted academic palette
C_FILL = "#C6DBEF"       # feasible region fill
C_POINT = "#4F81BD"      # feasible points
C_FRONTIER = "#08306B"   # feasibility frontier
C_LIMIT = "#8C8C8C"      # physical IT limit
C_AXIS = "#4D4D4D"
C_GRID = "#D9D9D9"
C_BOX = "#F5F5F5"

A_SLICES = [0, 4, 8]

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

fig, axes = plt.subplots(
    1, 3,
    figsize=(13.5, 4.2),
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
        linestyle=(0, (4, 3)),
        linewidth=1.6,
        zorder=3
    )

    # -------------------------------------------------------------------------
    # Feasible combinations and frontier
    # -------------------------------------------------------------------------
    if len(fx) > 0:

        # Case 1: 2D feasible region
        if len(fx) > 1:

            # Fill feasible region under frontier
            fx_fill = np.r_[fx[0], fx, fx[-1]]
            fy_fill = np.r_[0, fy, 0]

            ax.fill_between(
                fx_fill,
                0,
                fy_fill,
                color=C_FILL,
                alpha=0.75,
                zorder=1
            )

            # Frontier line
            ax.plot(
                fx,
                fy,
                color=C_FRONTIER,
                linewidth=2.5,
                marker="o",
                markersize=4.2,
                markerfacecolor=C_FRONTIER,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=5
            )

            point_size = 22
            point_edge_width = 0.55

        # Case 2: 1D feasible set, typically Tier A = 8 MW
        else:
            b1_val = fx[0]
            b2_max = fy[0]

            # Thick vertical frontier to show B2 remains feasible at B1 = 0
            ax.vlines(
                b1_val,
                0,
                b2_max,
                color=C_FRONTIER,
                linewidth=4.8,
                zorder=5
            )

            point_size = 46
            point_edge_width = 1.4

        # Feasible simulation points
        ax.scatter(
            x_feas,
            y_feas,
            s=point_size,
            color=C_POINT,
            edgecolor="white",
            linewidth=point_edge_width,
            alpha=0.95,
            zorder=6
        )

    # -------------------------------------------------------------------------
    # Tier A textbox above subplot
    # -------------------------------------------------------------------------
    ax.text(
        0.5,
        1.08,
        f"Tier A = {a_val:.0f} MW",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=C_AXIS,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="none",
            edgecolor="none",
            linewidth=0.8
        )
    )

    # -------------------------------------------------------------------------
    # Axes styling
    # -------------------------------------------------------------------------
    ax.set_xlim(-0.2, 16.2)
    ax.set_ylim(-0.2, 16.2)

    ax.set_xticks(np.arange(0, 17, 4))
    ax.set_yticks(np.arange(0, 17, 4))

    # Horizontal grid lines only
    ax.grid(axis="y", color=C_GRID, linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", visible=False)

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)

    ax.tick_params(axis="both", colors=C_AXIS)

    ax.set_xlabel("Tier B1 capacity [MW]")

axes[0].set_ylabel("Tier B2 capacity [MW]")

# =============================================================================
# 5. LEGEND
# =============================================================================

legend_handles = [
    mpatches.Patch(
        facecolor=C_FILL,
        edgecolor="none",
        alpha=0.75,
        label="Feasible region"
    ),
    Line2D(
        [0], [0],
        color=C_POINT,
        marker="o",
        linestyle="None",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Feasible simulation point"
    ),
    # Line2D(
    #     [0], [0],
    #     color=C_FRONTIER,
    #     lw=2.5,
    #     marker="o",
    #     markersize=4,
    #     markerfacecolor=C_FRONTIER,
    #     markeredgecolor="white",
    #     markeredgewidth=0.8,
    #     label="Feasibility frontier"
    # ),
    Line2D(
        [0], [0],
        color=C_LIMIT,
        lw=1.6,
        linestyle=(0, (4, 3)),
        label="Remaining IT capacity"
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.065),
    ncol=4,
    frameon=False,
    columnspacing=1.6,
    handlelength=2.2
)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# =============================================================================
# 6. EXPORT
# =============================================================================

save_svg = os.path.join(BASE_FOLDER, "Thesis_Feasible_Workload_Combinations_Clean.svg")
save_png = os.path.join(BASE_FOLDER, "Thesis_Feasible_Workload_Combinations_Clean.png")
save_pdf = os.path.join(BASE_FOLDER, "Thesis_Feasible_Workload_Combinations_Clean.pdf")

plt.savefig(save_svg, bbox_inches="tight")
plt.savefig(save_png, bbox_inches="tight", dpi=300)
plt.savefig(save_pdf, bbox_inches="tight")

print(f"Saved SVG: {save_svg}")
print(f"Saved PNG: {save_png}")
print(f"Saved PDF: {save_pdf}")

plt.show()