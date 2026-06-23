# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:21:05 2026

@author: thijs
"""

# -*- coding: utf-8 -*-
"""
Section 2.X - Methodology: Queue FIFO Age Mechanics
Publication-Grade Conceptual Diagram explaining volume-to-age translation.
Refined for maximum readability, minimal text clutter, and clear visual hierarchy.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

def generate_queue_age_schematic():
    # --- 1. SETUP PARAMETERS ---
    T_deadline = 24         # SLA Deadline (Hours)
    queue_volume = 28       # Total blocks currently in queue
    overdue = queue_volume - T_deadline

    # --- 2. EDITORIAL PALETTE & TYPOGRAPHY ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    C_SAFE    = '#2b6cb0'   # Editorial Steel Blue
    C_OVERDUE = '#e28743'   # Muted Terracotta / Clay
    C_LIMIT   = '#c0392b'   # Striking Red for SLA Boundary
    C_TEXT    = '#2c3e50'   # Charcoal Navy
    C_MUTED   = '#718096'   # Soft Slate for secondary labels
    C_AXIS    = '#2d3748'   # Crisp dark spine grey

    # Balanced figure dimensions
    fig, ax = plt.subplots(figsize=(11.5, 3.5), dpi=300, facecolor='white')

    # --- 3. QUEUE BAR GENERATION (Discrete Hourly Blocks) ---
    bar_y = 0.26
    bar_h = 0.28
    block_width = 0.88      # Sharp, precise gaps between discrete blocks

    for i in range(queue_volume):
        # Age increases linearly from right (newest) to left (oldest)
        age = queue_volume - i - 1
        color = C_OVERDUE if age >= T_deadline else C_SAFE

        ax.add_patch(
            Rectangle(
                (i + (1 - block_width)/2, bar_y), block_width, bar_h,
                facecolor=color,
                edgecolor='white',
                linewidth=0.8,
                zorder=4
            )
        )

    # --- 4. STRUCTURAL SLA BOUNDARY ---
    deadline_x = queue_volume - T_deadline

    # Distinct, thick red dashed line for the SLA Limit
    ax.plot(
        [deadline_x, deadline_x], [0.15, 0.75],
        color=C_LIMIT,
        linestyle='--',
        linewidth=2.0,
        zorder=5
    )
    ax.text(deadline_x, 0.78, "SLA Limit (24h)", ha='center', va='bottom', color=C_LIMIT, fontsize=12, fontweight='bold')

    # --- 5. FLOW DIRECTIONAL ARROWS ---
    # Workload Inflow (Right)
    ax.add_patch(FancyArrowPatch(
        (queue_volume + 2.0, bar_y + bar_h / 2),
        (queue_volume + 0.2, bar_y + bar_h / 2),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=C_TEXT, zorder=5
    ))
    ax.text(queue_volume + 1.1, bar_y + bar_h + 0.05, "Inflow", 
            ha='center', va='bottom', color=C_TEXT, fontsize=12, fontweight='bold')

    # FIFO Service Output (Left)
    ax.add_patch(FancyArrowPatch(
        (-0.2, bar_y + bar_h / 2),
        (-2.0, bar_y + bar_h / 2),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=C_TEXT, zorder=5
    ))
    ax.text(-1.1, bar_y + bar_h + 0.05, "Served", 
            ha='center', va='bottom', color=C_TEXT, fontsize=12, fontweight='bold')

    # Unified Aging Vector Line (Top)
    ax.add_patch(FancyArrowPatch(
        (queue_volume, 0.95),
        (0, 0.95),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=C_MUTED, zorder=5
    ))
    ax.text(queue_volume / 2, 1.0, "Increasing Age in Queue", 
            ha='center', va='bottom', color=C_MUTED, fontsize=11, fontweight='bold')

    # --- 6. METRIC BRACKETS & METADATA ---
    # Tightly aligned age text inside the compressed lower gap
    ax.text(queue_volume - 0.5, bar_y - 0.05, "Age: 0 h", ha='right', va='top', color=C_MUTED, fontsize=10)
    ax.text(0.5, bar_y - 0.05, f"Age: {queue_volume} h", ha='left', va='top', color=C_MUTED, fontsize=10)

    # --- 7. AXIS & LEGEND FORMATTING ---
    # Compressed data range to pull the bottom line tightly against the blocks
    ax.set_xlim(-2.5, queue_volume + 2.5)
    ax.set_ylim(0.0, 1.25)
    ax.set_yticks([])

    # Clean bottom axis setup with larger, descriptive labels
    ax.set_xticks([0, deadline_x, queue_volume])
    ax.set_xticklabels([
        "Oldest\n(Front of Queue)", 
        "Deadline\nThreshold", 
        "Newest\n(Back of Queue)"
    ], fontsize=11, fontweight='bold', color=C_TEXT)

    # Minimize interface lines to maximize data emphasis
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color(C_AXIS)
    ax.tick_params(axis='x', length=5, width=1.5, color=C_AXIS)

    # Editorial Legend Layout - simplified and larger
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=C_SAFE, edgecolor='none', label='Within Deadline'),
        Rectangle((0, 0), 1, 1, facecolor=C_OVERDUE, edgecolor='none', label='Deadline Violated')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 1.25), 
              frameon=False, fontsize=11, ncol=2)

    plt.tight_layout()
    
    # Save target outputs
    save_path = 'Queue_FIFO_SLA_Mechanics.svg'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Layout refined. Schematic saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_queue_age_schematic()