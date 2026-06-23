# -*- coding: utf-8 -*-
"""
Section 2.X - Methodology: Queueing Dynamics
Publication-Grade Editorial Visualization of Stock-and-Flow Queueing Logic.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os

def generate_queue_visualization():
    # --- 1. GLOBAL EDITORIAL TYPOGRAPHY & STYLE ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['text.color'] = '#2c3e50'
    plt.rcParams['axes.labelcolor'] = '#2c3e50'
    plt.rcParams['xtick.color'] = '#555555'
    plt.rcParams['ytick.color'] = '#555555'

    # --- 2. DATA SETUP ---
    T = 73
    t = np.arange(T)
    
    A_B = np.ones(T) * 1.0  # Constant 1 MW Arrival
    
    # Power allocation profile
    P_B = np.zeros(T)
    P_B[8:18] = [0.5, 1.2, 2.0, 2.8, 3.0, 3.0, 2.8, 2.0, 1.2, 0.5]
    P_B[52:68] = [0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.0, 4.5, 3.5, 2.5, 1.5, 0.5, 0.2, 0.1, 0.1, 0]

    # Queue state calculation
    Q_B = np.zeros(T)
    SLA_LIMIT = 24.0
    for i in range(1, T):
        Q_B[i] = max(0, Q_B[i-1] + A_B[i] - P_B[i])

    # --- 3. PREMIUM EDITORIAL PALETTE ---
    C_ARRIVE   = '#5a6b7c'  # Muted Slate Gray
    C_SERVE    = '#1a365d'  # Premium Midnight Navy
    C_DEFICIT  = '#e28743'  # Muted Terracotta/Clay
    C_DEF_FILL = '#faebd7'  # Soft Warm Cream/Rose Tint
    C_SURPLUS  = '#2e7d32'  # Muted Forest Green
    C_SUR_FILL = '#e8f5e9'  # Very Pale Sage
    C_QUEUE    = '#2b6cb0'  # Deep Editorial Blue
    C_Q_FILL   = '#f7fafc'  # Clean Ice/Off-White Base
    C_VIOL_FILL= '#fde8e8'  # Ultra-soft Coral/Red tint for violations

    # --- 4. LAYOUT CREATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True, 
                                   gridspec_kw={'hspace': 0.18})
    
    # Text path effects for pristine line crossings
    halo = [path_effects.withStroke(linewidth=4, foreground="white", alpha=0.95)]
    
    # --- 5. TOP PANEL: POWER ALLOCATION ---
    ax1.plot(t, A_B, color=C_ARRIVE, linewidth=2.0, linestyle='--', zorder=4, label='Job Arrivals (1 MW Constant)')
    ax1.plot(t, P_B, color=C_SERVE, linewidth=2.5, zorder=5, label='Allocated IT Power (Served)')
    
    # Refined chronological fills
    ax1.fill_between(t, A_B, P_B, where=(A_B > P_B), interpolate=True, color=C_DEF_FILL, alpha=0.75, zorder=3, label='Power Deficit (Arrivals > Served)')
    ax1.fill_between(t, A_B, P_B, where=(P_B > A_B), interpolate=True, color=C_SUR_FILL, alpha=0.85, zorder=3, label='Power Surplus (Queue Clearing)')

    ax1.set_ylabel('Workload Power\n(MW)', fontsize=11, fontweight='bold', labelpad=15)
    ax1.set_ylim(0, 5.5)
    
    # Unified horizontal row legend above the plot box
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=10)

    # --- 6. BOTTOM PANEL: QUEUE LEVEL ACCUMULATION ---
    ax2.plot(t, Q_B, '-', color=C_QUEUE, linewidth=2.5, zorder=4, label='Queue Volume')
    
    # Symmetric, high-contrast zone masking
    ax2.fill_between(t, 0, np.minimum(Q_B, SLA_LIMIT), color=C_Q_FILL, alpha=1.0, zorder=2)
    ax2.fill_between(t, 0, np.minimum(Q_B, SLA_LIMIT), color=C_QUEUE, alpha=0.08, zorder=2, label='Safe Volume (Within Deadline)')
    ax2.fill_between(t, SLA_LIMIT, Q_B, where=(Q_B > SLA_LIMIT), color=C_VIOL_FILL, alpha=1.0, zorder=3)
    ax2.fill_between(t, SLA_LIMIT, Q_B, where=(Q_B > SLA_LIMIT), color=C_DEFICIT, alpha=0.35, zorder=3, label='SLA Violation (Overdue Workloads)')
    
    # Absolute SLA baseline threshold
    ax2.axhline(SLA_LIMIT, color=C_DEFICIT, linestyle=':', linewidth=1.8, zorder=5, label='24-Hour SLA Limit (24 MWh)')
    
    ax2.set_ylabel('Queue Volume\n(MWh)', fontsize=11, fontweight='bold', labelpad=12)
    ax2.set_xlabel('Time (Hours)', fontsize=11, fontweight='bold', labelpad=10)
    ax2.set_ylim(0, 39)
    ax2.set_xlim(0, 72)
    ax2.set_xticks(np.arange(0, 73, 6))
    
    ax2.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=False, fontsize=10)

    # --- 7. EXECUTIVE ANNOTATIONS (Sleek callouts & clean paths) ---
    
    # Panel 1 Callout: Pinpointing the Weather Interruption
    ax1.annotate('Power Deficit:\nQueueing Workloads', 
                 xy=(25, 0.3), xytext=(25, 2.0),
                 ha='center', va='bottom', zorder=10,
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color=C_DEFICIT, lw=1.2, mutation_scale=10), 
                 fontsize=9.5, fontweight='bold', color=C_DEFICIT, path_effects=halo)

    # Panel 2 Callout A: Permissible Delay Buffering
    ax2.annotate('Acceptable Queue Build-up\nWorkloads deferred due to low generation.\nQueue volume increases within SLA boundary.', 
                 xy=(31, 11), xytext=(15, 15), 
                 ha='center', va='center', zorder=10,
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.12", color=C_SERVE, lw=1.2, mutation_scale=10), 
                 fontsize=9.5, color='#4a5568', path_effects=halo)

    # Panel 2 Callout B: Regulatory Milestone Breach
    ax2.annotate('SLA Limit Breached\nQueue volume exceeds 24h capacity.\nWorkloads exceed deadlines.', 
                 xy=(54.5, 25.5), xytext=(64, 33.5), 
                 ha='center', va='center', zorder=10,
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.12", color=C_DEFICIT, lw=1.2, mutation_scale=10), 
                 fontsize=9.5, fontweight='bold', color=C_DEFICIT, path_effects=halo)

    # --- 8. GRAPHIC REFINEMENT & RUGGED ARCHITECTURE ---
    for ax in [ax1, ax2]:
        # Minimal clean baseline grid architecture
        ax.grid(axis='y', linestyle='-', alpha=0.25, color='#cbd5e0', zorder=0)
        ax.grid(axis='x', visible=False)
        
        # Clean corporate presentation: Strip upper/right/left clutter lines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['left'].set_color('#a0aec0')
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('#2d3748')
        
        ax.tick_params(axis='y', length=4, width=0.8, color='#a0aec0', labelsize=9.5)
        ax.tick_params(axis='x', length=5, width=1.5, color='#2d3748', labelsize=9.5)

    # Force continuous horizontal label symmetry across the dashboard stack
    fig.align_ylabels([ax1, ax2])

    plt.tight_layout()
    
    # Save target outputs
    save_path = 'Thesis_Queue_Dynamics_PublishingGrade.svg'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"✅ Saved clean editorial queue visualization to: {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_queue_visualization()