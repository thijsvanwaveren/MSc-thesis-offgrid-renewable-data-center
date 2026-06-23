import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def generate_methodology_diagram():
    # --- 1. SET UP CONTINUOUS WEATHER DATA ---
    hours_per_week = 168
    total_hours = hours_per_week * 3
    t = np.arange(total_hours)
    
    # Wind: Drops smoothly across the 3 weeks with stronger multi-day weather fronts
    wind = 14 + 8 * np.cos(t * np.pi / 504) + 2 * np.sin(t * 2 * np.pi / 168)
    
    # Solar: Large daily solar peaks
    solar = 22 * np.maximum(0, -np.cos(t * 2 * np.pi / 24))
    
    # Add a generation boost to Week 1 solar peaks to force visible curtailment
    solar[0:168] += 4 * np.maximum(0, -np.cos(t[0:168] * 2 * np.pi / 24))
    
    # Continuous Generation Envelope
    avail_power = wind + solar
    
    # --- 2. CALCULATE STRICT MINIMUMS FOR BASELOADS ---
    IT_CAPACITY = 45.0
    
    # Local minimums define the max 100% reliable guarantee for that specific week
    min_w1 = np.min(avail_power[0:168])
    min_w2 = np.min(avail_power[168:336])
    min_w3 = np.min(avail_power[336:504])
    
    YEARLY_BASELOAD = min_w3 # The lowest point of the year is our thick, unbroken foundation
    
    weekly_firm_total = np.zeros(total_hours)
    weekly_firm_total[0:168] = min_w1
    weekly_firm_total[168:336] = min_w2
    weekly_firm_total[336:504] = min_w3

    # --- 3. THE SCHEDULING ALGORITHM (Strict IT Boundaries) ---
    batch_load = np.zeros(total_hours)
    batch_avg_line = np.zeros(total_hours)
    spot_load = np.zeros(total_hours)
    curtailment = np.zeros(total_hours)
    
    BATCH_TARGET_MW = 7.0      
    MAX_BATCH_POWER = 30.0     
    
    for day in range(total_hours // 24):
        start = day * 24
        end = start + 24
        
        # Calculate available generation above the firm guarantees
        avail_flex = avail_power[start:end] - weekly_firm_total[start:end]
        
        # NEW: Calculate exact IT capacity remaining so Batch never exceeds the ceiling
        it_headroom = np.maximum(0, IT_CAPACITY - weekly_firm_total[start:end])
        max_possible_batch = np.minimum(avail_flex, it_headroom)
        max_possible_batch = np.minimum(max_possible_batch, MAX_BATCH_POWER)
        
        # Draw the "Target Quota" line for visual reference
        batch_avg_line[start:end] = BATCH_TARGET_MW
        
        batch_daily = np.zeros(24)
        deficit = 0.0
        
        # Step A: Try to meet the flat target, constrained by IT Capacity and Weather
        for h in range(24):
            if max_possible_batch[h] >= BATCH_TARGET_MW:
                batch_daily[h] = BATCH_TARGET_MW
            else:
                batch_daily[h] = max(0, max_possible_batch[h])
                deficit += (BATCH_TARGET_MW - batch_daily[h])
                
        # Step B: Reschedule the missed work (deficit) directly into the solar peaks
        if deficit > 0:
            rem_capacity = max_possible_batch - batch_daily
            best_hours = np.argsort(avail_flex)[::-1] # Sort sunniest/windiest hours first
            
            for h in best_hours:
                if deficit <= 0: break
                if rem_capacity[h] > 0:
                    add = min(deficit, rem_capacity[h])
                    batch_daily[h] += add
                    deficit -= add
                        
        batch_load[start:end] = batch_daily

    # Spot instances act as a fluid, sponging up whatever is left OVER AFTER batch is satisfied
    for i in range(total_hours):
        rigid_and_batch = weekly_firm_total[i] + batch_load[i]
        rem_power = max(0, avail_power[i] - rigid_and_batch)
        
        # Cannot exceed installed IT capacity
        available_it_headroom = max(0, IT_CAPACITY - rigid_and_batch)
        spot_load[i] = min(rem_power, available_it_headroom)
        
        # Anything above IT capacity is curtailed
        curtailment[i] = max(0, avail_power[i] - IT_CAPACITY)

    # --- 4. PLOTTING ---
    fig, ax = plt.subplots(figsize=(16, 8))
    
    c_yearly = '#1A5F7A'  # Dark Teal
    c_weekly = '#0e8796'  # Mid Teal
    c_batch  = '#3DACC2'  # Saturated Teal for Schedulable load
    c_spot   = '#C4EDF5'  # Light, airy blue for opportunistic fluid fill
    c_curt   = '#E63946'  # Red
    c_avail  = '#2D3436'  # Dark Grey
    c_dash   = '#E67E22'  # Orange for the Average Quota line

    # 1. Yearly Baseload
    ax.fill_between(t, 0, YEARLY_BASELOAD, facecolor=c_yearly, linewidth=0)

    # 2. Weekly Extra Baseload
    ax.fill_between(t, YEARLY_BASELOAD, weekly_firm_total, facecolor=c_weekly, linewidth=0)
    
    # 3. Batch Load (Solid fill, vertical bar borders removed)
    ax.fill_between(t, weekly_firm_total, weekly_firm_total + batch_load, facecolor=c_batch, linewidth=0)
    
    # 3b. Batch Target Line (Shows the unoptimized flat target)
    ax.step(t, weekly_firm_total + batch_avg_line, where='post', 
            color=c_dash, linestyle='--', linewidth=2.5, alpha=0.9)
    
    # 4. Spot Load (Fluid fill over the batch layer)
    y_spot_base = weekly_firm_total + batch_load
    ax.fill_between(t, y_spot_base, y_spot_base + spot_load, facecolor=c_spot, linewidth=0)
    
    # 5. Curtailment (Perfectly spills over the IT capacity line)
    stack_bottom = weekly_firm_total + batch_load + spot_load
    ax.fill_between(t, stack_bottom, stack_bottom + curtailment, facecolor=c_curt, linewidth=0)

    # Available Power Envelope & IT Limit
    ax.plot(t, avail_power, color=c_avail, linewidth=2.5, linestyle='-')
    ax.axhline(IT_CAPACITY, color='black', linestyle='--', linewidth=2.5)

    # --- 5. FORMATTING AND ANNOTATIONS ---
    # Vertical week dividers
    ax.axvline(168, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.axvline(336, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

    # Y-Axis (Qualitative)
    ax.set_yticks([])
    ax.set_ylabel("Power Capacity $\\longrightarrow$", fontsize=14, fontweight='bold', loc='top')
    ax.set_ylim(0, IT_CAPACITY + 8) 
    ax.set_xlim(0, total_hours)

    # X-axis
    ax.set_xticks([84, 168 + 84, 336 + 84])
    ax.set_xticklabels([
        "WEEK 1\nHigh Generation Forecast", 
        "WEEK 2\nAverage Generation Forecast", 
        "WEEK 3\nLow Generation Forecast"
    ], fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', pad=10)
    
    ax.set_title("Dynamic Workload Stacking", fontsize=18, fontweight='bold', pad=20)

    # Custom Legend setup
    patch_yearly = mpatches.Patch(color=c_yearly, label='Yearly Firm Load (Tier A)')
    patch_weekly = mpatches.Patch(color=c_weekly, label='Weekly Capacity Blocks')
    patch_batch  = mpatches.Patch(color=c_batch, label='Daily Flexible (Tier B1)')
    patch_spot   = mpatches.Patch(color=c_spot, label='Fully Flexible (Tier C)')
    patch_curt   = mpatches.Patch(color=c_curt, label='Curtailed Power')
    
    line_avail = Line2D([0], [0], color=c_avail, linewidth=2.5, label='Power Forecast (Wind + Solar)')
    line_it = Line2D([0], [0], color='black', linestyle='--', linewidth=2.5, label='Max Installed IT Capacity')
    line_avg = Line2D([0], [0], color=c_dash, linestyle='--', linewidth=2.5, label='Average Daily Flexible Load (Tier B1)')
    
    handles = [line_it, line_avail, patch_curt, patch_spot, patch_batch, line_avg, patch_weekly, patch_yearly]
    ax.legend(handles=handles, loc='upper right', fontsize=10, shadow=True, framealpha=1)

    # --- 6. ANNOTATIONS ---
    # Curtailment Peak
    ax.annotate('IT Capacity Max \n(Curtailment)', xy=(36, IT_CAPACITY), xytext=(10, IT_CAPACITY + 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                fontsize=10, fontweight='bold', color='black')
                
   
    plt.tight_layout()
    plt.savefig('Thesis_Methodology_Diagram_Final.svg', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_methodology_diagram()