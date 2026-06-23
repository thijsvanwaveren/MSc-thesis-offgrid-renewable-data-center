# -*- coding: utf-8 -*-
"""
EMS-based generation duration curve for 8 MW Tier A IT load.

This script:
- Runs the actual HyDesign EMS for an 8 MW Tier A IT workload.
- Uses the EMS BESS dispatch instead of a manually modeled BESS.
- Applies PUE through the EMS power balance.
- Plots renewable generation allocation on a facility-side electrical power basis.

Interpretation:
- 8 MW Tier A = IT workload power.
- With PUE = 1.15, this requires 9.2 MW facility-side electrical power.
- The BESS charge/discharge and curtailment come directly from the EMS solution.
"""

import os
import sys
import yaml
import inspect
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# USER SETTINGS
# =============================================================================

TIER_A_IT_MW = 8.0
PUE = 1.15
TIER_A_FACILITY_MW = TIER_A_IT_MW * PUE

HOURS_TO_PLOT = 8760
SITE_NAME = "Denmark_good_solar"

# Fixed HPP design:
# [clearance, specific_power, rated_power, Nwt, wind_MW_per_km2,
#  solar_MW, surface_tilt, surface_azimuth, DC_AC_ratio,
#  battery_power_MW, battery_duration_h, cost_of_battery_P_fluct_in_peak_price_ratio]
FIXED_DESIGN = [35, 300, 5, 20, 7, 180, 39, 180, 1.25, 25, 8, 10]

# Exclude Tier C for this diagnostic plot.
# This ensures the EMS only serves the 8 MW Tier A workload.
os.environ["REWARD_C2"] = "1.0"

# =============================================================================
# DIRECTORY SETUP & HYDESIGN IMPORTS
# =============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_dir = C:\Users\thijs\Downloads\hydesign\hydesign\examples\Thesis_ThijsvanWaveren

ROOT_DIR = r"C:\Users\thijs\Downloads\hydesign"
if sys.path[0] != ROOT_DIR:
    sys.path.insert(0, ROOT_DIR)

from hydesign.assembly.hpp_assembly_offgrid_thijs_2_2_26 import (
    hpp_model_constant_output_offgrid as hpp_model
)

# =============================================================================
# PARAMETER CONFIGURATION
# =============================================================================

def configure_parameters(thesis_dir):
    """
    Creates temporary off-grid parameter file.
    Ensures:
    - no grid connection,
    - BESS one-way efficiency corresponds to 86% round-trip efficiency.
    """
    par_fn = os.path.join(thesis_dir, "inputs", "hpp_pars.yml")

    with open(par_fn, "r") as f:
        sim_pars = yaml.safe_load(f)

    sim_pars["G_MW"] = 0
    sim_pars["battery_charge_efficiency"] = float(np.sqrt(0.86))

    temp_fn = os.path.join(thesis_dir, "inputs", "hpp_pars_offgrid_tierA_8MW_temp.yml")

    with open(temp_fn, "w") as f:
        yaml.dump(sim_pars, f)

    return temp_fn

# =============================================================================
# RUN EMS CASE
# =============================================================================

def run_tier_a_case():
    """
    Runs HyDesign EMS for an 8 MW Tier A IT load.
    Returns first-year EMS outputs as a DataFrame.
    """

    N_life = 25 * 8760

    examples_sites = pd.read_csv(
        os.path.join(thesis_dir, "..", "examples_sites.csv"),
        sep=";"
    )

    ex_site = examples_sites.loc[examples_sites.name == SITE_NAME]
    weather_fn = os.path.join(thesis_dir, "..", ex_site["input_ts_fn"].values[0])
    sim_pars_fn = configure_parameters(thesis_dir)

    print("--- Running HyDesign EMS for 8 MW Tier A IT load ---")
    print(f"Tier A IT load:                     {TIER_A_IT_MW:.2f} MW_IT")
    print(f"PUE:                                {PUE:.2f}")
    print(f"Facility-side demand if fully met:  {TIER_A_FACILITY_MW:.2f} MW_el")

    # 8 MW Tier A profile on IT basis.
    tier_a_ts = np.full(N_life, TIER_A_IT_MW)

    # No committed flexible workloads.
    tier_b_ts = np.zeros(N_life)
    tier_b2_ts = np.zeros(N_life)

    # Installed IT capacity equals Tier A load.
    # This prevents any additional workload from being served.
    load_profile_ts = np.full(N_life, TIER_A_IT_MW)

    hpp_kwargs = dict(
        latitude=ex_site["latitude"].values[0],
        longitude=ex_site["longitude"].values[0],
        altitude=ex_site["altitude"].values[0],
        num_batteries=1,
        work_dir=current_dir,
        input_ts_fn=weather_fn,
        sim_pars_fn=sim_pars_fn,
        tier_a_profile=tier_a_ts,
        tier_b_profile=tier_b_ts,
        load_profile_ts=load_profile_ts,
        battery_deg=False
    )

    # Robust handling in case the local assembly also requires tier_b2_profile.
    sig = inspect.signature(hpp_model)
    if "tier_b2_profile" in sig.parameters:
        hpp_kwargs["tier_b2_profile"] = tier_b2_ts

    hpp = hpp_model(**hpp_kwargs)

    print("Evaluating fixed HPP design through EMS...")
    hpp.evaluate(*FIXED_DESIGN)
    prob = hpp.prob

    print("Extracting first-year EMS outputs...")

    wind = prob.get_val("ems.wind_t_ext")[:HOURS_TO_PLOT]
    solar = prob.get_val("ems.solar_t_ext")[:HOURS_TO_PLOT]
    available_re = wind + solar

    # In your EMS, ems.hpp_t is served IT workload power,
    # despite the variable name suggesting HPP output.
    served_it = prob.get_val("ems.hpp_t")[:HOURS_TO_PLOT]
    served_facility = PUE * served_it

    curtailment = prob.get_val("ems.hpp_curt_t")[:HOURS_TO_PLOT]

    # b_t positive = discharge, negative = charge
    b_net = prob.get_val("ems.b_t")[:HOURS_TO_PLOT]
    bess_discharge = np.maximum(b_net, 0.0)
    bess_charge = np.maximum(-b_net, 0.0)

    try:
        unserved_a_it = prob.get_val("ems.Unserved_A")[:HOURS_TO_PLOT]
    except Exception:
        unserved_a_it = np.maximum(TIER_A_IT_MW - served_it, 0.0)

    # Facility-side unserved load.
    unserved_a_facility = PUE * unserved_a_it

    # Renewable generation used directly to serve the facility load.
    # From the balance:
    # RE + discharge = facility load + charge + curtailment
    # Therefore:
    # RE used directly for load = facility load - BESS discharge
    # and:
    # RE = direct_to_load + charge + curtailment
    re_direct_to_load = served_facility - bess_discharge
    re_direct_to_load = np.clip(re_direct_to_load, 0.0, available_re)

    # Energy balance check:
    # RE + discharge = facility load + charge + curtailment
    balance_residual = (
        available_re + bess_discharge
        - served_facility
        - bess_charge
        - curtailment
    )

    df = pd.DataFrame({
        "Wind_MW": wind,
        "Solar_MW": solar,
        "Available_RE_MW": available_re,
        "Served_IT_MW": served_it,
        "Served_Facility_MW": served_facility,
        "Unserved_A_IT_MW": unserved_a_it,
        "Unserved_A_Facility_MW": unserved_a_facility,
        "RE_Direct_To_Load_MW": re_direct_to_load,
        "BESS_Charge_MW": bess_charge,
        "BESS_Discharge_MW": bess_discharge,
        "Curtailment_MW": curtailment,
        "BESS_Net_MW": b_net,
        "Balance_Residual_MW": balance_residual
    })

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    tol = 1e-6
    flf = np.mean(unserved_a_it <= tol)
    energy_served_share = 1 - np.sum(unserved_a_it) / (TIER_A_IT_MW * HOURS_TO_PLOT)

    avg_re = df["Available_RE_MW"].mean()
    avg_direct = df["RE_Direct_To_Load_MW"].mean()
    avg_charge = df["BESS_Charge_MW"].mean()
    avg_discharge = df["BESS_Discharge_MW"].mean()
    avg_curtail = df["Curtailment_MW"].mean()
    avg_served_it = df["Served_IT_MW"].mean()
    avg_served_facility = df["Served_Facility_MW"].mean()

    annual_charge = df["BESS_Charge_MW"].sum()
    annual_discharge = df["BESS_Discharge_MW"].sum()
    annual_bess_losses = annual_charge - annual_discharge

    print("\n--- Summary ---")
    print(f"Average available RE:                 {avg_re:.2f} MW")
    print(f"Average served IT load:               {avg_served_it:.2f} MW_IT")
    print(f"Average served facility load:         {avg_served_facility:.2f} MW_el")
    print(f"Average RE directly serving load:     {avg_direct:.2f} MW")
    print(f"Average RE sent to BESS charging:     {avg_charge:.2f} MW")
    print(f"Average BESS discharge to load:       {avg_discharge:.2f} MW")
    print(f"Average curtailment:                  {avg_curtail:.2f} MW")
    print(f"Tier A FLF:                           {flf * 100:.4f}%")
    print(f"Tier A energy-served reliability:     {energy_served_share * 100:.4f}%")
    print(f"Total unserved Tier A energy:         {df['Unserved_A_IT_MW'].sum():.4f} MWh_IT")
    print(f"Annual BESS charge energy:            {annual_charge:.2f} MWh")
    print(f"Annual BESS discharge energy:         {annual_discharge:.2f} MWh")
    print(f"Annual BESS losses / net storage use: {annual_bess_losses:.2f} MWh")
    print(f"Max absolute balance residual:        {np.max(np.abs(balance_residual)):.6e} MW")

    out_csv = os.path.join(current_dir, "TierA_8MW_EMS_Generation_Allocation_Data.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved EMS allocation data to: {out_csv}")

    return df

# =============================================================================
# PLOT
# =============================================================================

def plot_smooth_after_bess_duration_curve(df):
    """
    Smooth EMS-based duration curve.

    Plots:
    1. Original renewable generation duration curve.
    2. EMS-shaped power after BESS:
           P_after_BESS = P_RE + P_BESS_net
       where P_BESS_net = discharge - charge.
    3. Firm-load threshold:
           8 MW IT = 9.2 MW facility-side power.

    This plot intentionally does not show BESS charge/discharge as separate
    timestep-level areas, because those create visual spikes when duration-sorted.
    """

    df = df.copy()

    # -------------------------------------------------------------------------
    # Core signals
    # -------------------------------------------------------------------------

    if "Available_RE_MW" not in df.columns:
        df["Available_RE_MW"] = df["Wind_MW"] + df["Solar_MW"]

    # Your EMS convention:
    # BESS_Net_MW > 0 means discharging
    # BESS_Net_MW < 0 means charging
    df["After_BESS_MW"] = df["Available_RE_MW"] + df["BESS_Net_MW"]

    # Optional balance check:
    # EMS balance implies:
    # Available_RE + BESS_Net = Served_Facility + Curtailment
    if "Served_Facility_MW" in df.columns and "Curtailment_MW" in df.columns:
        residual = (
            df["After_BESS_MW"]
            - (df["Served_Facility_MW"] + df["Curtailment_MW"])
        )
        print(
            "Max |After_BESS - (Served_Facility + Curtailment)|:",
            f"{np.max(np.abs(residual)):.6e} MW"
        )

    # -------------------------------------------------------------------------
    # Duration curves: sort each signal independently
    # -------------------------------------------------------------------------
    # This gives smooth duration curves and avoids timestep-level BESS spikes.
    # These are distributions, not chronological curves.

    re_sorted = np.sort(df["Available_RE_MW"].values)[::-1]
    after_bess_sorted = np.sort(df["After_BESS_MW"].values)[::-1]

    x = np.arange(len(after_bess_sorted))

    firm_facility_mw = TIER_A_IT_MW * PUE  # 8 * 1.15 = 9.2 MW
    


    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="white")

    C_AFTER = "#113b5e"      # Deep navy
    C_RE = "#7f8c8d"         # Grey
    C_EXCESS = "#73b3d8"     # Light blue
    C_LINE = "#c0392b"       # Red


    ax.fill_between(
        x,
        re_sorted,
        firm_facility_mw,
        where=(re_sorted < firm_facility_mw),
        color="red",
        alpha=0.55,
        zorder=3,
        label="BESS discharge required to meet firm load"
    )
    
    # Clear horizontal firm-load line
    ax.axhline(
        y=firm_facility_mw,
        color="red",
        linestyle="--",
        linewidth=2.0,
        zorder=7,
        label=f"Firm load: {TIER_A_IT_MW:.0f} MW IT = {firm_facility_mw:.1f} MW facility"
    )
    
    # BESS discharging annotation
    ax.annotate(
        "BESS DISCHARGING",
        xy=(8600, firm_facility_mw), xycoords="data",
        xytext=(8600, 45), textcoords="data",
        arrowprops=dict(
            arrowstyle="->",
            color="red",
            lw=1.5,
            connectionstyle="arc3,rad=-0.2"
        ),
        fontsize=10,
        fontweight="bold",
        color="red",
        ha="center",
        zorder=8
    )


    # Firm-load area below 9.2 MW
    ax.fill_between(
        x,
        0,
        np.minimum(after_bess_sorted, firm_facility_mw),
        color=C_AFTER,
        alpha=0.90,
        zorder=2,
        label=f"Firm load served"
    )

    # Excess after BESS above firm load
    ax.fill_between(
        x,
        firm_facility_mw,
        after_bess_sorted,
        where=after_bess_sorted > firm_facility_mw,
        color=C_EXCESS,
        alpha=0.55,
        zorder=1,
        label="Excess Generation"
    )

    # Original renewable generation duration curve
    ax.plot(
        x,
        re_sorted,
        color=C_RE,
        linewidth=1.6,
        linestyle="--",
        alpha=0.95,
        zorder=4,
        label="HPP output excl. BESS"
    )

    # After-BESS duration curve
    ax.plot(
        x,
        after_bess_sorted,
        color=C_AFTER,
        linewidth=2.2,
        zorder=5,
        label="HPP output incl. BESS"
    )

    # # Firm-load threshold
    # ax.axhline(
    #     y=firm_facility_mw,
    #     color=C_LINE,
    #     linestyle="--",
    #     linewidth=2.2,
    #     zorder=6,
    #     label=f"Firm-load: {TIER_A_IT_MW:.0f} MW IT = {firm_facility_mw:.1f} MW facility"
    # )

    # -------------------------------------------------------------------------
    # Annotation
    # -------------------------------------------------------------------------

    ax.text(
        4100,
        firm_facility_mw - 8.5,
        f"Matched Demand: {firm_facility_mw:.1f} MW facility power ({TIER_A_IT_MW:.0f} MW Tier A)",
        fontsize=10,
        fontweight="bold",
        color='white',
        ha="center",
        va="bottom"
    )

    ax.text(
        2300,
        max(firm_facility_mw + 30, 45),
        "Excess generation \n(Curtailed without flexible workloads)",
        fontsize=11,
        fontweight="bold",
        color=C_AFTER,
        ha="center",
        va="center"
    )
    ax.annotate("BESS DISCHARGING",
                xy=(8600, 10), xycoords='data',
                xytext=(8600, 45), textcoords='data',
                arrowprops=dict(arrowstyle="->", color='red', lw=1.5, connectionstyle="arc3,rad=-0.2"),
                fontsize=10, fontweight='bold', color='red', ha='center', zorder=7)
    # Optional small annotation for the shift
    # ax.text(
    #     6100,
    #     firm_facility_mw + 18,
    #     "BESS flattens the distribution:\ncharges during surplus, discharges during deficits",
    #     fontsize=10,
    #     fontweight="bold",
    #     color=C_AFTER,
    #     ha="center",
    #     va="center"
    # )

    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------

    ax.set_xlabel(
        "Hours of the year",
        fontsize=12,
        fontweight="bold",
        color="#444444"
    )

    ax.set_ylabel(
        "Power (MW)",
        fontsize=12,
        fontweight="bold",
        color="#444444"
    )

    ax.set_xlim(0, len(after_bess_sorted))
    ax.set_ylim(0, max(re_sorted.max(), after_bess_sorted.max()) * 1.05)

    ax.grid(axis="y", linestyle="-", alpha=0.3, color="#b0b0b0")
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("#444444")

    ax.tick_params(axis="both", colors="#333333", labelsize=11)

    # # Secondary y-axis: IT-equivalent power
    # secax = ax.secondary_yaxis(
    #     "right",
    #     functions=(lambda y: y / PUE, lambda y: y * PUE)
    # )
    # secax.set_ylabel(
    #     "IT-equivalent workload power (MW$_{IT}$)",
    #     fontsize=11,
    #     fontweight="bold",
    #     color="#444444"
    # )
    # secax.tick_params(axis="y", labelsize=10, colors="#444444")

    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#e0e0e0",
        framealpha=0.95,
        fontsize=9
    )

    plt.tight_layout()

    out_svg = os.path.join(current_dir, "TierA_8MW_RE_vs_After_BESS_Duration_Curve.svg")
    out_png = os.path.join(current_dir, "TierA_8MW_RE_vs_After_BESS_Duration_Curve.png")

    plt.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    print(f"Saved plot to: {out_svg}")
    print(f"Saved plot to: {out_png}")

    # Useful numerical summary
    print("\n--- Duration curve summary ---")
    print(f"Firm-load threshold: {TIER_A_IT_MW:.2f} MW_IT = {firm_facility_mw:.2f} MW_facility")
    print(f"Average original RE: {np.mean(df['Available_RE_MW']):.2f} MW")
    print(f"Average after BESS:  {np.mean(df['After_BESS_MW']):.2f} MW")
    print(f"Average BESS net:    {np.mean(df['BESS_Net_MW']):.4f} MW")

    plt.show()# MAIN
# =============================================================================

if __name__ == "__main__":
    df_results = run_tier_a_case()
    plot_smooth_after_bess_duration_curve(df_results)