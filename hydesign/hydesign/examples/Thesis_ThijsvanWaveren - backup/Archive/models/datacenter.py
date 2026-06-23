# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 15:01:57 2026

@author: thijs
"""

# models/datacenter.py
import numpy as np

class DataCenterModel:
    def __init__(self, total_it_capacity, pue=1.15):
        """
        Initialize the Data Center Model.
        
        Args:
            total_it_capacity (float): Total IT Power Capacity in MW.
            pue (float): Power Usage Effectiveness (default 1.15)[cite: 522].
        """
        self.capacity = total_it_capacity
        self.pue = pue

    def generate_profile(self, scenario_name):
        """
        Generates Tier A and Tier B load profiles based on Workload Mix Scenarios.
        
        Implements Table 2.3 from the Thesis[cite: 615, 616].
        
        Args:
            scenario_name (str): Name of the scenario (e.g., "Batch_Focused").
            
        Returns:
            tier_a_profile (np.array): Hourly power demand (MW) for 8760 hours.
            tier_b_daily_energy (np.array): Daily energy requirement (MWh) for 365 days.
        """
        
        # Define scenarios based on Table 2.3 
        # Format: (Tier A Share, Tier B Share)
        scenarios = {
            "Not_Flexible":            (1.0, 0.0), # 100% Interactive
            "Interactive_Focused":     (0.8, 0.2), # 80% Interactive
            "Interactive_Traditional": (0.6, 0.4), # 60% Interactive
            "Batch_Traditional":       (0.4, 0.6), # 60% Batch
            "Batch_Focused":           (0.2, 0.8), # 80% Batch
            "Infinitely_Flexible":     (0, 1), # 100% Batch
        }

        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}")

        share_a, share_b = scenarios[scenario_name]
        
        # --- 1. Tier A (Constant Baseload) [cite: 550] ---
        # Returns an array of 8760 hours
        # Tier A is constant power (MW)
        tier_a_profile = np.full(8760, self.capacity * share_a *self.pue)

        # --- 2. Tier B (Daily Energy Buckets) [cite: 560] ---
        # Returns an array of 365 days
        # E_req = Power * 24 hours (MWh/day)
        tier_b_daily_energy = np.full(365, self.capacity * share_b * 24 * self.pue)

        return tier_a_profile, tier_b_daily_energy