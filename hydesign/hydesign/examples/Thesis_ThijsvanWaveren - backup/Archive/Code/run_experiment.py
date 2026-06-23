# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 15:05:01 2026

@author: thijs
"""


dc_model = DataCenterModel(total_it_capacity=20, pue=1.15)
L_a, E_b = dc_model.generate_profile("Batch_Focused")

results = optimize_thesis_dispatch(wind_data, solar_data, L_a, E_b, cost_params)