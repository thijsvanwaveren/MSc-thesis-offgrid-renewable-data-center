# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:15:26 2026

@author: thijs
"""

import os

# Set the folder to search (current folder)
search_path = "." 
search_str = "mendelev"

print(f"Searching for '{search_str}' in {os.path.abspath(search_path)}...")

for root, dirs, files in os.walk(search_path):
    for file in files:
        if file.endswith((".py", ".yaml", ".csv", ".json")):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if search_str in content:
                        print(f"\n FOUND IT IN: {filepath}")
                        # Print the context
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if search_str in line:
                                print(f"   Line {i+1}: {line.strip()}")
            except Exception as e:
                print(f"Could not read {filepath}: {e}")

print("\nSearch complete.")