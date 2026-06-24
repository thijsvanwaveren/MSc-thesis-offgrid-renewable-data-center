# Off-Grid Data Center Optimization with Hybrid Power Plants

This repository contains the simulation and optimization codebase for the master's thesis: 
**"Powering Flexible Data Centers with Off-Grid Renewable Hybrid Power Plants"**. The thesis is available on [repository.tudelft.nl](https://repository.tudelft.nl/).

## About the Project
Global data center demand is rising, but grid congestion increasingly forces operators to seek off-grid alternatives. 
This project models an off-grid Data Center powered by a Hybrid Power Plant (HPP) combining wind (100 MW), solar (180 MW), and battery storage (25 MW / 8-hour).


Because variable renewable supply conflicts with the high-reliability power typically required by data centers, 
this codebase models an Energy Management System (EMS) that dispatches different tiers of workload flexibility (firm, daily flexible, weekly flexible, and fully flexible) to achieve a 99.9% reliability target.

**Key Findings:**
* **Flexibility is crucial:** Supplying only firm loads is inefficient, resulting in high curtailment and high levelized cost of energy delivered (LCOED). Flexible workloads shift demand to high-generation periods, improving HPP utilization and economic performance.
* **Sizing trade-offs:** The economically optimal data center size is a trade-off. Smaller sizes are constrained by IT hardware limits, while larger sizes are constrained by the physical generation profile of the HPP.

## Repository Structure

The codebase builds upon the [hydesign](https://github.com/DTUWindEnergy/hydesign) framework, introducing a custom EMS and assembly for multi-tier workload dispatch.

```text
hydesign/                                        # Root repository
│
├── hydesign/                                    # Main source code
│   ├── ems/                   
│   │   └── ems_incltierb2_thijs_3_3_26.py       # Custom EMS handling supply and workload dispatch
│   ├── assembly/              
│   │   └── hpp_assembly_tierb2_thijs_3_3_26.py  # Custom HPP assembly integrating the EMS
│   │
│   └── examples/
│       ├── example_sites.csv                             # Contains all ERA5 weather data
│       └── Thesis_ThijsvanWaveren - backup/              # Main thesis working directory
│           ├── inputs/                                   # Input data and YAML parameters
│           └── scripts/                                  # Core execution scripts
│               ├── MAIN PARAMETER SWEEP.py               # Full parameter sweep for feasible combinations
│               └── Financial Performance of DC Sizes.py  # Macroeconomic calculation engine
