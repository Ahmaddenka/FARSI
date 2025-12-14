# FARSI_Project

This repository contains the implementation of FARSI, an FCM-executable architecture for organizational safety simulation and analysis.
All parameters are aligned with the values reported in the paper. For quick testing on a standard PC, use lighter settings (see Quickstart).

## Project Structure

- `field_data/` : Input data required to instantiate the network based on the organizational structure. Python scripts read from this path.
- `graph_data/` : Excel files required for graph construction. Python scripts read from this path.
- `graph_file/` : Pre-built graph files (no need to rebuild from Excel). Python scripts read from this path.
- `main/` : Main Python source code.
- `output/` : All generated outputs will be written here.

## How to Run

### 1) Build / Export Graph
- **Graph_Builder.py**  
  Builds the graph from `field_data/` and/or `graph_data/` and stores the generated graph in `graph_file/`.

### 2) Culture Analysis
- **Run_Culture_Analysis.py**  
  Runs the culture analysis workflow and stores outputs in `output/`.

### 3) Main Monte Carlo Simulation
- **Run_Monte_Carlo_Simulation.py**  
  Runs the main Monte Carlo simulation and stores outputs in `output/`.

### 4) Parameter Calibration (restricted)
- **Run_Param_Calibration.py**  
  Runs parameter calibration. This script requires confidential organizational evaluation data.
  The public version of the repository does not include the real calibration dataset.

### 5) Screening (Positive Shift / Randomized)
- **Run_positive_shift_randomized_screening.py**  
  Runs the screening workflow to identify influential organizational mechanisms.

### 6) Sensitivity Analysis
- **Run_Sensetivity_Analysis.py**  
  Runs the sensitivity analysis workflow and stores outputs in `output/`.

### 7) Test Trace (non-MC)
- **Run_Simulation_Test_Trace.py**  
  Runs a lightweight single-run simulation (without Monte Carlo) to inspect boundary behavior.

### 8) Validation / Uncertainty Analysis
- **Run_Validation.py**  
  Runs the validation and uncertainty-related analysis and stores outputs in `output/`.

## Quickstart (Lightweight Test)
For a quick test, reduce computational settings (e.g., number of runs, time steps, or simulation horizon) inside the run scripts or in the configuration section.
Suggested lightweight settings:
- Reduce Monte Carlo repetitions
- Reduce number of scenarios 

## Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
