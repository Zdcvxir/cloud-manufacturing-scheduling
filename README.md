# Scheduling in a cloud manufacturing environment with deteriorating effects and maintenance activities

This repository contains the code used for a parallel-batch scheduling problem
with position-dependent processing times and preventive maintenance decisions.
It includes:

- `vns.py`: the full Variable Neighborhood Search (VNS) heuristic.
- `simplified_vns.py`: a lighter VNS variant with random insertion shaking.
- `gurobi_model.py`: the exact mixed-integer programming model implemented with
  Gurobi.
- `config.py`: the benchmark instance and model parameters.
- `LICENSE`: MIT License terms for reuse and redistribution.

Job and time-slot identifiers in printed schedules are one-based to match the
paper notation.

## Requirements

The VNS scripts use only the Python standard library. The exact MIP model
requires Gurobi and a valid Gurobi license:

```bash
pip install -r requirements.txt
```

If you only want to run the VNS heuristics, `gurobipy` is not required.

## Quick Start

Run the full VNS heuristic with the default experimental settings:

```bash
python vns.py
```

Run one short reproducibility test:

```bash
python vns.py --runs 1 --time-limit 5 --seed 42
```

Run the simplified VNS variant:

```bash
python simplified_vns.py --runs 1 --time-limit 5 --seed 42
```

Build and solve the exact Gurobi model:

```bash
python gurobi_model.py --time-limit 3600
```

Build the Gurobi model without solving:

```bash
python gurobi_model.py --build-only
```

## Command Line Options

Both VNS scripts support:

- `--runs`: number of independent runs.
- `--time-limit`: time limit per run in seconds.
- `--seed`: base random seed.
- `--output`: path for the convergence-curve file.
- `--no-output`: disable convergence-curve output.
- `--quiet`: suppress per-improvement logs in a single run.

By default, VNS uses 20 seconds per run for instances with at least 100 jobs and
10 seconds otherwise, matching the original experimental scripts.

## Configuration

The benchmark data are stored in `config.py`. The lowercase variable names match
the mathematical notation in the paper, while uppercase aliases make the code
easier to read:

- `N`: number of jobs.
- `T`: number of time slots.
- `B`: maximum number of batches.
- `a`: position-dependent processing exponent.
- `seta`: setup time.
- `c`: batch capacity.
- `w`, `k`, `lam`: corrective-maintenance cost parameters.
- `u`: preventive-maintenance setup cost.
- `tao_l`: slot-specific processing cost coefficient.
- `p_j`: job processing times.
- `L_l`: slot length limits.
- `CP_l`: fixed costs for selected slots.

`config.validate_instance()` checks the basic dimensions before each algorithm
starts.

## Outputs

The VNS scripts print the best objective value, preventive-maintenance policy,
slot occupancy, and cost breakdown. Unless `--no-output` is set, they also write
a tab-separated convergence curve next to the script or to the path given by
`--output`.

## License

This project is released under the MIT License. See `LICENSE` for details.
