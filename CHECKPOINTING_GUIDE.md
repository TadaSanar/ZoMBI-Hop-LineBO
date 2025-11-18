# ZoMBIHop Checkpointing Guide

## Overview

The `zombihop_linebo_final.py` file now includes automatic checkpointing functionality that saves the complete state of the optimizer at every iteration. This allows you to:

1. Resume interrupted optimization runs
2. Inspect the state at any point in the optimization
3. Track progress over time
4. Analyze distances between expected and actual points

## Directory Structure

When you run ZoMBIHop, it creates the following directory structure:

```
zombihop_checkpoints/
└── run_{uuid}/
    ├── config.json                    # Run configuration
    ├── current_state.txt              # Points to the latest state
    └── states/
        ├── init/                      # Initial state
        ├── act0_zoom0_iter0/          # Each iteration
        ├── act0_zoom0_iter1/
        ├── ...
        ├── act0_zoom0_iter{n}_needle/ # When a needle is found
        └── final/                     # Final state
```

Each state directory contains:
- `tensors.pt` - PyTorch tensors (all data, including distances)
- `needles_results.json` - List of needles found so far
- `tracking.json` - Iteration tracking (activation, zoom, iteration)
- `stats.json` - Summary statistics (best value, noise, distances)
- `all_points.csv` - Human-readable CSV with all points and distances

## Usage

### Starting a New Run

```python
import torch
from zombihop_linebo_final import ZoMBIHop

# Initialize with your objective function and initial data
optimizer = ZoMBIHop(
    objective=my_objective_function,
    bounds=bounds,
    X_init_actual=X_init_actual,
    X_init_expected=X_init_expected,
    Y_init=Y_init,
    # ... other parameters ...
)

# Run will automatically create a new UUID and checkpoint directory
# The UUID will be printed when you start
results = optimizer.run()
print(f"Run UUID: {optimizer.run_uuid}")
```

### Resuming from a Saved Run

```python
# To resume from a saved run, just pass the 4-digit UUID
# All other parameters (bounds, X_init, etc.) will be ignored
optimizer = ZoMBIHop(
    objective=my_objective_function,  # Still needed for evaluation
    bounds=None,  # Will be loaded from checkpoint
    X_init_actual=None,
    X_init_expected=None,
    Y_init=None,
    run_uuid="a3f4",  # Your 4-digit UUID
    checkpoint_dir='zombihop_checkpoints'  # Optional: specify directory
)

# Continue from where it left off
results = optimizer.run()
```

## What Gets Saved

At each iteration, the following data is saved:

### Tensors (in `tensors.pt`)
- `bounds` - Current optimization bounds
- `X_init_actual` - Initial actual points
- `X_init_expected` - Initial expected points
- `Y_init` - Initial objective values
- `X_all_actual` - All actual points sampled so far
- `X_all_expected` - All expected points sampled so far
- `Y_all` - All objective values
- `distances` - Euclidean distances between expected and actual points
- `needles` - All needles found
- `needle_vals` - Values at needles
- `needle_indices` - Indices of needles in X_all
- `needle_penalty_radii` - Penalty radius for each needle
- `penalty_mask` - Boolean mask of penalized points

### Statistics (in `stats.json`)
- `iteration_label` - Label for this iteration
- `timestamp` - Unix timestamp
- `num_points_total` - Total number of points sampled
- `num_needles` - Number of needles found
- `best_value` - Best objective value found
- `input_noise` - Normalized input noise level
- `mean_distance` - Mean distance between expected/actual
- `median_distance` - Median distance
- `max_distance` - Maximum distance

### Human-Readable Data (in `all_points.csv`)
CSV file with columns:
- `index` - Point index
- `y_value` - Objective value
- `x_actual_{i}` - Actual coordinates (one column per dimension)
- `x_expected_{i}` - Expected coordinates (one column per dimension)
- `distance` - Distance between expected and actual
- `penalized` - Whether this point is in a penalized region

## Checkpoint Labels

Different checkpoint labels indicate different events:
- `init` - Initial state
- `act{a}_zoom{z}_iter{i}` - Normal iteration
- `act{a}_zoom{z}_iter{i}_needle` - Needle found
- `act{a}_zoom{z}_iter{i}_failed` - Iteration failed
- `act{a}_zoom{z}_complete` - Zoom level complete
- `act{a}_zoom{z}_finished` - Optimization finished (too much area penalized)
- `final` - Final state after all activations

## Analyzing Saved Data

You can load and analyze saved states without running the optimizer:

```python
import torch
import json
from pathlib import Path

# Load a specific state
run_dir = Path('zombihop_checkpoints/run_a3f4')
state_dir = run_dir / 'states' / 'act0_zoom0_iter5'

# Load tensors
tensors = torch.load(state_dir / 'tensors.pt')
X_all_actual = tensors['X_all_actual']
Y_all = tensors['Y_all']
distances = tensors['distances']

# Load statistics
with open(state_dir / 'stats.json', 'r') as f:
    stats = json.load(f)
    
print(f"Best value: {stats['best_value']}")
print(f"Mean distance: {stats['mean_distance']}")

# Or analyze the CSV directly
import pandas as pd
df = pd.read_csv(state_dir / 'all_points.csv')
print(df.describe())
```

## Notes

- The checkpoint directory can be customized with the `checkpoint_dir` parameter
- The UUID is automatically generated as a 4-digit string (first 4 chars of a UUID4)
- States are never deleted automatically - manage disk space as needed
- When resuming, the optimizer continues from the last saved state
- All parameters are saved in `config.json` for reference

