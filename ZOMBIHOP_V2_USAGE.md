# ZoMBI-Hop v2 Usage Guide

## Overview

`zombihop_linebo_v2.py` combines the modern checkpoint-enabled architecture from `zombihop_linebo_final.py` with the database communication functionality from `zombihop_linebo_v1.py`.

## Key Features

1. **Checkpoint/Resume Support**: Automatically saves progress and can resume from any 4-digit UUID
2. **Database Communication**: Integrates with the Archerfish system via SQLite databases
3. **Simplex-Constrained Optimization**: Optimized for compositional space (sum = 1)
4. **GPU Acceleration**: Uses PyTorch and GPyTorch for fast computations
5. **LineBO Sampling**: Efficient line-based experimental design
6. **Unlimited by Default**: New trials run with unlimited time and activations until manually stopped

## Quick Start

### Starting a New Trial

```python
from zombihop_linebo_v2 import run_zombi_main

# Run with default settings (unlimited time and activations)
results = run_zombi_main()

# Or specify limits
results = run_zombi_main(
    resume_uuid=None,  # New trial
    max_activations=5,  # Find 5 needles
    time_limit_hours=24  # 24 hour limit
)
```

### Resuming an Existing Trial

```python
from zombihop_linebo_v2 import run_zombi_main

# Resume from checkpoint using 4-digit UUID
results = run_zombi_main(
    resume_uuid='a2fe',  # Your 4-digit trial UUID
    max_activations=5,
    time_limit_hours=24
)
```

### Command Line Usage

```bash
# Start new trial with unlimited time/activations (default)
python zombihop_linebo_v2.py

# Start new trial with limits
python zombihop_linebo_v2.py --activations 5 --time-limit 24

# Resume existing trial using 4-digit UUID
python zombihop_linebo_v2.py --resume a2fe --activations 5 --time-limit 24
```

## Configuration

### Key Parameters

The main configuration is in the `run_zombi_main()` function:

```python
# Problem dimensions
dimensions = 3  # Number of optimizing dimensions
n_experiments = 24  # Points per line evaluation

# Optimization settings
max_iterations = 10  # Iterations per zoom level
max_gp_points = 150  # Maximum GP training points
max_activations = 5  # Maximum needles to find

# Zoom settings
max_zooms = 3  # Zoom levels per activation
top_m_points = 3  # Points to use for bounds calculation

# LineBO settings
linebo_num_lines = 30  # Number of candidate lines
linebo_pts_per_line = 50  # Points per line for integration

# Penalty settings
penalization_threshold = 1e-3  # Convergence threshold
penalty_max_radius = 0.3  # Maximum penalty radius
```

### Optimizing Dimensions

Change which dimensions to optimize by modifying:

```python
OPTIMIZING_DIMS = [0, 1, 8]  # Which of the 10 dimensions to use
```

### Fixed Composition Endpoints

To use fixed composition endpoints:

```python
fixed_comp = True
fixed_comp_start = [0, 1, 0]  # Starting composition
fixed_comp_end = [0, 0, 1]  # Ending composition
```

## Database Setup

The code requires three SQLite databases in `./sql/`:

1. **objective.db**: Main objective values from experiments
2. **compositions.db**: Composition data
3. **objective_memory.db**: Memory cache for objectives

The communication protocol uses a handshake mechanism:
- During initialization: processes available data without handshake
- After initialization: waits for `new_objective_available` flag

## Checkpointing

Checkpoints are automatically saved to `./checkpoints/run_<uuid>/`:

```
checkpoints/
└── run_a2fe1234/
    ├── config.json          # Configuration
    ├── current_state.txt    # Current state label
    └── states/
        ├── act0_zoom0_iter0/
        │   ├── data.pt      # X, Y, bounds
        │   ├── gp_state.pt  # GP model state
        │   └── meta.json    # Metadata
        └── ...
```

### Finding Trial UUID

The UUID is a 4-digit hexadecimal code (e.g., `a2fe`, `f358`) printed when starting a new trial:

```
✅ Starting new trial with UUID: a2fe
```

Or check `checkpoints/run_*/config.json`:

```json
{
  "run_uuid": "a2fe",
  ...
}
```

Or simply look at the checkpoint directory names:
```
checkpoints/
├── run_a2fe/
├── run_f358/
└── run_9722/
```

## Advanced Usage

### Custom Objective Function

To use a different objective function:

```python
def my_objective(ordered_endpoints, num_experiments=24):
    """
    Args:
        ordered_endpoints: (num_lines, 2, dimensions) numpy array
    Returns:
        x_meas: (n, dimensions) measured compositions
        y: (n,) objective values
    """
    # Your implementation here
    return x_meas, y

# Use in LineBO
linebo = LineBO(
    objective_function=my_objective,
    dimensions=3,
    device='cuda'
)
```

### Programmatic Access

For more control, use the classes directly:

```python
from zombihop_linebo_v2 import ZoMBIHop, LineBO, proj_simplex, random_simplex, zero_sum_dirs
import torch

# Setup
device = 'cuda'
dimensions = 3
bounds = torch.tensor([[0.0, 1.0]] * dimensions, device=device)

# Create LineBO
linebo = LineBO(
    objective_function=your_objective,
    dimensions=dimensions,
    num_points_per_line=50,
    num_lines=30,
    device=device
)

# Initial data (your initialization logic)
X_init_actual = ...
X_init_expected = ...
Y_init = ...

# Create optimizer
optimizer = ZoMBIHop(
    objective=linebo.sampler,
    bounds=bounds,
    X_init_actual=X_init_actual,
    X_init_expected=X_init_expected,
    Y_init=Y_init,
    proj_fn=proj_simplex,
    random_sampler=lambda n, b: random_simplex(n, b[:, 0], b[:, 1], device=device),
    random_direction_sampler=lambda n: zero_sum_dirs(n, dimensions, device=device),
    max_activations=5,
    device=device
)

# Run
results = optimizer.run(max_activations=5, time_limit_hours=24)
```

## Results

The `run()` method returns a dictionary with:

```python
{
    'needles': [...],              # List of needle locations
    'needle_values': [...],        # Objective values at needles
    'penalty_regions': [...],      # Penalty radii
    'best_value': float,           # Best value found
    'total_evaluations': int,      # Total evaluations performed
    'X_actual': Tensor,            # All evaluated points
    'Y': Tensor,                   # All objective values
    'converged_activations': int   # Number of converged activations
}
```

## Troubleshooting

### No Data Available

If you see "No data available" errors:
1. Check that databases exist in `./sql/`
2. Verify communication module is working
3. Check handshake mechanism in database

### Resume Not Working

If resume fails:
1. Check UUID is correct
2. Verify checkpoint directory exists: `./checkpoints/run_<uuid>/`
3. Check that state files are not corrupted

### GPU Memory Issues

If running out of GPU memory:
1. Reduce `max_gp_points` (default: 150)
2. Reduce `raw` parameter (default: 10000)
3. Reduce `linebo_num_lines` (default: 30)

### Convergence Issues

If not finding needles:
1. Increase `max_iterations` (default: 10)
2. Increase `max_zooms` (default: 3)
3. Decrease `penalization_threshold` (default: 1e-3)
4. Adjust `improvement_threshold_noise_mult` (default: 1.5)

## Dependencies

```
torch>=2.0.0
gpytorch>=1.11
numpy>=1.20.0
```

## Differences from v1

1. **Architecture**: Uses torch throughout instead of numpy
2. **Checkpointing**: Automatic state saving and loading
3. **Resume Support**: Can continue from any saved state
4. **Better GP**: Uses GPyTorch for better performance
5. **Cleaner Interface**: Simplified API with `run()` method

## See Also

- `zombihop_linebo_v1.py` - Original numpy-based implementation
- `zombihop_linebo_final.py` - Base class with checkpoint support
- `test_24hour_variable_minima.py` - Example usage with test functions

