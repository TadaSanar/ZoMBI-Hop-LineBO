# 24-Hour ZoMBIHop Trial: 10 Minima in 10D

This is a comprehensive long-running test designed to evaluate ZoMBIHop's performance on a challenging multi-minima optimization problem.

## Test Configuration

### Problem Setup
- **Dimensions**: 10D simplex
- **Number of Minima**: 10 (well-separated, minimum distance 0.3)
- **Objective Function**: Multi-Minima Ackley with hard parameters
- **Time Limit**: 24 hours
- **Activations**: Unlimited (runs until time limit or convergence)

### Hard Parameters
```python
HARD_PARAMS = {
    'global_scale': 6.0,      # Lower scale makes landscape harder
    'exp_scale': 0.3,         # Higher exp_scale makes basins narrower  
    'sharpness': 9.0,         # Higher sharpness creates sharper peaks
}
```

### Noise Levels
- **Input Noise**: σ² = 1e-4
- **Output Noise**: σ² = 1e-3

## Running the Test

### Start a New Trial

```bash
python test_24hour_10minima.py
```

This will:
1. Generate 10 distinguishable minima on the 10D simplex
2. Save minima locations to `trial_24hour_10minima/minima_locations.pt`
3. Initialize ZoMBIHop with 3 random initial points
4. Start optimization with unlimited activations and 24-hour time limit
5. Save checkpoints at every iteration to `trial_24hour_10minima/checkpoints/`
6. Print progress updates including elapsed time

### Resume an Interrupted Trial

If the trial is interrupted (power loss, crash, manual stop), resume with:

```bash
python test_24hour_10minima.py {UUID}
```

Replace `{UUID}` with the 4-digit UUID that was printed when the trial started. For example:

```bash
python test_24hour_10minima.py a3f4
```

The script will:
1. Load the saved minima locations
2. Load the optimizer state from the checkpoint
3. Continue from exactly where it left off
4. Use the remaining time from the original 24-hour limit

### Manual Stop

You can safely interrupt the trial at any time with `Ctrl+C`. The current state will be saved, and you'll see a message showing how to resume:

```
Trial interrupted by user. State has been saved.
Resume with: run_24hour_trial(resume_uuid='a3f4')
```

## Output and Results

### Directory Structure

```
trial_24hour_10minima/
├── minima_locations.pt              # True minima locations
├── trial_metadata.json              # Trial configuration and start time
├── results_{uuid}.json              # Final results (when complete)
└── checkpoints/
    └── run_{uuid}/
        ├── config.json              # ZoMBIHop configuration
        ├── current_state.txt        # Points to latest state
        └── states/
            ├── init/                # Initial state
            ├── act0_zoom0_iter0/    # Each iteration
            ├── act0_zoom0_iter1/
            ├── ...
            └── final/               # Final state (or act{N}_timeout)
```

### Each Checkpoint Contains

Every state directory includes:
- `tensors.pt` - All optimization data (X, Y, needles, distances, etc.)
- `needles_results.json` - List of needles found
- `tracking.json` - Current position (activation/zoom/iteration)
- `stats.json` - Summary statistics
- `all_points.csv` - Human-readable data with all points and distances

### Progress Monitoring

During the run, you'll see:
```
---------- Activation 3 ----------
Elapsed time: 8.45 / 24.00 hours
---------- Zoom 2/3  ----------
Bounds: tensor([[...]])
---------- Iteration 5/10  ----------
Time: 1730000000.123
Elapsed: 8.46h / 24.00h (35.2%)
Sampling candidate: tensor([...])
Number of points in GP: 2450
```

### Needle Evaluation

After each needle is found, you'll see:
```
================================================================================
🎯 NEEDLE EVALUATION RESULTS 🎯
================================================================================
  ✅ Distance from minimum  1 to closest needle: 0.023456
  ✅ Distance from minimum  2 to closest needle: 0.034567
  ⚠️  Distance from minimum  3 to closest needle: 0.089012
  ...
--------------------------------------------------------------------------------
  Mean distance: 0.045678
  Max distance:  0.089012
  Needles found: 10
================================================================================
```

Status indicators:
- ✅ Distance < 0.05 (excellent)
- ⚠️ Distance 0.05-0.10 (acceptable)
- ❌ Distance > 0.10 (needs improvement)

### Final Results

When the trial completes (time limit reached or convergence), results are saved to `results_{uuid}.json`:

```json
{
  "run_uuid": "a3f4",
  "trial_duration_hours": 24.0,
  "total_points": 5432,
  "num_needles": 10,
  "distances_to_minima": [0.023, 0.034, 0.089, ...],
  "mean_distance": 0.045,
  "max_distance": 0.089,
  "best_value": 18.234,
  "success": true
}
```

**Success Criteria**: All 10 minima found within distance 0.10

## ZoMBIHop Configuration

The test uses these parameters:

```python
penalization_threshold = 1e-3
improvement_threshold_noise_mult = 2.0
input_noise_threshold_mult = 3.0
n_consecutive_no_improvements = 5
top_m_points = 4
max_zooms = 3
max_iterations = 10
n_restarts = 100
raw = 10000
penalty_num_directions = 100
penalty_max_radius = 0.3
penalty_radius_step = 0.01
max_gp_points = 3000
```

## Analyzing Results

### Load and Inspect a Checkpoint

```python
import torch
import json
from pathlib import Path

# Load a specific state
run_dir = Path('trial_24hour_10minima/checkpoints/run_a3f4')
state_dir = run_dir / 'states' / 'act2_zoom1_iter7'

# Load tensors
tensors = torch.load(state_dir / 'tensors.pt')
X_all = tensors['X_all_actual']
Y_all = tensors['Y_all']
needles = tensors['needles']
distances = tensors['distances']

# Load statistics
with open(state_dir / 'stats.json', 'r') as f:
    stats = json.load(f)

print(f"Points sampled: {stats['num_points_total']}")
print(f"Needles found: {stats['num_needles']}")
print(f"Best value: {stats['best_value']}")
```

### Compare Needles to True Minima

```python
import torch

# Load true minima
minima = torch.load('trial_24hour_10minima/minima_locations.pt')

# Load needles from final state
tensors = torch.load('trial_24hour_10minima/checkpoints/run_a3f4/states/final/tensors.pt')
needles = tensors['needles']

# Calculate distances
distances = torch.cdist(needles, minima)
print("Pairwise distances between needles and minima:")
print(distances)
```

### Plot Progress Over Time

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load all stats files
run_dir = Path('trial_24hour_10minima/checkpoints/run_a3f4')
states_dir = run_dir / 'states'

timestamps = []
num_points = []
num_needles = []
best_values = []

for state_dir in sorted(states_dir.iterdir()):
    if state_dir.is_dir():
        with open(state_dir / 'stats.json', 'r') as f:
            stats = json.load(f)
            timestamps.append(stats['timestamp'])
            num_points.append(stats['num_points_total'])
            num_needles.append(stats['num_needles'])
            best_values.append(stats['best_value'])

# Convert to relative hours
start_time = timestamps[0]
hours = [(t - start_time) / 3600 for t in timestamps]

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(hours, num_points)
axes[0].set_ylabel('Total Points Sampled')
axes[0].grid(True)

axes[1].plot(hours, num_needles)
axes[1].set_ylabel('Needles Found')
axes[1].axhline(y=10, color='r', linestyle='--', label='Target')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(hours, best_values)
axes[2].set_ylabel('Best Objective Value')
axes[2].set_xlabel('Time (hours)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('trial_progress.png', dpi=150)
print("Saved progress plot to trial_progress.png")
```

## Tips for Success

1. **Monitor GPU Usage**: Check that CUDA is being used effectively
   ```bash
   nvidia-smi -l 1
   ```

2. **Check Disk Space**: Each checkpoint is ~10-50 MB depending on data size
   ```bash
   du -sh trial_24hour_10minima/
   ```

3. **Resume After Updates**: If you need to update code, you can resume from any checkpoint

4. **Multiple Trials**: Run multiple trials in parallel on different GPUs
   ```bash
   CUDA_VISIBLE_DEVICES=0 python test_24hour_10minima.py &
   CUDA_VISIBLE_DEVICES=1 python test_24hour_10minima.py &
   ```

## Expected Behavior

- **First few hours**: Should find 2-4 needles quickly
- **Middle period**: Gradual discovery of remaining needles
- **Later hours**: Fine-tuning to distinguish close minima
- **Memory usage**: Stable (limited by max_gp_points=3000)
- **Activation count**: Typically 5-15 activations needed

## Troubleshooting

**Problem**: Trial stops early with "Too much area penalized"
- This means most of the simplex has been explored
- Check if all needles were found before stopping
- May need to adjust penalty_max_radius

**Problem**: Needles found are too close together
- Minima may not be distinguishable enough
- Try regenerating with larger min_distance (0.4 or 0.5)

**Problem**: Very slow progress after many hours
- Normal as search space becomes constrained
- Check if additional needles are being found
- Consider if current needles are sufficient

**Problem**: GPU out of memory
- Reduce max_gp_points (try 2000 or 1500)
- Reduce raw (try 5000)
- Reduce n_restarts (try 50)

