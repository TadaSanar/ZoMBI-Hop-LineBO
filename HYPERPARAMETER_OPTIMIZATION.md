# ZoMBI-Hop Hyperparameter Optimization

Comprehensive tool for optimizing ZoMBI-Hop hyperparameters using differential evolution on synthetic objectives with LineBO sampling.

## Quick Start

```bash
# 1. Test setup (2 minutes)
python hyperparameter_optimization.py --mode test

# 2. Run optimization (hours to days)
python hyperparameter_optimization.py --mode optimize --num_objectives 10 --max_iterations 100

# 3. Analyze results
python hyperparameter_optimization.py --mode analyze
```

## Features

### Synthetic Objectives
- Uses `MultiMinimaAckley` from `test_functions_torch.py`
- Samples points using `LineBO` (line-based Bayesian optimization)
- Mimics real experimental setup in `zombihop_linebo_final.py`
- **Key properties:**
  - Random number of minima (2-10 per objective)
  - Random difficulty parameters (easy to hard)
  - All minima separated by ≥0.5 distance
  - Consistent noise level (sigma=0.01)
  - **Same objectives for all hyperparameter trials** (fair comparison)

### Optimization Strategy
Optimizes 8 hyperparameters using differential evolution:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `top_m_points` | 2-10 | Points for zoom bounds determination |
| `n_restarts` | 10-50 | Acquisition optimization restarts |
| `penalization_threshold` | 10^-4 to 10^-1 | Gradient threshold for penalty radius |
| `penalty_num_directions` | 50-200 | Directions sampled for penalty calculation |
| `penalty_max_radius` | 0.1-0.5 | Maximum penalty radius around needles |
| `improvement_threshold_mult` | 1.0-5.0 | Output noise threshold multiplier |
| `input_noise_threshold_mult` | 1.0-5.0 | Input noise threshold multiplier |
| `n_consecutive_no_improvements` | 3-10 | Iterations before declaring needle |

### Performance Metric
```
Combined Score = 10.0 × average_distance + percentage_penalty
                 └─────────┬─────────┘      └────────┬────────┘
                      PRIMARY GOAL        Only if < 90% found
```

**Goal:** Find ≥90% of minima with **minimum distance** to true minima

## Usage

### Mode: Test
Quick verification that everything works:
```bash
python hyperparameter_optimization.py --mode test --dimension 10
```

**What it does:**
- Generates 1 test objective
- Tests LineBO evaluation
- Tests ZoMBI-Hop initialization
- Verifies CUDA availability

**Expected output:**
```
✓ Generated objective with 3 minima
✓ LineBO evaluation successful
✓ ZoMBI-Hop initialized
✓✓✓ ALL TESTS PASSED ✓✓✓
```

### Mode: Optimize
Run full hyperparameter optimization:
```bash
python hyperparameter_optimization.py \
    --mode optimize \
    --num_objectives 10 \
    --dimension 10 \
    --min_minima 2 \
    --max_minima 10 \
    --max_iterations 100 \
    --workers 1 \
    --device cuda \
    --results_dir hyperopt_results \
    --seed 42
```

**Parameters:**
- `--num_objectives`: Number of test objectives (default: 10)
- `--dimension`: Problem dimensionality (default: 10)
- `--min_minima`: Minimum minima per objective (default: 2)
- `--max_minima`: Maximum minima per objective (default: 10)
- `--max_iterations`: Differential evolution iterations (default: 100)
- `--workers`: Parallel workers (default: 1, increase if multiple GPUs)
- `--device`: 'cuda' or 'cpu' (default: cuda)
- `--results_dir`: Output directory (default: hyperopt_results)
- `--seed`: Random seed for reproducibility (default: 42)

**Output during run:**
```
Generating 10 synthetic objectives...
  Objective 1/10: 7 minima, global_scale=18.45, exp_scale=0.22
  ...
Evaluating hyperparameters:
  top_m_points: 6
  n_restarts: 28
  ...
Found 6/7 minima (85.7%)
Average distance: 0.0823
```

### Mode: Analyze
Analyze existing results:
```bash
python hyperparameter_optimization.py \
    --mode analyze \
    --results_dir hyperopt_results \
    --analysis_dir hyperopt_analysis
```

**What it does:**
- Loads all iteration results
- Identifies best hyperparameters
- Shows parameter correlations
- Creates visualization plots
- Saves analysis to CSV

## Output Structure

### Results Directory (`hyperopt_results/`)
```
hyperopt_results/
├── result_iter_0001.json    # Individual iteration results
├── result_iter_0002.json
├── ...
├── final_results.json        # Summary of optimization
└── objectives_info.json      # Details of test objectives
```

**Example `result_iter_0001.json`:**
```json
{
  "iteration": 1,
  "timestamp": "2024-01-15T10:30:00",
  "hyperparams": {
    "top_m_points": 6,
    "n_restarts": 28,
    "penalization_threshold": 0.001,
    ...
  },
  "score": 0.8234
}
```

### Analysis Directory (`hyperopt_analysis/`)
```
hyperopt_analysis/
├── best_hyperparameters.json     # ⭐ Best params (use this!)
├── full_analysis.csv              # All iteration data
├── optimization_progress.png      # Score over iterations
├── parameter_correlations.png     # Which params help most
└── score_distribution.png         # Performance distribution
```

**Example `best_hyperparameters.json`:**
```json
{
  "top_m_points": 5,
  "n_restarts": 30,
  "penalization_threshold": 0.001,
  "penalty_num_directions": 100,
  "penalty_max_radius": 0.3,
  "improvement_threshold_mult": 2.0,
  "input_noise_threshold_mult": 3.0,
  "n_consecutive_no_improvements": 5
}
```

## Using Best Hyperparameters

After optimization, apply the best parameters to your real problems:

```python
from zombihop_linebo_final import ZoMBIHop
import json
import torch

# Load best hyperparameters
with open('hyperopt_analysis/best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)

# Use them in your optimization
zombihop = ZoMBIHop(
    objective=your_objective,
    bounds=your_bounds,
    X_init_actual=X_init,
    X_init_expected=X_init,
    Y_init=Y_init,
    max_zooms=3,
    max_iterations=10,
    top_m_points=best_params['top_m_points'],
    n_restarts=best_params['n_restarts'],
    penalization_threshold=best_params['penalization_threshold'],
    penalty_num_directions=best_params['penalty_num_directions'],
    penalty_max_radius=best_params['penalty_max_radius'],
    improvement_threshold_noise_mult=best_params['improvement_threshold_mult'],
    input_noise_threshold_mult=best_params['input_noise_threshold_mult'],
    n_consecutive_no_improvements=best_params['n_consecutive_no_improvements'],
    device='cuda',
    dtype=torch.float64
)

# Run optimization
results = zombihop.run(max_activations=5)
```

## Understanding Results

### Score Components
- **Lower score = Better** (minimization problem)
- **Distance** (×10 weight): Average distance to nearest true minimum
- **Percentage penalty**: Only adds penalty if < 90% of minima found

### Success Criteria
Good hyperparameters achieve:
- ✓ ≥90% of minima found across all objectives
- ✓ **Low average distance** (< 0.15, primary goal)
- ✓ Consistent performance across different objectives

### Visualization Plots

**1. Optimization Progress** (`optimization_progress.png`)
- Score decreasing over iterations
- Red line shows best score achieved
- Plateau indicates convergence

**2. Parameter Correlations** (`parameter_correlations.png`)
- Negative correlation = parameter helps reduce score
- Shows which parameters are most important
- Guides manual fine-tuning

**3. Score Distribution** (`score_distribution.png`)
- Shows variability in performance
- Narrow distribution = consistent search space
- Best score highlighted in red

## Performance

### Expected Runtime
On modern GPU (e.g., RTX 3090):
- **Per objective evaluation**: 5-10 minutes
- **Per DE iteration**: 1-2 hours (10 objectives)
- **Full optimization (100 iterations)**: 100-200 hours

### Recommendations
1. Start with **20-30 iterations** to test (4-6 hours)
2. Use **parallel workers** if multiple GPUs available
3. Monitor early progress - may converge before max iterations
4. **Can interrupt** (Ctrl+C) - progress is saved incrementally

## Troubleshooting

### CUDA Out of Memory
```python
# In evaluate_hyperparameters(), reduce:
max_gp_points=500  # Instead of 1000
```

### Optimization Too Slow
```bash
# Reduce iterations
python hyperparameter_optimization.py --mode optimize --max_iterations 20

# Or enable parallel processing (if multiple GPUs)
python hyperparameter_optimization.py --mode optimize --workers 4
```

### No Minima Found
- Check if objectives are too difficult
- Increase noise tolerance in code
- Verify minima are properly on simplex
- Try easier parameter ranges (lower global_scale)

### Poor Convergence
- Increase differential evolution iterations
- Expand hyperparameter search ranges
- Check if objectives are too diverse

## Advanced Usage

### Custom Objective Difficulty
Edit `generate_synthetic_objectives()` in `hyperparameter_optimization.py`:
```python
# Make objectives harder
global_scale = np.random.uniform(20.0, 40.0)  # Higher = harder
exp_scale = np.random.uniform(0.05, 0.15)     # Lower = harder
```

### Custom Hyperparameter Ranges
Edit `bounds_hyperparam` in `optimize_hyperparameters()`:
```python
bounds_hyperparam = [
    (2, 15),      # Wider range for top_m_points
    (10, 100),    # More restarts allowed
    (-5, -1),     # Wider penalization threshold range
    ...
]
```

### Different Noise Levels
```bash
# Edit sigma in generate_synthetic_objectives():
sigma = 0.02  # More noise (harder)
```

## Technical Details

### LineBO Integration
Unlike simple point evaluation, this uses **LineBO** sampling:
1. Draws random zero-sum directions from current point
2. Finds line segments on simplex
3. Samples multiple points along selected line
4. Adds input/output noise
5. Returns actual sampled points

This mimics real experimental setups where you request a point but get noisy measurements along a line.

### MultiMinimaAckley
Uses the `MultiMinimaAckley` class from `test_functions_torch.py`:
- Multiple distinct basins of attraction
- Configurable per-minimum parameters (amplitude, sharpness, offset)
- Works on simplex domains
- Numerically stable

### Differential Evolution
Standard scipy implementation:
- Population-based global optimizer
- Strategy: 'best1bin'
- No gradient information needed
- Robust to local minima
- Can use parallel workers

## Files

### Main Files
- `hyperparameter_optimization.py` - Complete optimization tool
- `zombihop_linebo_final.py` - ZoMBI-Hop implementation
- `test_functions_torch.py` - MultiMinimaAckley function
- `HYPERPARAMETER_OPTIMIZATION.md` - This file

### Generated Files
- `hyperopt_results/` - Optimization results
- `hyperopt_analysis/` - Analysis outputs
- `hyperopt_checkpoints/` - ZoMBI-Hop checkpoints (auto-cleaned)

## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- BoTorch
- GPyTorch
- SciPy
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Examples

### Minimal Test Run
```bash
# Quick test with 3 objectives, 10 iterations
python hyperparameter_optimization.py \
    --mode optimize \
    --num_objectives 3 \
    --max_iterations 10 \
    --dimension 5
```

### Production Run
```bash
# Full optimization with 10 objectives, 100 iterations
python hyperparameter_optimization.py \
    --mode optimize \
    --num_objectives 10 \
    --max_iterations 100 \
    --dimension 10 \
    --workers 1 \
    --device cuda
```

### Analysis Only
```bash
# Analyze existing results
python hyperparameter_optimization.py \
    --mode analyze \
    --results_dir my_results \
    --analysis_dir my_analysis
```

## Next Steps

1. **Run test** to verify setup
2. **Start with short run** (10-20 iterations) to check progress
3. **Monitor results** in `hyperopt_results/`
4. **Run full optimization** when confident
5. **Apply best parameters** to your real problems
6. **Validate** on different problem instances

## Citation

If you use this hyperparameter optimization in your research, please cite the ZoMBI-Hop paper.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments in `hyperparameter_optimization.py`
3. Inspect intermediate results in `hyperopt_results/`
4. Modify code as needed for your specific use case

