================================================================================
ZOMBI-HOP HYPERPARAMETER OPTIMIZATION - CONSOLIDATED VERSION
================================================================================

WHAT'S INCLUDED:
---------------
✓ Single Python file: hyperparameter_optimization.py (830 lines)
✓ Single README: HYPERPARAMETER_OPTIMIZATION.md (comprehensive guide)
✓ Uses LineBO sampling (like in zombihop_linebo_final.py)
✓ Uses MultiMinimaAckley (from test_functions_torch.py)
✓ Command-line arguments for all functionalities

REMOVED REDUNDANT FILES:
-----------------------
✗ optimize_hyperparameters.py (consolidated)
✗ run_hyperparameter_optimization.py (now --mode optimize)
✗ test_hyperopt_setup.py (now --mode test)
✗ HYPERPARAMETER_OPTIMIZATION_README.md (consolidated)
✗ HYPEROPT_SUMMARY.md (consolidated)
✗ QUICK_START.md (consolidated)

QUICK COMMANDS:
--------------

1. TEST SETUP (2 minutes):
   python hyperparameter_optimization.py --mode test

2. RUN OPTIMIZATION (hours):
   python hyperparameter_optimization.py --mode optimize --num_objectives 10 --max_iterations 100

3. ANALYZE RESULTS (instant):
   python hyperparameter_optimization.py --mode analyze

FULL COMMAND OPTIONS:
--------------------
python hyperparameter_optimization.py \
    --mode [test|optimize|analyze] \
    --num_objectives 10 \
    --dimension 10 \
    --min_minima 2 \
    --max_minima 10 \
    --max_iterations 100 \
    --workers 1 \
    --device cuda \
    --results_dir hyperopt_results \
    --analysis_dir hyperopt_analysis \
    --seed 42

KEY FEATURES:
------------
✓ Synthetic objectives with LineBO (mimics real experimental setup)
✓ Uses MultiMinimaAckley from test_functions_torch.py
✓ Random 2-10 minima per objective, separated by ≥0.5 distance
✓ Same objectives for ALL hyperparameter trials (fair comparison)
✓ Differential evolution optimization
✓ PRIMARY GOAL: Minimize distance to true minima (10x weight)
✓ Secondary goal: Find ≥90% of minima
✓ Automatic analysis with plots and best parameter identification

OUTPUTS:
-------
hyperopt_results/
  ├── result_iter_XXXX.json      # Each iteration
  ├── final_results.json          # Summary
  └── objectives_info.json        # Test objectives

hyperopt_analysis/
  ├── best_hyperparameters.json  # ⭐ USE THIS!
  ├── full_analysis.csv
  ├── optimization_progress.png
  ├── parameter_correlations.png
  └── score_distribution.png

HYPERPARAMETERS OPTIMIZED:
-------------------------
1. top_m_points (2-10)
2. n_restarts (10-50)
3. penalization_threshold (10^-4 to 10^-1)
4. penalty_num_directions (50-200)
5. penalty_max_radius (0.1-0.5)
6. improvement_threshold_mult (1.0-5.0)
7. input_noise_threshold_mult (1.0-5.0)
8. n_consecutive_no_improvements (3-10)

PERFORMANCE METRIC:
------------------
Score = 10.0 × avg_distance + percentage_penalty
        └──────┬──────┘       └────────┬────────┘
          PRIMARY GOAL      Only if < 90% found

Lower score = Better

USAGE EXAMPLE:
-------------
# Quick test
python hyperparameter_optimization.py --mode test

# Small test run (30 min - 1 hour)
python hyperparameter_optimization.py --mode optimize --num_objectives 3 --max_iterations 10

# Full production run (100-200 hours)
python hyperparameter_optimization.py --mode optimize --num_objectives 10 --max_iterations 100

# Analyze results
python hyperparameter_optimization.py --mode analyze

# Use best parameters
import json
with open('hyperopt_analysis/best_hyperparameters.json') as f:
    best_params = json.load(f)
# Apply to your ZoMBI-Hop runs

DOCUMENTATION:
-------------
See HYPERPARAMETER_OPTIMIZATION.md for:
- Complete usage guide
- Detailed parameter descriptions
- Troubleshooting tips
- Advanced customization
- Performance expectations
- Technical details

FILES:
-----
hyperparameter_optimization.py     - Main script (all-in-one)
HYPERPARAMETER_OPTIMIZATION.md     - Complete documentation
README_HYPEROPT.txt                - This file (quick reference)
zombihop_linebo_final.py          - ZoMBI-Hop implementation
test_functions_torch.py            - MultiMinimaAckley function

GET STARTED:
-----------
1. python hyperparameter_optimization.py --mode test
2. python hyperparameter_optimization.py --mode optimize --max_iterations 10
3. Check results in hyperopt_analysis/best_hyperparameters.json

================================================================================

