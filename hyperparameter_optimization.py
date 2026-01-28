"""
Comprehensive hyperparameter optimization for ZoMBI-Hop.

This script optimizes ZoMBI-Hop hyperparameters using differential evolution
on multiple synthetic objectives with LineBO sampling.

Usage:
    # Run full optimization
    python hyperparameter_optimization.py --mode optimize --num_objectives 10 --max_iterations 100
    
    # Quick test
    python hyperparameter_optimization.py --mode test
    
    # Analyze existing results
    python hyperparameter_optimization.py --mode analyze --results_dir hyperopt_results
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
from pathlib import Path
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple, Callable
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

from zombihop_linebo_final import ZoMBIHop, LineBO
from test_functions_torch import MultiMinimaAckley


class SyntheticLineObjective:
    """
    Synthetic objective using MultiMinimaAckley with LineBO sampling.
    Mimics the structure used in zombihop_linebo_final.py.
    """
    
    def __init__(self,
                 dimension: int,
                 num_minima: int,
                 minima_locations: torch.Tensor,
                 amplitudes: torch.Tensor = None,
                 sharpness: torch.Tensor = None,
                 offsets: torch.Tensor = None,
                 global_scale: float = 20.0,
                 exp_scale: float = 0.2,
                 sigma: float = 0.01,
                 num_points_per_line: int = 100,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float64):
        """
        Initialize synthetic objective with LineBO sampling.
        
        Args:
            dimension: Dimensionality
            num_minima: Number of local minima
            minima_locations: (num_minima, dimension) tensor of minima on simplex
            amplitudes, sharpness, offsets: MultiMinimaAckley parameters
            global_scale, exp_scale: Ackley function parameters
            sigma: Noise level
            num_points_per_line: Number of points to sample along lines
            device: Computation device
            dtype: Data type
        """
        self.d = dimension
        self.num_minima = num_minima
        self.sigma = sigma
        self.num_points_per_line = num_points_per_line
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Create MultiMinimaAckley function
        self.ackley_func = MultiMinimaAckley(
            minima_locations=minima_locations,
            amplitudes=amplitudes,
            sharpness=sharpness,
            offsets=offsets,
            global_scale=global_scale,
            exp_scale=exp_scale
        )
        
        # Store minima for distance calculation
        self.minima_locations = minima_locations.to(device=device, dtype=dtype)
    
    def evaluate_line(self, endpoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate objective along a line (LineBO interface).
        
        Args:
            endpoints: (2, d) tensor with line endpoints
            
        Returns:
            x_actual: (num_points_per_line, d) sampled points with noise
            y: (num_points_per_line,) function values
        """
        endpoints = endpoints.to(device=self.device, dtype=self.dtype)
        
        # Sample points along the line
        t_values = torch.linspace(0, 1, self.num_points_per_line,
                                device=self.device, dtype=self.dtype)
        
        # Interpolate between endpoints
        x_expected = (endpoints[0].unsqueeze(0) * (1 - t_values).unsqueeze(1) +
                     endpoints[1].unsqueeze(0) * t_values.unsqueeze(1))
        
        # Add input noise
        x_actual = x_expected + torch.randn_like(x_expected) * self.sigma
        
        # Project back to simplex
        x_actual = ZoMBIHop.proj_simplex(x_actual)
        
        # Evaluate function (negate because MultiMinimaAckley returns values to minimize)
        y_values = -self.ackley_func.evaluate(x_actual)
        
        # Add output noise
        y = y_values + torch.randn_like(y_values) * self.sigma
        
        return x_actual, y
    
    def __call__(self, x_requested: torch.Tensor, bounds: torch.Tensor,
                 acquisition_function=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Objective interface for ZoMBI-Hop - uses LineBO sampler.
        
        Args:
            x_requested: Requested point (d,)
            bounds: Search bounds
            acquisition_function: Acquisition function
            
        Returns:
            x_expected, x_actual, y
        """
        # Project to simplex
        x_tell = ZoMBIHop.proj_simplex(x_requested.unsqueeze(0)).squeeze(0)
        
        # Create LineBO sampler
        linebo = LineBO(
            objective_function=self.evaluate_line,
            dimensions=self.d,
            num_points_per_line=self.num_points_per_line,
            num_lines=20,
            device=str(self.device)
        )
        
        # Sample using LineBO
        x_expected, x_actual, y = linebo.sampler(x_tell, bounds, acquisition_function)
        
        return x_expected, x_actual, y


def generate_synthetic_objectives(
    num_objectives: int = 10,
    dimension: int = 10,
    min_minima: int = 2,
    max_minima: int = 10,
    min_separation: float = 0.3,
    sigma: float = 0.01,
    num_points_per_line: int = 100,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float64,
    seed: int = 42
) -> List[SyntheticLineObjective]:
    """
    Generate multiple synthetic objectives with random parameters.
    
    Args:
        num_objectives: Number of objectives to create
        dimension: Dimensionality
        min_minima: Minimum number of minima per objective
        max_minima: Maximum number of minima per objective
        min_separation: Minimum distance between minima
        sigma: Noise level
        num_points_per_line: Points sampled per line in LineBO
        device: Computation device
        dtype: Data type
        seed: Random seed
        
    Returns:
        List of SyntheticLineObjective instances
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    objectives = []
    
    print(f"Generating {num_objectives} synthetic objectives...")
    print(f"  Dimension: {dimension}")
    print(f"  Minima per objective: {min_minima}-{max_minima}")
    print(f"  Minimum separation: {min_separation}")
    print(f"  Noise level (sigma): {sigma}\n")
    
    for obj_idx in range(num_objectives):
        # Random number of minima
        num_minima = np.random.randint(min_minima, max_minima + 1)
        
        # Random Ackley parameters (easy to hard)
        global_scale = np.random.uniform(10.0, 30.0)  # Higher = harder
        exp_scale = np.random.uniform(0.1, 0.3)       # Lower = harder
        
        # Random per-minimum parameters
        amplitudes = torch.tensor([np.random.uniform(0.5, 1.5) for _ in range(num_minima)],
                                 dtype=dtype, device=device)
        sharpness = torch.tensor([np.random.uniform(3.0, 8.0) for _ in range(num_minima)],
                                dtype=dtype, device=device)
        offsets = torch.zeros(num_minima, dtype=dtype, device=device)
        
        # Generate well-separated minima on simplex
        minima = []
        max_attempts = 10000
        
        for _ in range(num_minima):
            attempts = 0
            while attempts < max_attempts:
                # Generate random simplex point
                candidate = ZoMBIHop.random_simplex(
                    1,
                    torch.zeros(dimension, device=device, dtype=dtype),
                    torch.ones(dimension, device=device, dtype=dtype),
                    S=1.0,
                    device=device,
                    torch_dtype=dtype
                ).squeeze(0)
                
                # Check separation from existing minima
                if len(minima) == 0:
                    minima.append(candidate)
                    break
                
                min_dist = min([torch.norm(candidate - m) for m in minima])
                if min_dist >= min_separation:
                    minima.append(candidate)
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                raise ValueError(f"Could not generate {num_minima} well-separated minima")
        
        minima_tensor = torch.stack(minima)
        
        obj = SyntheticLineObjective(
            dimension=dimension,
            num_minima=num_minima,
            minima_locations=minima_tensor,
            amplitudes=amplitudes,
            sharpness=sharpness,
            offsets=offsets,
            global_scale=global_scale,
            exp_scale=exp_scale,
            sigma=sigma,
            num_points_per_line=num_points_per_line,
            device=device,
            dtype=dtype
        )
        
        objectives.append(obj)
        print(f"  Objective {obj_idx+1}/{num_objectives}: "
              f"{num_minima} minima, global_scale={global_scale:.2f}, exp_scale={exp_scale:.2f}")
    
    print(f"\n✓ Generated {len(objectives)} objectives\n")
    return objectives


def evaluate_hyperparameters(
    hyperparams: np.ndarray,
    objectives: List[SyntheticLineObjective],
    dimension: int,
    bounds: torch.Tensor,
    X_init: torch.Tensor,
    Y_init: torch.Tensor,
    max_iterations_total: int = 50,
    distance_threshold: float = 0.15,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float64
) -> float:
    """
    Evaluate hyperparameter configuration on all objectives.
    
    Returns:
        Negative combined score (for minimization by differential_evolution)
    """
    # Unpack hyperparameters
    (top_m_points, n_restarts, penalization_threshold_log, penalty_num_directions,
     penalty_max_radius, improvement_threshold_mult, input_noise_threshold_mult,
     n_consecutive_no_improvements) = hyperparams
    
    # Convert from log space and ensure integer constraints
    top_m_points = int(top_m_points)
    n_restarts = int(n_restarts)
    penalization_threshold = 10 ** penalization_threshold_log
    penalty_num_directions = int(penalty_num_directions)
    n_consecutive_no_improvements = int(n_consecutive_no_improvements)
    
    print(f"\n{'='*80}")
    print(f"Evaluating hyperparameters:")
    print(f"  top_m_points: {top_m_points}")
    print(f"  n_restarts: {n_restarts}")
    print(f"  penalization_threshold: {penalization_threshold:.2e}")
    print(f"  penalty_num_directions: {penalty_num_directions}")
    print(f"  penalty_max_radius: {penalty_max_radius:.3f}")
    print(f"  improvement_threshold_mult: {improvement_threshold_mult:.2f}")
    print(f"  input_noise_threshold_mult: {input_noise_threshold_mult:.2f}")
    print(f"  n_consecutive_no_improvements: {n_consecutive_no_improvements}")
    print(f"{'='*80}")
    
    all_minima_found = []
    all_distances = []
    all_percentages = []
    
    for obj_idx, objective in enumerate(objectives):
        print(f"\n--- Objective {obj_idx+1}/{len(objectives)} ---")
        
        try:
            # Initialize ZoMBI-Hop
            zombihop = ZoMBIHop(
                objective=objective,
                bounds=bounds,
                X_init_actual=X_init.clone(),
                X_init_expected=X_init.clone(),
                Y_init=Y_init.clone(),
                max_zooms=2,
                max_iterations=max_iterations_total // 2,
                top_m_points=top_m_points,
                n_restarts=n_restarts,
                raw=500,
                penalization_threshold=penalization_threshold,
                penalty_num_directions=penalty_num_directions,
                penalty_max_radius=penalty_max_radius,
                penalty_radius_step=0.01,
                improvement_threshold_noise_mult=improvement_threshold_mult,
                input_noise_threshold_mult=input_noise_threshold_mult,
                n_consecutive_no_improvements=n_consecutive_no_improvements,
                max_gp_points=1000,
                device=device,
                dtype=dtype,
                checkpoint_dir='hyperopt_checkpoints'
            )
            
            # Run optimization
            needles_results, needles, needle_vals, X_all, Y_all = zombihop.run(
                max_activations=2,
                time_limit_hours=None
            )
            
            # Calculate distances to true minima
            true_minima = objective.minima_locations
            found_minima_mask = torch.zeros(objective.num_minima, dtype=torch.bool)
            distances_to_minima = []
            
            for needle in needles:
                # Find closest true minimum
                dists = torch.norm(true_minima - needle.unsqueeze(0), dim=1)
                min_dist = dists.min().item()
                min_idx = dists.argmin().item()
                
                distances_to_minima.append(min_dist)
                
                # Mark as found if within threshold
                if min_dist < distance_threshold:
                    found_minima_mask[min_idx] = True
            
            num_found = found_minima_mask.sum().item()
            percentage_found = (num_found / objective.num_minima) * 100
            avg_distance = np.mean(distances_to_minima) if distances_to_minima else float('inf')
            
            all_minima_found.append(num_found)
            all_distances.extend(distances_to_minima)
            all_percentages.append(percentage_found)
            
            print(f"  Found {num_found}/{objective.num_minima} minima ({percentage_found:.1f}%)")
            print(f"  Average distance: {avg_distance:.4f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_minima_found.append(0)
            all_percentages.append(0.0)
            all_distances.append(10.0)
    
    # Calculate performance metrics
    total_minima = sum([obj.num_minima for obj in objectives])
    total_found = sum(all_minima_found)
    overall_percentage = (total_found / total_minima) * 100
    avg_distance = np.mean(all_distances) if all_distances else float('inf')
    
    # Combined score: PRIORITIZE distance (10x weight)
    percentage_penalty = max(0, (90 - overall_percentage) / 100.0)
    score = 10.0 * avg_distance + percentage_penalty
    
    print(f"\n{'='*80}")
    print(f"OVERALL PERFORMANCE:")
    print(f"  Total minima found: {total_found}/{total_minima} ({overall_percentage:.1f}%)")
    print(f"  Average distance: {avg_distance:.4f}")
    print(f"  Combined score: {score:.4f}")
    print(f"{'='*80}")
    
    return score


def optimize_hyperparameters(
    objectives: List[SyntheticLineObjective],
    dimension: int,
    bounds: torch.Tensor,
    X_init: torch.Tensor,
    Y_init: torch.Tensor,
    max_iterations: int = 100,
    workers: int = 1,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float64,
    results_dir: str = 'hyperopt_results'
) -> Dict:
    """Optimize ZoMBI-Hop hyperparameters using differential evolution."""
    os.makedirs(results_dir, exist_ok=True)
    
    # Define hyperparameter search space
    bounds_hyperparam = [
        (2, 10),           # top_m_points
        (10, 50),          # n_restarts
        (-4, -1),          # log10(penalization_threshold)
        (50, 200),         # penalty_num_directions
        (0.1, 0.5),        # penalty_max_radius
        (1.0, 5.0),        # improvement_threshold_mult
        (1.0, 5.0),        # input_noise_threshold_mult
        (3, 10)            # n_consecutive_no_improvements
    ]
    
    all_results = []
    iteration_count = [0]
    
    def objective_wrapper(hyperparams):
        """Wrapper for differential evolution."""
        iteration_count[0] += 1
        
        score = evaluate_hyperparameters(
            hyperparams=hyperparams,
            objectives=objectives,
            dimension=dimension,
            bounds=bounds,
            X_init=X_init,
            Y_init=Y_init,
            max_iterations_total=50,
            distance_threshold=0.15,
            device=device,
            dtype=dtype
        )
        
        # Save result
        result = {
            'iteration': iteration_count[0],
            'timestamp': datetime.now().isoformat(),
            'hyperparams': {
                'top_m_points': int(hyperparams[0]),
                'n_restarts': int(hyperparams[1]),
                'penalization_threshold': float(10 ** hyperparams[2]),
                'penalty_num_directions': int(hyperparams[3]),
                'penalty_max_radius': float(hyperparams[4]),
                'improvement_threshold_mult': float(hyperparams[5]),
                'input_noise_threshold_mult': float(hyperparams[6]),
                'n_consecutive_no_improvements': int(hyperparams[7])
            },
            'score': float(score)
        }
        
        all_results.append(result)
        
        # Save incrementally
        result_file = Path(results_dir) / f"result_iter_{iteration_count[0]:04d}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return score
    
    print(f"\n{'='*80}")
    print("STARTING DIFFERENTIAL EVOLUTION OPTIMIZATION")
    print(f"{'='*80}\n")
    
    # Run differential evolution
    result = differential_evolution(
        objective_wrapper,
        bounds_hyperparam,
        maxiter=max_iterations,
        popsize=10,
        strategy='best1bin',
        workers=workers,
        updating='deferred' if workers > 1 else 'immediate',
        polish=False,
        disp=True,
        seed=42
    )
    
    # Save final results
    final_results = {
        'best_hyperparams': {
            'top_m_points': int(result.x[0]),
            'n_restarts': int(result.x[1]),
            'penalization_threshold': float(10 ** result.x[2]),
            'penalty_num_directions': int(result.x[3]),
            'penalty_max_radius': float(result.x[4]),
            'improvement_threshold_mult': float(result.x[5]),
            'input_noise_threshold_mult': float(result.x[6]),
            'n_consecutive_no_improvements': int(result.x[7])
        },
        'best_score': float(result.fun),
        'n_iterations': iteration_count[0],
        'all_results': all_results
    }
    
    with open(Path(results_dir) / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best hyperparameters:")
    for key, value in final_results['best_hyperparams'].items():
        print(f"  {key}: {value}")
    print(f"Best score: {final_results['best_score']:.4f}")
    print(f"{'='*80}\n")
    
    return final_results


def analyze_results(results_dir: str = 'hyperopt_results',
                   output_dir: str = 'hyperopt_analysis'):
    """Analyze hyperparameter optimization results."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("ANALYZING RESULTS")
    print(f"{'='*80}\n")
    
    # Load all results
    results = []
    results_path = Path(results_dir)
    
    for json_file in sorted(results_path.glob("result_iter_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded {len(results)} results\n")
    
    # Create DataFrame
    data = []
    for result in results:
        row = {
            'iteration': result['iteration'],
            'score': result['score'],
            **{f"param_{k}": v for k, v in result['hyperparams'].items()}
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Summary statistics
    print("Summary Statistics:")
    print(f"  Best score: {df['score'].min():.4f}")
    print(f"  Worst score: {df['score'].max():.4f}")
    print(f"  Mean score: {df['score'].mean():.4f}")
    print(f"  Median score: {df['score'].median():.4f}")
    print(f"  Std score: {df['score'].std():.4f}\n")
    
    # Best parameters
    best_idx = df['score'].idxmin()
    best_result = df.loc[best_idx]
    
    print("Best Parameters:")
    param_cols = [col for col in df.columns if col.startswith('param_')]
    for col in sorted(param_cols):
        param_name = col.replace('param_', '')
        print(f"  {param_name}: {best_result[col]}")
    print(f"  Score: {best_result['score']:.4f}\n")
    
    # Save best parameters
    best_params = {col.replace('param_', ''): best_result[col] for col in param_cols}
    with open(Path(output_dir) / 'best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best parameters to {output_dir}/best_hyperparameters.json\n")
    
    # Create visualizations
    sns.set_style('whitegrid')
    
    # 1. Score over iterations
    plt.figure(figsize=(12, 6))
    plt.plot(df['iteration'], df['score'], 'o-', alpha=0.6, markersize=4)
    plt.axhline(df['score'].min(), color='r', linestyle='--',
                label=f'Best: {df["score"].min():.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('Score (lower is better)')
    plt.title('Hyperparameter Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'optimization_progress.png', dpi=300)
    print(f"Saved: {output_dir}/optimization_progress.png")
    plt.close()
    
    # 2. Parameter correlation with score
    param_cols = [col for col in df.columns if col.startswith('param_')]
    if param_cols:
        corr_with_score = df[param_cols + ['score']].corr()['score'].drop('score').sort_values()
        
        plt.figure(figsize=(10, 6))
        corr_with_score.plot(kind='barh', color='steelblue')
        plt.xlabel('Correlation with Score')
        plt.title('Parameter Correlations with Score\n(Negative = helps reduce score)')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'parameter_correlations.png', dpi=300)
        print(f"Saved: {output_dir}/parameter_correlations.png")
        plt.close()
    
    # 3. Distribution of scores
    plt.figure(figsize=(10, 6))
    plt.hist(df['score'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(df['score'].min(), color='r', linestyle='--', linewidth=2,
                label=f'Best: {df["score"].min():.4f}')
    plt.axvline(df['score'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {df["score"].median():.4f}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'score_distribution.png', dpi=300)
    print(f"Saved: {output_dir}/score_distribution.png")
    plt.close()
    
    # Save full analysis
    df_sorted = df.sort_values('score')
    df_sorted.to_csv(Path(output_dir) / 'full_analysis.csv', index=False)
    print(f"Saved: {output_dir}/full_analysis.csv")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


def test_setup(dimension: int = 10, device: str = 'cuda'):
    """Quick test to verify setup works."""
    print("="*80)
    print("TESTING SETUP")
    print("="*80)
    
    dtype = torch.float64
    
    print(f"\nDevice: {device}")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Generate test objective
    print("\n1. Generating test objective...")
    objectives = generate_synthetic_objectives(
        num_objectives=1,
        dimension=dimension,
        min_minima=2,
        max_minima=3,
        device=device,
        dtype=dtype
    )
    objective = objectives[0]
    print(f"   ✓ Generated objective with {objective.num_minima} minima")
    
    # Test evaluation
    print("\n2. Testing LineBO evaluation...")
    bounds = torch.stack([
        torch.zeros(dimension, device=device, dtype=dtype),
        torch.ones(dimension, device=device, dtype=dtype)
    ])
    
    test_point = ZoMBIHop.random_simplex(1, bounds[0], bounds[1], S=1.0,
                                        device=device, torch_dtype=dtype).squeeze(0)
    
    x_expected, x_actual, y = objective(test_point, bounds, None)
    print(f"   ✓ LineBO evaluation successful")
    print(f"     Returned {x_actual.shape[0]} points")
    
    # Test ZoMBI-Hop initialization
    print("\n3. Testing ZoMBI-Hop initialization...")
    n_init = 5
    X_init = ZoMBIHop.random_simplex(n_init, bounds[0], bounds[1], S=1.0,
                                    device=device, torch_dtype=dtype)
    Y_init = torch.randn(n_init, 1, device=device, dtype=dtype)
    
    zombihop = ZoMBIHop(
        objective=objective,
        bounds=bounds,
        X_init_actual=X_init.clone(),
        X_init_expected=X_init.clone(),
        Y_init=Y_init.clone(),
        max_zooms=1,
        max_iterations=2,
        device=device,
        dtype=dtype,
        checkpoint_dir='test_checkpoints'
    )
    print(f"   ✓ ZoMBI-Hop initialized (UUID: {zombihop.run_uuid})")
    
    print("\n" + "="*80)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("="*80)
    print("\nReady to run full optimization!")


def main():
    """Main execution with command-line arguments."""
    parser = argparse.ArgumentParser(
        description='ZoMBI-Hop Hyperparameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python hyperparameter_optimization.py --mode test
  
  # Run optimization
  python hyperparameter_optimization.py --mode optimize --num_objectives 10 --max_iterations 100
  
  # Analyze results
  python hyperparameter_optimization.py --mode analyze --results_dir hyperopt_results
        """
    )
    
    parser.add_argument('--mode', type=str, default='optimize',
                       choices=['test', 'optimize', 'analyze'],
                       help='Mode: test setup, run optimization, or analyze results')
    parser.add_argument('--num_objectives', type=int, default=10,
                       help='Number of test objectives (default: 10)')
    parser.add_argument('--dimension', type=int, default=10,
                       help='Problem dimensionality (default: 10)')
    parser.add_argument('--min_minima', type=int, default=2,
                       help='Minimum minima per objective (default: 2)')
    parser.add_argument('--max_minima', type=int, default=10,
                       help='Maximum minima per objective (default: 10)')
    parser.add_argument('--min_separation', type=float, default=0.3,
                       help='Minimum distance between minima (default: 0.3)')
    parser.add_argument('--max_iterations', type=int, default=100,
                       help='Max differential evolution iterations (default: 100)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--results_dir', type=str, default='hyperopt_results',
                       help='Results directory (default: hyperopt_results)')
    parser.add_argument('--analysis_dir', type=str, default='hyperopt_analysis',
                       help='Analysis output directory (default: hyperopt_analysis)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.mode == 'test':
        test_setup(dimension=args.dimension, device=args.device)
        return
    
    elif args.mode == 'analyze':
        analyze_results(results_dir=args.results_dir, output_dir=args.analysis_dir)
        return
    
    elif args.mode == 'optimize':
        # Configuration
        device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
        dtype = torch.float64
        
        print("="*80)
        print("ZOMBI-HOP HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Dimension: {args.dimension}")
        print(f"  Number of objectives: {args.num_objectives}")
        print(f"  Minima per objective: {args.min_minima}-{args.max_minima}")
        print(f"  Minimum separation: {args.min_separation}")
        print(f"  Max DE iterations: {args.max_iterations}")
        print(f"  Device: {device}")
        print(f"  Workers: {args.workers}")
        print(f"  Random seed: {args.seed}\n")
        
        # Generate objectives
        objectives = generate_synthetic_objectives(
            num_objectives=args.num_objectives,
            dimension=args.dimension,
            min_minima=args.min_minima,
            max_minima=args.max_minima,
            min_separation=args.min_separation,
            device=device,
            dtype=dtype,
            seed=args.seed
        )
        
        # Save objectives info
        os.makedirs(args.results_dir, exist_ok=True)
        objectives_info = []
        for i, obj in enumerate(objectives):
            info = {
                'objective_id': i,
                'num_minima': obj.num_minima,
                'minima_locations': obj.minima_locations.cpu().tolist()
            }
            objectives_info.append(info)
        
        with open(Path(args.results_dir) / 'objectives_info.json', 'w') as f:
            json.dump(objectives_info, f, indent=2)
        
        # Create initial points
        bounds = torch.stack([
            torch.zeros(args.dimension, device=device, dtype=dtype),
            torch.ones(args.dimension, device=device, dtype=dtype)
        ])
        
        n_init = 20
        X_init = ZoMBIHop.random_simplex(n_init, bounds[0], bounds[1], S=1.0,
                                        device=device, torch_dtype=dtype)
        Y_init = torch.randn(n_init, 1, device=device, dtype=dtype)
        
        # Run optimization
        results = optimize_hyperparameters(
            objectives=objectives,
            dimension=args.dimension,
            bounds=bounds,
            X_init=X_init,
            Y_init=Y_init,
            max_iterations=args.max_iterations,
            workers=args.workers,
            device=device,
            dtype=dtype,
            results_dir=args.results_dir
        )
        
        # Analyze results
        analyze_results(results_dir=args.results_dir, output_dir=args.analysis_dir)
        
        print("\n" + "="*80)
        print("ALL DONE!")
        print("="*80)
        print(f"\nResults: {args.results_dir}/")
        print(f"Analysis: {args.analysis_dir}/")
        print(f"Best params: {args.analysis_dir}/best_hyperparameters.json")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()

