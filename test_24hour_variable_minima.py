"""
24-Hour Hard Trial for ZoMBIHop with Variable Number of Distinguishable Minima

⚠️ IMPORTANT: This script runs EXACTLY ONE trial - it does NOT loop or repeat.

This script runs a challenging optimization task designed to test ZoMBIHop's ability
to find multiple well-separated minima over an extended period.

- Creates ONE objective function with configurable number of minima
- Initializes ONE ZoMBIHop optimizer instance
- Executes ONE 24-hour optimization run
- No loops, no repeats, no multiple runs

To run multiple trials, execute this script multiple times.

Usage:
    python test_24hour_variable_minima.py [num_minima] [dimensions] [time_hours]
    
Examples:
    python test_24hour_variable_minima.py 5 10 24    # 5 minima in 10D for 24 hours
    python test_24hour_variable_minima.py 3 8 12     # 3 minima in 8D for 12 hours
    python test_24hour_variable_minima.py            # Use defaults (3 minima, 10D, 24h)
"""

import torch
import json
import os
import time
import uuid
from pathlib import Path
from zombihop_linebo_final import ZoMBIHop, LineBO

# Import test functions
try:
    from test_functions_torch import MultiMinimaAckley
except ImportError:
    print("Warning: test_functions_torch not found. Make sure it's in the same directory.")
    raise

# Default Configuration (can be overridden by command line arguments)
DEFAULT_DIMENSIONS = 10
DEFAULT_NUM_MINIMA = 3
DEFAULT_TIME_LIMIT_HOURS = 24.0
NUM_EXPERIMENTS = 24  # Number of points sampled per line
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Noise parameters - very low noise for near-deterministic behavior
# INPUT_SIGMA2 = 1e-10  # Nearly no input noise (expected vs actual)
# OUTPUT_SIGMA2 = 1e-10  # Nearly no output noise (objective function)
OUTPUT_SIGMA2 = 6.5e-2  # Output noise variance
INPUT_SIGMA2 = 4.3036e-3  # Input noise variance


# Hard parameters for challenging optimization
# HARD_PARAMS = {
#     'global_scale': 6.0,      # Lower scale makes the landscape harder
#     'exp_scale': 0.3,         # Higher exp_scale makes basins narrower
#     'sharpness': 9.0,         # Higher sharpness creates sharper peaks
# }

# Easy parameters for easier optimization
HARD_PARAMS = {
    'global_scale': 20.0,
    'exp_scale': 0.2,
    'sharpness': 5.0,
}


def generate_distinguishable_minima(dimensions, num_minima, min_distance=0.5, device='cuda'):
    """
    Generate distinguishable minima locations on the simplex.
    
    Args:
        dimensions: Number of dimensions
        num_minima: Number of minima to generate
        min_distance: Minimum L2 distance between any two minima
        device: Device to create tensors on
        
    Returns:
        Tensor of shape (num_minima, dimensions) with minima locations
    """
    print(f"Generating {num_minima} distinguishable minima in {dimensions}D with min_distance={min_distance}...")
    
    minima = []
    max_attempts = 10000
    
    for i in range(num_minima):
        attempts = 0
        while attempts < max_attempts:
            # Generate random point on simplex
            x = torch.rand(dimensions, device=device, dtype=torch.float64)
            x = x / x.sum()
            
            # Check distance to all existing minima
            if len(minima) == 0:
                minima.append(x)
                break
            
            # Calculate distances to existing minima
            distances = torch.stack([torch.norm(x - m) for m in minima])
            
            if torch.all(distances >= min_distance):
                minima.append(x)
                print(f"  Minimum {i+1}/{num_minima}: min_dist to others = {distances.min().item():.4f}")
                break
            
            attempts += 1
        
        if attempts >= max_attempts:
            raise ValueError(f"Could not generate {num_minima} distinguishable minima with min_distance={min_distance}")
    
    minima_tensor = torch.stack(minima)
    
    # Verify all pairwise distances
    print("\nVerifying pairwise distances:")
    for i in range(len(minima)):
        for j in range(i+1, len(minima)):
            dist = torch.norm(minima[i] - minima[j]).item()
            print(f"  Distance between minimum {i+1} and {j+1}: {dist:.4f}")
    
    print(f"\nSuccessfully generated {num_minima} distinguishable minima!")
    return minima_tensor


def create_multi_minima_objective(minima_locs, params, device='cuda'):
    """
    Creates a challenging objective function with variable number of minima.
    
    Args:
        minima_locs: Tensor of shape (num_minima, dimensions) with minima locations
        params: Dictionary with 'global_scale', 'exp_scale', and 'sharpness'
        device: Device to use
        
    Returns:
        Objective function compatible with LineBO
    """
    device = torch.device(device)
    num_minima = minima_locs.shape[0]
    
    # Create amplitudes and sharpness tensors
    amplitudes = torch.ones(num_minima, device=device, dtype=torch.float64)
    sharpness = params['sharpness'] * torch.ones(num_minima, device=device, dtype=torch.float64)
    
    # Create the multi-minima Ackley function
    func = MultiMinimaAckley(
        minima_locs,
        amplitudes=amplitudes,
        sharpness=sharpness,
        global_scale=params['global_scale'],
        exp_scale=params['exp_scale']
    )
    
    def _line_objective(endpoints):
        """
        Objective function that samples along a line and returns noisy observations.
        
        Args:
            endpoints: Tensor of shape (2, d) representing line endpoints
            
        Returns:
            Tuple of (x_actual, y) where:
                x_actual: Tensor of shape (num_experiments, d) - actual sampled points
                y: Tensor of shape (num_experiments,) - objective values
        """
        if not torch.is_tensor(endpoints):
            endpoints = torch.tensor(endpoints, device=device, dtype=torch.float64)
        if endpoints.dim() == 2:
            endpoints = endpoints.unsqueeze(0)
        
        # Generate requested points along line
        t_values = torch.linspace(0, 1, NUM_EXPERIMENTS, device=device, dtype=torch.float64)
        x_requested = (1 - t_values).unsqueeze(1) * endpoints[0, 0].unsqueeze(0) + \
                     t_values.unsqueeze(1) * endpoints[0, 1].unsqueeze(0)
        
        # Add input noise
        input_noise = torch.randn_like(x_requested) * torch.sqrt(torch.tensor(INPUT_SIGMA2, dtype=torch.float64, device=device))
        x_actual = x_requested + input_noise
        
        # Project back to simplex
        x_actual = torch.clamp(x_actual, min=0.0)
        x_actual = x_actual / x_actual.sum(dim=1, keepdim=True)
        
        # Evaluate function (negate for maximization) and add output noise
        y = -func.evaluate(x_actual)
        output_noise = torch.randn_like(y) * torch.sqrt(torch.tensor(OUTPUT_SIGMA2, dtype=torch.float64, device=device))
        y = y + output_noise
        
        return x_actual, y
    
    def _evaluate_needles(needles):
        """
        Evaluates how well the found needles match the known minima locations.
        
        Args:
            needles: Tensor of shape (num_needles, d) containing needle locations
            
        Returns:
            List of distances from each minimum to its closest needle, or None if insufficient needles
        """
        if needles.shape[0] < len(minima_locs):
            print(f"\n⚠️  Warning: Only {needles.shape[0]} needles found, need at least {len(minima_locs)}")
            return None
        
        # Calculate pairwise distances between needles and minima
        needles = needles.to(device=device)
        minima = minima_locs.to(device=device)
        
        distances = torch.cdist(needles, minima)
        
        # For each minimum, find the closest needle using greedy assignment
        min_distances = []
        remaining_needles = list(range(len(needles)))
        
        for i in range(len(minima)):
            if not remaining_needles:
                return None
            
            # Find closest remaining needle to this minimum
            min_dist = float('inf')
            best_needle_idx = -1
            
            for needle_idx in remaining_needles:
                dist = distances[needle_idx, i].item()
                if dist < min_dist:
                    min_dist = dist
                    best_needle_idx = needle_idx
            
            if best_needle_idx >= 0:
                min_distances.append(min_dist)
                remaining_needles.remove(best_needle_idx)
            else:
                return None
        
        # Print evaluation results
        print("\n" + "="*80)
        print("🎯 NEEDLE EVALUATION RESULTS 🎯")
        print("="*80)
        for i, dist in enumerate(min_distances):
            status = "✅" if dist < 0.05 else "⚠️" if dist < 0.10 else "❌"
            print(f"  {status} Distance from minimum {i+1:2d} to closest needle: {dist:.6f}")
        print("-"*80)
        print(f"  Mean distance: {sum(min_distances)/len(min_distances):.6f}")
        print(f"  Max distance:  {max(min_distances):.6f}")
        print(f"  Needles found: {needles.shape[0]}")
        print("="*80 + "\n")
        
        return min_distances
    
    # Attach evaluator to objective function
    _line_objective.evaluate_needles = _evaluate_needles
    _line_objective.minima_locs = minima_locs
    _line_objective.func = func
    
    return _line_objective


def find_trial_directory_by_uuid(uuid):
    """
    Find the trial directory containing a specific UUID.
    
    Args:
        uuid: The UUID to search for
        
    Returns:
        Path to the trial directory, or None if not found
    """
    trial_dirs = [d for d in Path('.').iterdir() if d.is_dir() and d.name.startswith('trial_')]
    
    for td in trial_dirs:
        checkpoints_dir = td / 'checkpoints'
        if checkpoints_dir.exists():
            run_dir = checkpoints_dir / f'run_{uuid}'
            if run_dir.exists():
                return td
    
    return None


def run_variable_trial(num_minima=None, dimensions=None, time_limit_hours=None, resume_uuid=None):
    """
    Run a SINGLE optimization trial with variable number of minima.
    
    This function runs exactly ONE trial - it does NOT loop or repeat.
    
    Args:
        num_minima: Number of minima to find (default: DEFAULT_NUM_MINIMA)
        dimensions: Number of dimensions (default: DEFAULT_DIMENSIONS)
        time_limit_hours: Time limit in hours (default: DEFAULT_TIME_LIMIT_HOURS)
        resume_uuid: Optional UUID to resume from a previous run
    """
    # Use defaults if not provided
    if num_minima is None:
        num_minima = DEFAULT_NUM_MINIMA
    if dimensions is None:
        dimensions = DEFAULT_DIMENSIONS
    if time_limit_hours is None:
        time_limit_hours = DEFAULT_TIME_LIMIT_HOURS
    
    print("="*80)
    print(f"ZOMBIHOP TRIAL: {num_minima} Minima in {dimensions}D")
    print("="*80)
    print(f"Dimensions: {dimensions}")
    print(f"Number of minima: {num_minima}")
    print(f"Time limit: {time_limit_hours} hours")
    print(f"Device: {DEVICE}")
    print(f"⚠️  THIS IS A SINGLE RUN (not repeated)")
    print("="*80 + "\n")
    
    # Set up device
    device = torch.device(DEVICE)
    
    # Create bounds tensor
    bounds = torch.zeros((2, dimensions), device=device, dtype=torch.float64)
    bounds[0] = 0.0  # Lower bounds
    bounds[1] = 1.0  # Upper bounds
    
    # Create trial directory with descriptive name
    trial_dir = Path(f'trial_{num_minima}minima_{dimensions}d_{int(time_limit_hours)}h')
    trial_dir.mkdir(exist_ok=True)
    
    if resume_uuid is None:
        # Generate distinguishable minima
        minima_locs = generate_distinguishable_minima(
            dimensions=dimensions,
            num_minima=num_minima,
            min_distance=0.3,  # Ensure good separation
            device=device
        )
        
        # Save minima locations
        minima_file = trial_dir / 'minima_locations.pt'
        torch.save(minima_locs, minima_file)
        print(f"Saved minima locations to {minima_file}\n")
        
        # Create objective function
        objective_fn = create_multi_minima_objective(minima_locs, HARD_PARAMS, device=device)
        
        # Create LineBO sampler
        linebo = LineBO(
            objective_fn,
            dimensions=dimensions,
            num_points_per_line=50,
            num_lines=100,
            device=device
        )
        
        # Generate initial random points
        print("Generating initial random points...")
        num_random_points = 3
        random_points = ZoMBIHop.random_simplex(num_random_points, bounds[0], bounds[1], device=device)
        
        X_init_actual = torch.empty((0, dimensions), device=device, dtype=torch.float64)
        X_init_expected = torch.empty((0, dimensions), device=device, dtype=torch.float64)
        Y_init = torch.empty((0, 1), device=device, dtype=torch.float64)
        
        for i, random_point in enumerate(random_points):
            print(f"  Sampling initial point {i+1}/{num_random_points}...")
            x_requested, x_actual, y = linebo.sampler(random_point, bounds=bounds)
            X_init_actual = torch.cat([X_init_actual, x_actual], dim=0)
            X_init_expected = torch.cat([X_init_expected, x_requested], dim=0)
            Y_init = torch.cat([Y_init, y.unsqueeze(1)], dim=0)
        
        print(f"Initial data: {X_init_actual.shape[0]} points\n")
        
        # Create ZoMBIHop optimizer with challenging parameters
        print("Initializing ZoMBIHop optimizer...")
        zombihop = ZoMBIHop(
            objective=linebo.sampler,
            bounds=bounds,
            X_init_actual=X_init_actual,
            X_init_expected=X_init_expected,
            Y_init=Y_init,
            # Hard optimization parameters
            penalization_threshold=1e-3,
            improvement_threshold_noise_mult=1.5,
            input_noise_threshold_mult=3.4,
            n_consecutive_no_improvements=5,
            top_m_points=3,
            max_zooms=3,
            max_iterations=10,
            n_restarts=100,
            raw=10000,
            penalty_num_directions=100,
            penalty_max_radius=0.3,
            penalty_radius_step=0.01,
            max_gp_points=3000,
            device=device,
            checkpoint_dir=str(trial_dir / 'checkpoints')
        )
        
        print(f"Starting new trial with UUID: {zombihop.run_uuid}\n")
        
        # Save trial metadata
        metadata = {
            'run_uuid': zombihop.run_uuid,
            'dimensions': dimensions,
            'num_minima': num_minima,
            'time_limit_hours': time_limit_hours,
            'hard_params': HARD_PARAMS,
            'input_sigma2': INPUT_SIGMA2,
            'output_sigma2': OUTPUT_SIGMA2,
            'start_time': time.time(),
        }
        
        with open(trial_dir / 'trial_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    else:
        # Resume from saved run - need to find the correct trial directory
        print(f"Resuming trial from UUID: {resume_uuid}\n")
        
        # Search for the trial directory containing this UUID
        found_trial_dir = find_trial_directory_by_uuid(resume_uuid)
        
        if found_trial_dir is None:
            print(f"Error: Could not find trial directory containing UUID {resume_uuid}")
            print("Available trial directories:")
            trial_dirs = [d for d in Path('.').iterdir() if d.is_dir() and d.name.startswith('trial_')]
            for td in trial_dirs:
                print(f"  {td.name}")
            return
        
        print(f"Found trial in directory: {found_trial_dir.name}")
        trial_dir = found_trial_dir
        
        # Load metadata to get the original parameters
        metadata_file = trial_dir / 'trial_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            dimensions = metadata.get('dimensions', dimensions)
            num_minima = metadata.get('num_minima', num_minima)
            time_limit_hours = metadata.get('time_limit_hours', time_limit_hours)
            print(f"Loaded original parameters: {num_minima} minima, {dimensions}D, {time_limit_hours}h")
        else:
            print("Warning: trial_metadata.json not found, using default parameters")
        
        # Load minima locations
        minima_file = trial_dir / 'minima_locations.pt'
        minima_locs = torch.load(minima_file, map_location=device)
        print(f"Loaded minima locations from {minima_file}")
        
        # Create objective function
        objective_fn = create_multi_minima_objective(minima_locs, HARD_PARAMS, device=device)
        
        # Create LineBO sampler
        linebo = LineBO(
            objective_fn,
            dimensions=dimensions,
            num_points_per_line=50,
            num_lines=100,
            device=device
        )
        
        # Create ZoMBIHop optimizer and load state
        zombihop = ZoMBIHop(
            objective=linebo.sampler,
            bounds=None,  # Will be loaded
            X_init_actual=None,
            X_init_expected=None,
            Y_init=None,
            run_uuid=resume_uuid,
            device=device,
            checkpoint_dir=str(trial_dir / 'checkpoints')
        )
        
        print(f"Resumed from activation={zombihop.current_activation}, "
              f"zoom={zombihop.current_zoom}, iteration={zombihop.current_iteration}\n")
    
    # Run optimization with time limit and unlimited activations
    print("="*80)
    print("STARTING 24-HOUR OPTIMIZATION")
    print("="*80 + "\n")
    
    trial_start = time.time()
    
    try:
        # *** SINGLE RUN - This executes exactly once, not in a loop ***
        needles_results, needles, needle_vals, X_all_actual, Y_all = zombihop.run(
            max_activations=float('inf'),  # Unlimited activations (within single run)
            time_limit_hours=time_limit_hours
        )
        # *** END OF SINGLE RUN ***
        
        trial_end = time.time()
        trial_duration = (trial_end - trial_start) / 3600.0
        
        print("\n" + "="*80)
        print("TRIAL COMPLETE")
        print("="*80)
        print(f"Trial duration: {trial_duration:.2f} hours")
        print(f"Total points sampled: {X_all_actual.shape[0]}")
        print(f"Needles found: {needles.shape[0]}")
        print(f"Run UUID: {zombihop.run_uuid}")
        print("="*80 + "\n")
        
        # Evaluate needles
        if needles.shape[0] > 0:
            distances = objective_fn.evaluate_needles(needles)
            
            if distances is not None:
                # Save final results
                results = {
                    'run_uuid': zombihop.run_uuid,
                    'trial_duration_hours': trial_duration,
                    'total_points': X_all_actual.shape[0],
                    'num_needles': needles.shape[0],
                    'distances_to_minima': distances,
                    'mean_distance': sum(distances) / len(distances),
                    'max_distance': max(distances),
                    'best_value': Y_all.max().item(),
                    'success': all(d < 0.10 for d in distances)  # Success if all within 0.10
                }
                
                results_file = trial_dir / f'results_{zombihop.run_uuid}.json'
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"Saved results to {results_file}")
                
                if results['success']:
                    print("\n🎉 SUCCESS! All minima found within tolerance! 🎉\n")
                else:
                    print(f"\n⚠️  {len([d for d in distances if d < 0.10])}/{num_minima} minima found within tolerance\n")
            else:
                print(f"\n❌ Insufficient needles found: {needles.shape[0]}/{num_minima}\n")
        else:
            print("\n❌ No needles found!\n")
    
    except KeyboardInterrupt:
        print("\n\nTrial interrupted by user. State has been saved.")
        print(f"Resume with: python test_24hour_variable_minima.py {zombihop.run_uuid}")
    
    except Exception as e:
        print(f"\n\nTrial failed with error: {e}")
        import traceback
        traceback.print_exc()
        print(f"Resume with: python test_24hour_variable_minima.py {zombihop.run_uuid}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("SINGLE TRIAL EXECUTION")
    print("="*80)
    print("This script will execute EXACTLY ONE optimization trial.")
    print("It does NOT repeat or loop multiple times.")
    print("="*80 + "\n")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        
        # Check for special commands
        if first_arg == 'list':
            print("Available trials:")
            print("="*80)
            trial_dirs = [d for d in Path('.').iterdir() if d.is_dir() and d.name.startswith('trial_')]
            if not trial_dirs:
                print("No trial directories found")
            else:
                for td in sorted(trial_dirs):
                    print(f"\nTrial directory: {td.name}")
                    checkpoints_dir = td / 'checkpoints'
                    if checkpoints_dir.exists():
                        for run_dir in sorted(checkpoints_dir.iterdir()):
                            if run_dir.is_dir() and run_dir.name.startswith('run_'):
                                uuid = run_dir.name.replace('run_', '')
                                # Load metadata if available
                                metadata_file = td / 'trial_metadata.json'
                                if metadata_file.exists():
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
                                    num_minima = metadata.get('num_minima', '?')
                                    dimensions = metadata.get('dimensions', '?')
                                    time_limit = metadata.get('time_limit_hours', '?')
                                    print(f"  UUID: {uuid} ({num_minima} minima, {dimensions}D, {time_limit}h)")
                                else:
                                    print(f"  UUID: {uuid}")
                    else:
                        print("  No checkpoints found")
            sys.exit(0)
        
        # Check if first argument is a UUID (for resuming) or trial parameters
        # If only one argument and it's a number, check if it's a UUID first
        if len(sys.argv) == 2 and first_arg.isdigit():
            # Single numeric argument - could be UUID or num_minima
            found_trial_dir = find_trial_directory_by_uuid(first_arg)
            if found_trial_dir is not None:
                # It's a valid UUID
                resume_uuid = first_arg
                print(f"Resuming trial with UUID: {resume_uuid}\n")
                run_variable_trial(resume_uuid=resume_uuid)
            else:
                # Not a UUID, treat as num_minima with defaults
                num_minima = int(first_arg)
                dimensions = DEFAULT_DIMENSIONS
                time_limit_hours = DEFAULT_TIME_LIMIT_HOURS
                print(f"Starting new trial with {num_minima} minima in {dimensions}D for {time_limit_hours} hours...\n")
                run_variable_trial(
                    num_minima=num_minima,
                    dimensions=dimensions,
                    time_limit_hours=time_limit_hours
                )
        else:
            # Multiple arguments or non-numeric first argument
            try:
                num_minima = int(first_arg)
                dimensions = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DIMENSIONS
                time_limit_hours = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_TIME_LIMIT_HOURS
                
                # If we can parse as numbers, treat as new trial parameters
                print(f"Starting new trial with {num_minima} minima in {dimensions}D for {time_limit_hours} hours...\n")
                run_variable_trial(
                    num_minima=num_minima,
                    dimensions=dimensions,
                    time_limit_hours=time_limit_hours
                )
            except ValueError:
                # If parsing as numbers fails, check if it's a UUID
                found_trial_dir = find_trial_directory_by_uuid(first_arg)
                
                if found_trial_dir is not None:
                    # It's a valid UUID
                    resume_uuid = first_arg
                    print(f"Resuming trial with UUID: {resume_uuid}\n")
                    run_variable_trial(resume_uuid=resume_uuid)
                else:
                    # Not a valid UUID and not valid trial parameters
                    print(f"Error: '{first_arg}' is not a valid UUID or trial parameter")
                    print("Usage:")
                    print("  python test_24hour_variable_minima.py [num_minima] [dimensions] [time_hours]")
                    print("  python test_24hour_variable_minima.py [resume_uuid]")
                    print("  python test_24hour_variable_minima.py list")
                    print("\nUUID formats supported:")
                    print("  - 4-digit numbers: 8364")
                    print("  - Short alphanumeric: a3f4")
                    print("  - Long UUIDs: a3f4b2c1-d5e6-7890-abcd-ef1234567890")
                    sys.exit(1)
    else:
        print(f"Starting new trial with defaults ({DEFAULT_NUM_MINIMA} minima, {DEFAULT_DIMENSIONS}D, {DEFAULT_TIME_LIMIT_HOURS}h)...\n")
        run_variable_trial()
    
    print("\n" + "="*80)
    print("SINGLE TRIAL COMPLETE")
    print("="*80)
    print("The script has completed its single trial execution.")
    print("To run another trial, execute this script again.")
    print("="*80 + "\n")

