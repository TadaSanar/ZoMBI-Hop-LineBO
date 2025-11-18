"""
Example script demonstrating ZoMBIHop checkpointing functionality.
"""

import torch
from zombihop_linebo_final import ZoMBIHop
from pathlib import Path
import json

def simple_objective(x, bounds, acq_fn):
    """
    Simple test objective function.
    Returns x_expected (what was requested), x_actual (what was sampled), 
    and y (objective value).
    """
    # Ensure x is on device
    device = x.device
    dtype = x.dtype
    
    # Simple quadratic objective (example)
    y = -(x - 0.5).pow(2).sum()
    
    # Simulate small noise between expected and actual
    noise = 0.01 * torch.randn_like(x)
    x_actual = x + noise
    
    # Project back to simplex
    x_actual = torch.clamp(x_actual, 0, 1)
    x_actual = x_actual / x_actual.sum()
    
    # Return in expected format
    x_expected = x.unsqueeze(0)  # (1, d)
    x_actual = x_actual.unsqueeze(0)  # (1, d)
    y = y.unsqueeze(0)  # (1,)
    
    return x_expected, x_actual, y


def example_new_run():
    """Example: Start a new optimization run."""
    print("=" * 60)
    print("EXAMPLE 1: Starting a new run")
    print("=" * 60)
    
    # Set up initial data
    d = 5  # dimensions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create bounds (simplex: all coords sum to 1)
    bounds = torch.stack([
        torch.zeros(d),
        torch.ones(d)
    ]).to(device=device, dtype=torch.float64)
    
    # Create initial points on simplex
    n_init = 10
    X_init = torch.randn(n_init, d).abs()
    X_init = X_init / X_init.sum(dim=1, keepdim=True)
    X_init = X_init.to(device=device, dtype=torch.float64)
    
    X_init_actual = X_init.clone()
    X_init_expected = X_init.clone()
    
    # Evaluate initial points
    Y_init = torch.empty(n_init, 1, device=device, dtype=torch.float64)
    for i in range(n_init):
        _, _, y = simple_objective(X_init[i], bounds, None)
        Y_init[i] = y
    
    # Initialize optimizer
    optimizer = ZoMBIHop(
        objective=simple_objective,
        bounds=bounds,
        X_init_actual=X_init_actual,
        X_init_expected=X_init_expected,
        Y_init=Y_init,
        max_zooms=2,
        max_iterations=5,
        device=device,
        checkpoint_dir='example_checkpoints'
    )
    
    print(f"Started new run with UUID: {optimizer.run_uuid}")
    print(f"Checkpoint directory: {optimizer.run_dir}")
    print()
    
    # Run optimization (will save checkpoints at each iteration)
    # Note: You might want to interrupt this to test resuming
    # results = optimizer.run()
    
    return optimizer.run_uuid


def example_resume_run(run_uuid: str):
    """Example: Resume from a saved run."""
    print("=" * 60)
    print(f"EXAMPLE 2: Resuming run {run_uuid}")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Resume from saved UUID
    # Note: All data will be loaded from checkpoint, so we only need
    # to provide the objective function
    optimizer = ZoMBIHop(
        objective=simple_objective,
        bounds=None,  # Will be loaded
        X_init_actual=None,  # Will be loaded
        X_init_expected=None,  # Will be loaded
        Y_init=None,  # Will be loaded
        run_uuid=run_uuid,
        device=device,
        checkpoint_dir='example_checkpoints'
    )
    
    print(f"Resuming from activation={optimizer.current_activation}, "
          f"zoom={optimizer.current_zoom}, iteration={optimizer.current_iteration}")
    print(f"Current number of points: {optimizer.X_all_actual.shape[0]}")
    print(f"Current number of needles: {optimizer.needles.shape[0]}")
    print()
    
    # Continue optimization
    results = optimizer.run()
    
    return results


def example_inspect_checkpoint(run_uuid: str):
    """Example: Inspect a saved checkpoint without running."""
    print("=" * 60)
    print(f"EXAMPLE 3: Inspecting checkpoint {run_uuid}")
    print("=" * 60)
    
    checkpoint_dir = Path('example_checkpoints')
    run_dir = checkpoint_dir / f'run_{run_uuid}'
    
    if not run_dir.exists():
        print(f"Checkpoint directory {run_dir} does not exist!")
        return
    
    # Load config
    with open(run_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    print("Configuration:")
    print(f"  Dimensions: {config['d']}")
    print(f"  Max zooms: {config['max_zooms']}")
    print(f"  Max iterations: {config['max_iterations']}")
    print()
    
    # Find latest state
    with open(run_dir / 'current_state.txt', 'r') as f:
        latest_state = f.read().strip()
    
    print(f"Latest state: {latest_state}")
    
    state_dir = run_dir / 'states' / latest_state
    
    # Load statistics
    with open(state_dir / 'stats.json', 'r') as f:
        stats = json.load(f)
    
    print("\nStatistics:")
    print(f"  Total points: {stats['num_points_total']}")
    print(f"  Needles found: {stats['num_needles']}")
    print(f"  Best value: {stats['best_value']}")
    print(f"  Input noise: {stats['input_noise']:.6f}")
    print(f"  Mean distance: {stats['mean_distance']:.6f}")
    print(f"  Median distance: {stats['median_distance']:.6f}")
    print()
    
    # Load and inspect tensors
    tensors = torch.load(state_dir / 'tensors.pt', map_location='cpu')
    
    print("Tensor shapes:")
    for key, tensor in tensors.items():
        print(f"  {key}: {tensor.shape}")
    print()
    
    # List all available states
    states_dir = run_dir / 'states'
    all_states = sorted([d.name for d in states_dir.iterdir() if d.is_dir()])
    print(f"Available states ({len(all_states)}):")
    for state in all_states[:10]:  # Show first 10
        print(f"  - {state}")
    if len(all_states) > 10:
        print(f"  ... and {len(all_states) - 10} more")


if __name__ == "__main__":
    # Example 1: Start a new run
    # Uncomment to test:
    # run_uuid = example_new_run()
    
    # Example 2: Resume from a saved run
    # Replace 'abcd' with your actual run UUID
    # example_resume_run('abcd')
    
    # Example 3: Inspect a checkpoint
    # Replace 'abcd' with your actual run UUID
    # example_inspect_checkpoint('abcd')
    
    print("\nExamples ready to use!")
    print("Uncomment the examples in __main__ to test them.")
    print("\nUsage:")
    print("1. Run example_new_run() to start a new optimization")
    print("2. Note the UUID that gets printed")
    print("3. Run example_resume_run(uuid) to continue from that run")
    print("4. Run example_inspect_checkpoint(uuid) to inspect saved data")

