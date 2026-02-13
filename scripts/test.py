"""
Quick test of ZoMBIHop + LineBO with L2-distance objective on the simplex.

Minimizes f(x) = ||x - target||_2 over x on the simplex (d-dimensional).
Uses 24 experiments per line. Artificially adds noise to inputs and outputs.
Convergence uses Probability of Improvement + stagnation window (same as main).
ZoMBIHop parameters match scripts/run_zombi_main.py (main.py).

Run: python -m scripts.test
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Tuple

from src import ZoMBIHop
from src.core.linebo import batch_line_simplex_segments, zero_sum_dirs

# Simplex dimension and experiments per line (same as run_zombi_main NUM_EXPERIMENTS)
NUM_EXPERIMENTS = 24
DIMENSIONS = 2  # Use 2 for fast smoke test; 10 for higher-dim check
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Artificial noise (same scale as typical run; ZoMBIHop uses input/output noise thresholds)
INPUT_NOISE_STD = 0.001   # std of Gaussian noise added to inputs (observed vs requested)
OUTPUT_NOISE_STD = 0.001  # std of Gaussian noise added to outputs (objective values)

# ZoMBIHop maximizes the objective, so we minimize distance by returning objective = -distance.
MINIMIZE_DISTANCE_OBJECTIVE = True  # if True, return y = -distance so maximizer minimizes distance

# Print candidate, line endpoints, and Y stats every time the objective is called (set True to debug).
DEBUG_OBJECTIVE = False


def l2_objective(
    ordered_endpoints: np.ndarray,
    target: np.ndarray,
    num_experiments: int = NUM_EXPERIMENTS,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate -L2 distance to target at points along the given lines.
    Adds artificial input noise (to x) and output noise (to y).

    ordered_endpoints: (2, 2, d) — 2 lines, each (left, right).
    target: (d,) — fixed point on simplex.
    Returns (x_meas, y) with x_meas (n, d) noisy inputs, y (n,) noisy -distance.
    """
    rng = rng or np.random.default_rng()
    x_list = []
    for line_idx in range(ordered_endpoints.shape[0]):
        left = ordered_endpoints[line_idx, 0]
        right = ordered_endpoints[line_idx, 1]
        for t in np.linspace(0, 1, num_experiments):
            x = left + t * (right - left)
            x_list.append(x)
    x_meas = np.array(x_list, dtype=np.float64)  # (2 * num_experiments, d)
    # Input noise: add Gaussian to each coordinate, then re-normalize to simplex
    x_meas = x_meas + rng.normal(0, INPUT_NOISE_STD, size=x_meas.shape).astype(np.float64)
    x_meas = np.clip(x_meas, 0.0, 1.0)
    x_meas = x_meas / x_meas.sum(axis=1, keepdims=True)
    dist = np.linalg.norm(x_meas - target, axis=1)  # (n,) — what we minimize
    # Negate so ZoMBIHop (maximizer) minimizes distance: y = -distance
    y = (-dist.astype(np.float64)) if MINIMIZE_DISTANCE_OBJECTIVE else dist.astype(np.float64)
    # Output noise: add Gaussian to y
    y = y + rng.normal(0, OUTPUT_NOISE_STD, size=y.shape).astype(np.float64)
    return x_meas, y


def make_objective_wrapper(target: np.ndarray, num_lines: int = 100, device=None, rng: np.random.Generator | None = None):
    """Build ZoMBIHop objective: LineBO + L2 objective with input/output noise. num_lines matches run_zombi_main."""
    target = np.asarray(target, dtype=np.float64)
    device = device or DEVICE
    rng = rng or np.random.default_rng()

    def objective_fn(ordered_endpoints: np.ndarray):
        return l2_objective(ordered_endpoints, target=target, rng=rng)

    _call_count = [0]  # mutable so wrapper can increment

    def wrapper(
        x_tell: torch.Tensor,
        bounds: torch.Tensor | None = None,
        acquisition_function=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _call_count[0] += 1
        candidate_np = x_tell.detach().cpu().numpy() if x_tell.dim() > 0 else x_tell.unsqueeze(0).cpu().numpy()
        if candidate_np.ndim > 1:
            candidate_np = candidate_np.reshape(-1)
        directions = zero_sum_dirs(num_lines, DIMENSIONS, device=device, dtype=torch.float64)
        x_left, x_right, t_min, t_max, mask = batch_line_simplex_segments(x_tell, directions)
        if x_left is None or x_left.shape[0] < 2:
            fallback = zero_sum_dirs(4, DIMENSIONS, device=device, dtype=torch.float64)
            x_left, x_right, t_min, t_max, mask = batch_line_simplex_segments(x_tell, fallback)
        if x_left is None or x_left.shape[0] < 2:
            x_left = x_tell.unsqueeze(0).repeat(2, 1)
            x_right = x_tell.unsqueeze(0).repeat(2, 1)
        x_left, x_right = x_left[:2], x_right[:2]
        ordered_endpoints = np.stack([x_left.cpu().numpy(), x_right.cpu().numpy()], axis=1)  # (2, 2, d)
        if DEBUG_OBJECTIVE:
            left0, right0 = ordered_endpoints[0, 0], ordered_endpoints[0, 1]
            left1, right1 = ordered_endpoints[1, 0], ordered_endpoints[1, 1]
            print(f"  [test objective #{_call_count[0]}] candidate (x_tell) = {candidate_np}")
            print(f"  [test objective #{_call_count[0]}] line0: left = {left0}, right = {right0}")
            print(f"  [test objective #{_call_count[0]}] line1: left = {left1}, right = {right1}")
        x_meas, y = objective_fn(ordered_endpoints)  # x_meas and y already have noise from l2_objective
        if DEBUG_OBJECTIVE:
            print(f"  [test objective #{_call_count[0]}] returned n={len(y)}, Y: min={float(np.min(y)):.4f}, max={float(np.max(y)):.4f}, mean={float(np.mean(y)):.4f}")
        X_actual = torch.tensor(x_meas, device=device, dtype=torch.float64)
        X_expected = X_actual.clone()
        Y = torch.tensor(y, device=device, dtype=torch.float64).reshape(-1)
        return X_actual, X_expected, Y

    return wrapper


def main():
    device = torch.device(DEVICE)
    dtype = torch.float64
    bounds = torch.zeros((2, DIMENSIONS), device=device, dtype=dtype)
    bounds[0] = 0.0
    bounds[1] = 1.0

    # Target on simplex (we will minimize L2 distance to this)
    d = DIMENSIONS
    target = np.ones(d, dtype=np.float64) / d  # uniform: [1/d, ..., 1/d], sum = 1
    assert np.isclose(target.sum(), 1.0), "Target must sum to 1"

    # Initial points on simplex (2 × 24 = 48 initial evaluations); add input/output noise
    rng = np.random.default_rng(42)
    X_init = ZoMBIHop.random_simplex(2 * NUM_EXPERIMENTS, bounds[0].cpu(), bounds[1].cpu(), device=str(device))
    X_init = X_init.cpu().numpy()
    X_init = X_init + rng.normal(0, INPUT_NOISE_STD, size=X_init.shape).astype(np.float64)
    X_init = np.clip(X_init, 0.0, 1.0)
    X_init = X_init / X_init.sum(axis=1, keepdims=True)
    dist_init = np.linalg.norm(X_init - target, axis=1)
    noise_init = rng.normal(0, OUTPUT_NOISE_STD, size=dist_init.shape).astype(np.float64)
    y_init = (-dist_init + noise_init) if MINIMIZE_DISTANCE_OBJECTIVE else (dist_init + noise_init)
    Y_init = torch.tensor(y_init, device=device, dtype=dtype).reshape(-1, 1)
    X_init_actual = torch.tensor(X_init[:, :DIMENSIONS], device=device, dtype=dtype)
    X_init_expected = X_init_actual.clone()

    objective_wrapper = make_objective_wrapper(target, num_lines=100, device=device, rng=rng)
    checkpoint_dir = Path("actual_runs") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Same ZoMBIHop parameters as run_zombi_main.py (main.py)
    dimensions = DIMENSIONS
    optimizer = ZoMBIHop(
        objective=objective_wrapper,
        bounds=bounds,
        X_init_actual=X_init_actual,
        X_init_expected=X_init_expected,
        Y_init=Y_init,
        penalization_threshold=0.0005915,
        convergence_pi_threshold=0.01,
        input_noise_threshold_mult=2.0,
        output_noise_threshold_mult=2.0,
        max_zooms=5,
        max_iterations=7,
        top_m_points=max(dimensions + 1, 4),
        n_restarts=50,
        raw=500,
        penalty_num_directions=10 * dimensions,
        penalty_max_radius=0.33633,
        penalty_radius_step=None,
        max_gp_points=3000,
        device=device,
        dtype=dtype,
        checkpoint_dir=str(checkpoint_dir),
        verbose=True,
    )

    print(f"Target (minimize L2 distance to): {target}")
    print(f"Objective = -distance so ZoMBIHop maximizer minimizes distance.")
    print(f"Note: Same candidate repeated means the acquisition is maximized at that point (e.g. vertex [0,1]); LineBO still evaluates different lines through it (see [test objective] prints).")
    print(f"Running ZoMBIHop for 2 activations ({d}D simplex, L2 objective, input/output noise, same params as main)...")
    needles, needle_locs, needle_vals, X_all_actual, Y_all = optimizer.run(max_activations=2, time_limit_hours=None)

    best_idx = Y_all.squeeze().argmax().item()
    best_x = X_all_actual[best_idx].cpu().numpy()
    best_y = Y_all[best_idx].item()
    dist_best = np.linalg.norm(best_x - target)
    print(f"\nBest point found: {best_x} (sum={best_x.sum():.4f})")
    print(f"Min distance achieved: {dist_best:.4f}  (objective Y = -distance: {best_y:.4f})")
    print(f"Target: {target}")
    # In higher dimensions the simplex is larger; allow slightly larger distance for 10D
    max_ok_dist = 0.5 if d <= 2 else 0.7
    assert dist_best < max_ok_dist, f"Expected distance < {max_ok_dist}, got {dist_best}"
    print("test.py passed.")


if __name__ == "__main__":
    main()
