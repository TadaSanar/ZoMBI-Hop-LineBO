"""
Database-driven ZoMBI-Hop runner (LineBO + serial/DB handshake).

This module contains the DB-backed objective + runner that used to live in
`zombihop_linebo_v2.py`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src import ZoMBIHop
from src.core.linebo import batch_line_simplex_segments, zero_sum_dirs

from scripts import communication

# Default Configuration
DEFAULT_DIMENSIONS = 10
DEFAULT_NUM_MINIMA = 3
DEFAULT_TIME_LIMIT_HOURS = 24.0
NUM_EXPERIMENTS = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Database interfacing settings
OPTIMIZING_DIMS = [5, 6, 7]


def normalize_last_axis(arr: np.ndarray) -> np.ndarray:
    """Normalize array along last axis to sum to 1, handling edge cases."""
    a = np.asarray(arr, dtype=float)
    sums = a.sum(axis=-1, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    result = a / sums
    mask = ~np.isfinite(result).all(axis=-1, keepdims=True)
    if np.any(mask):
        d = a.shape[-1]
        result = np.where(mask, 1.0 / d, result)
    return result


def get_y_measurements(
    x,
    db: str = "./sql/objective.db",
    verbose: bool = False,
    ready_for_objectives: bool = False,
):
    import os as _os
    import sqlite3
    import time as _time

    consecutive_errors = 0
    max_consecutive_errors = 10

    if ready_for_objectives:
        while True:
            try:
                conn = sqlite3.connect(db, timeout=10.0)
                cur = conn.cursor()
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS handshake (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    new_objective_available INTEGER DEFAULT 0
                )"""
                )
                cur.execute("INSERT OR IGNORE INTO handshake (id, new_objective_available) VALUES (1, 0)")
                cur.execute("SELECT new_objective_available FROM handshake WHERE id = 1")
                flag = cur.fetchone()
                conn.close()
                if flag and flag[0] == 1:
                    break
                _time.sleep(1)
            except Exception:
                _time.sleep(1)
                continue

    while True:
        try:
            if not _os.path.exists(db):
                _time.sleep(1)
                continue
            from scripts.communication import _objective_db_lock, _objective_writing

            if _objective_writing:
                _time.sleep(0.1)
                continue
            with _objective_db_lock:
                conn = sqlite3.connect(db, timeout=30.0)
                cur = conn.cursor()
                cur.execute("SELECT * FROM objective")
                all_rows = cur.fetchall()
                if not all_rows:
                    conn.close()
                    _time.sleep(1)
                    continue
                if len(all_rows) == 1 and len(all_rows[0]) > 1:
                    flat = list(all_rows[0])
                elif len(all_rows) > 1 and len(all_rows[0]) == 1:
                    flat = [r[0] for r in all_rows]
                else:
                    conn.close()
                    _time.sleep(1)
                    continue
                y_all = np.array(flat, dtype=float)
                valid_mask = ~np.isnan(y_all)
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) == 0:
                    conn.close()
                    _time.sleep(1)
                    continue
                y = y_all[valid_indices].reshape(-1)
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='compositions'",
                )
                if cur.fetchone():
                    cur.execute("SELECT * FROM compositions")
                    comp_rows = cur.fetchall()
                    X_meas_full = np.array(comp_rows, dtype=float) if comp_rows else None
                else:
                    X_meas_full = None
                if X_meas_full is not None and X_meas_full.shape[0] >= len(flat):
                    x_meas = X_meas_full[valid_indices][:, OPTIMIZING_DIMS]
                elif X_meas_full is not None and X_meas_full.shape[0] > 0:
                    x_meas = np.zeros((len(valid_indices), len(OPTIMIZING_DIMS)), dtype=float)
                    n_to_copy = min(X_meas_full.shape[0], len(valid_indices))
                    x_meas[:n_to_copy] = X_meas_full[:n_to_copy][:, OPTIMIZING_DIMS]
                else:
                    x_meas = np.zeros((len(valid_indices), len(OPTIMIZING_DIMS)), dtype=float)
                if verbose:
                    print(f"[get_y_measurements] ✅ NEW DATA RECEIVED: {len(y)} objective values")
                conn.close()
                break
        except Exception:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                _time.sleep(5.0)
                consecutive_errors = 0
            else:
                _time.sleep(1)
            continue

    if ready_for_objectives:
        try:
            conn = sqlite3.connect(db, timeout=10.0)
            cur = conn.cursor()
            cur.execute("UPDATE handshake SET new_objective_available = 0 WHERE id = 1")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[get_y_measurements] Error clearing handshake flag: {e}")
    return y, x_meas


def _pad_to_10d(arr):
    arr = np.atleast_2d(arr)
    out = np.zeros((arr.shape[0], 10), dtype=arr.dtype)
    out[:, OPTIMIZING_DIMS] = arr
    return out


def objective_function_init(ordered_endpoints, num_experiments: int = NUM_EXPERIMENTS):
    best_start = ordered_endpoints[0][0]
    best_end = ordered_endpoints[0][1]
    cache_start = ordered_endpoints[1][0]
    cache_end = ordered_endpoints[1][1]

    x = np.array([best_start + t * (best_end - best_start) for t in np.linspace(0, 1, num_experiments)])
    x_cache = np.array([cache_start + t * (cache_end - cache_start) for t in np.linspace(0, 1, num_experiments)])

    best_start_norm = _pad_to_10d(normalize_last_axis(np.round(best_start, 3)))[0]
    best_end_norm = _pad_to_10d(normalize_last_axis(np.round(best_end, 3)))[0]
    cache_start_norm = _pad_to_10d(normalize_last_axis(np.round(cache_start, 3)))[0]
    cache_end_norm = _pad_to_10d(normalize_last_axis(np.round(cache_end, 3)))[0]

    x_norm = _pad_to_10d(normalize_last_axis(np.round(x, 3)))
    x_cache_norm = _pad_to_10d(normalize_last_axis(np.round(x_cache, 3)))

    communication.write_compositions(
        start=best_start_norm,
        end=best_end_norm,
        array=x_norm,
        start_cache=cache_start_norm,
        end_cache=cache_end_norm,
        array_cache=x_cache_norm,
        timestamp=time.time(),
    )

    y, x_meas = get_y_measurements(x, verbose=True, ready_for_objectives=False)
    # Reverse sign so that ZoMBIHop finds minima instead of maxima
    return x_meas, -y.ravel()


def objective_function(ordered_endpoints, num_experiments: int = NUM_EXPERIMENTS):
    best_start = ordered_endpoints[0][0]
    best_end = ordered_endpoints[0][1]
    cache_start = ordered_endpoints[1][0]
    cache_end = ordered_endpoints[1][1]

    x = np.array([best_start + t * (best_end - best_start) for t in np.linspace(0, 1, num_experiments)])
    x_cache = np.array([cache_start + t * (cache_end - cache_start) for t in np.linspace(0, 1, num_experiments)])

    best_start_norm = _pad_to_10d(normalize_last_axis(np.round(best_start, 3)))[0]
    best_end_norm = _pad_to_10d(normalize_last_axis(np.round(best_end, 3)))[0]
    cache_start_norm = _pad_to_10d(normalize_last_axis(np.round(cache_start, 3)))[0]
    cache_end_norm = _pad_to_10d(normalize_last_axis(np.round(cache_end, 3)))[0]

    x_norm = _pad_to_10d(normalize_last_axis(np.round(x, 3)))
    x_cache_norm = _pad_to_10d(normalize_last_axis(np.round(x_cache, 3)))

    communication.write_compositions(
        start=best_start_norm,
        end=best_end_norm,
        array=x_norm,
        start_cache=cache_start_norm,
        end_cache=cache_end_norm,
        array_cache=x_cache_norm,
        timestamp=time.time(),
    )

    y, x_meas = get_y_measurements(x, verbose=True, ready_for_objectives=True)
    return x_meas, -y.ravel()


def create_db_objective_wrapper(objective_fn, dimensions: int, num_lines: int = 100, device="cuda"):
    """Bridge ZoMBIHop objective interface to DB-backed line objective."""

    def wrapper(
        x_tell: torch.Tensor,
        bounds: torch.Tensor | None = None,
        acquisition_function=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        directions = zero_sum_dirs(num_lines, dimensions, device=device, dtype=torch.float64)
        x_left, x_right, t_min, t_max, mask = batch_line_simplex_segments(x_tell, directions)

        if x_left is None or x_left.shape[0] < 2:
            print("⚠️ Warning: Not enough valid lines, using random fallback")
            fallback_dirs = zero_sum_dirs(2, dimensions, device=device, dtype=torch.float64)
            x_left, x_right, t_min, t_max, mask = batch_line_simplex_segments(x_tell, fallback_dirs)
            if x_left is None or x_left.shape[0] == 0:
                print("⚠️ Using ultimate fallback: single point")
                x_left = x_tell.unsqueeze(0).repeat(2, 1)
                x_right = x_tell.unsqueeze(0).repeat(2, 1)

        x_left = x_left[:2]
        x_right = x_right[:2]
        if x_left.shape[0] < 2:
            x_left = torch.cat([x_left, x_left], dim=0)
            x_right = torch.cat([x_right, x_right], dim=0)

        ordered_endpoints = np.stack([x_left.cpu().numpy(), x_right.cpu().numpy()], axis=1)

        x_meas, y = objective_fn(ordered_endpoints)

        X_actual = torch.tensor(x_meas, device=device, dtype=torch.float64)
        X_expected = X_actual.clone()
        Y = torch.tensor(y, device=device, dtype=torch.float64).reshape(-1)
        return X_actual, X_expected, Y

    return wrapper


def run_zombi_main(resume_uuid: str | None = None):
    """Run DB-driven ZoMBI-Hop loop (new or resume)."""
    if resume_uuid is None:
        communication.reset_objective()

    dimensions = len(OPTIMIZING_DIMS)
    device = torch.device(DEVICE)
    bounds = torch.zeros((2, dimensions), device=device, dtype=torch.float64)
    bounds[0] = 0.0
    bounds[1] = 1.0

    base_dir = Path("actual_runs")
    base_dir.mkdir(exist_ok=True)
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    objective_wrapper = create_db_objective_wrapper(
        objective_function,
        dimensions=dimensions,
        num_lines=100,
        device=device,
    )

    if resume_uuid is None:
        print("=" * 80)
        print("STARTING NEW ZOMBIHOP TRIAL (DATABASE-DRIVEN)")
        print("=" * 80)
        print(f"Dimensions: {dimensions} (from OPTIMIZING_DIMS: {OPTIMIZING_DIMS})")
        print(f"Device: {device}")
        print("=" * 80 + "\n")

        print("Generating initial data via database...")
        start_points = ZoMBIHop.random_simplex(4, bounds[0], bounds[1], device=device)
        endpoints_list = []
        for i in range(0, len(start_points), 2):
            endpoints_list.append([start_points[i].cpu().numpy(), start_points[i + 1].cpu().numpy()])
        ordered_endpoints = np.array(endpoints_list)

        x_meas, y = objective_function_init(ordered_endpoints, num_experiments=NUM_EXPERIMENTS)
        X_init_actual = torch.tensor(x_meas, device=device, dtype=torch.float64)
        X_init_expected = X_init_actual.clone()
        Y_init = torch.tensor(y, device=device, dtype=torch.float64).reshape(-1, 1)

        optimizer = ZoMBIHop(
            objective=objective_wrapper,
            bounds=bounds,
            X_init_actual=X_init_actual,
            X_init_expected=X_init_expected,
            Y_init=Y_init,
            penalization_threshold=1e-3,
            improvement_threshold_noise_mult=1.5,
            input_noise_threshold_mult=3.4,
            n_consecutive_no_improvements=5,
            top_m_points=3,
            max_zooms=3,
            max_iterations=10,
            n_restarts=50,
            raw=5000,
            penalty_num_directions=100,
            penalty_max_radius=0.3,
            penalty_radius_step=0.01,
            max_gp_points=3000,
            device=device,
            checkpoint_dir=str(checkpoint_dir),
        )

        print(f"✅ Starting new trial with UUID: {optimizer.run_uuid}")
    else:
        run_dir = checkpoint_dir / f"run_{resume_uuid}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found for UUID {resume_uuid} (expected {run_dir})")

        optimizer = ZoMBIHop(
            objective=objective_wrapper,
            bounds=None,
            X_init_actual=None,
            X_init_expected=None,
            Y_init=None,
            run_uuid=resume_uuid,
            device=device,
            checkpoint_dir=str(checkpoint_dir),
        )
        print(
            f"✅ Resumed from activation={optimizer.current_activation}, "
            f"zoom={optimizer.current_zoom}, iteration={optimizer.current_iteration}\n"
        )

    print("=" * 80)
    print("STARTING OPTIMIZATION")
    print("=" * 80 + "\n")

    optimizer.run(max_activations=float("inf"), time_limit_hours=None)
