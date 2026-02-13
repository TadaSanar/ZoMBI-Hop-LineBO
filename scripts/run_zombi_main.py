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
from src.core.linebo import batch_line_simplex_segments, line_simplex_segment, zero_sum_dirs

from scripts import communication

# # Default Configuration
# DEFAULT_DIMENSIONS = 10
# DEFAULT_NUM_MINIMA = 3
# DEFAULT_TIME_LIMIT_HOURS = 24.0
NUM_EXPERIMENTS = 24
NUM_INIT_DATA = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Database interfacing settings
OPTIMIZING_DIMS = [0, 8, 9]


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
    """
    Evaluate each initial line separately: one write_compositions + get_y_measurements per line,
    then concatenate (x_meas, y) so the apparatus receives one line at a time.
    Real line = current line being measured; cache line = the other initial line (different from real).
    """
    num_lines = len(ordered_endpoints)
    x_meas_list = []
    y_list = []
    for line_idx in range(num_lines):
        # Real line: current line
        left = ordered_endpoints[line_idx][0]
        right = ordered_endpoints[line_idx][1]
        x = np.array([left + t * (right - left) for t in np.linspace(0, 1, num_experiments)])
        left_norm = _pad_to_10d(normalize_last_axis(np.round(left, 3)))[0]
        right_norm = _pad_to_10d(normalize_last_axis(np.round(right, 3)))[0]
        x_norm = _pad_to_10d(normalize_last_axis(np.round(x, 3)))
        # Cache line: other initial line (so real and cache differ)
        cache_idx = (line_idx + 1) % num_lines
        left_cache = ordered_endpoints[cache_idx][0]
        right_cache = ordered_endpoints[cache_idx][1]
        x_cache = np.array([left_cache + t * (right_cache - left_cache) for t in np.linspace(0, 1, num_experiments)])
        left_cache_norm = _pad_to_10d(normalize_last_axis(np.round(left_cache, 3)))[0]
        right_cache_norm = _pad_to_10d(normalize_last_axis(np.round(right_cache, 3)))[0]
        x_cache_norm = _pad_to_10d(normalize_last_axis(np.round(x_cache, 3)))
        communication.write_compositions(
            start=left_norm,
            end=right_norm,
            array=x_norm,
            start_cache=left_cache_norm,
            end_cache=right_cache_norm,
            array_cache=x_cache_norm,
            timestamp=time.time(),
        )
        y, x_meas = get_y_measurements(x, verbose=True, ready_for_objectives=False)
        x_meas_list.append(x_meas)
        y_list.append(-np.asarray(y).ravel())
    x_meas = np.vstack(x_meas_list)
    y = np.concatenate(y_list)
    return x_meas, y


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


def objective_function_dry(ordered_endpoints, num_experiments: int = NUM_EXPERIMENTS):
    """
    Same as objective_function but NO communication: compute compositions that WOULD
    be sent, print everything, return synthetic (x_meas, y) so the optimizer can continue.
    """
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

    # ---- PRINT EVERYTHING (no communication) ----
    print("\n" + "=" * 80)
    print("[DRY RUN] WOULD SEND TO APPARATUS (not sending)")
    print("=" * 80)
    print("[DRY RUN] Line 0 (best): left  =", best_start, "  right =", best_end)
    print("[DRY RUN] Line 1 (cache): left =", cache_start, "  right =", cache_end)
    print("[DRY RUN] Line 0 (best)  normalized 10d:", best_start_norm, "->", best_end_norm)
    print("[DRY RUN] Line 1 (cache) normalized 10d:", cache_start_norm, "->", cache_end_norm)
    print("[DRY RUN] Compositions along line 0 (first 5):")
    for i in range(min(5, len(x_norm))):
        print(f"  [{i}] {x_norm[i]}")
    if len(x_norm) > 5:
        print(f"  ... ({len(x_norm)} total)")
    print("[DRY RUN] Compositions along line 1 (first 5):")
    for i in range(min(5, len(x_cache_norm))):
        print(f"  [{i}] {x_cache_norm[i]}")
    if len(x_cache_norm) > 5:
        print(f"  ... ({len(x_cache_norm)} total)")
    print("[DRY RUN] OPTIMIZING_DIMS =", OPTIMIZING_DIMS)
    x_meas_0 = x_norm[:, OPTIMIZING_DIMS]
    x_meas_1 = x_cache_norm[:, OPTIMIZING_DIMS]
    x_meas = np.vstack([x_meas_0, x_meas_1]).astype(np.float64)
    print("[DRY RUN] Expected x_meas shape:", x_meas.shape, "(points × OPTIMIZING_DIMS)")
    # Synthetic Y (negative for minimization; random so optimizer keeps going)
    y_fake = -np.random.uniform(0.5, 2.0, size=x_meas.shape[0]).astype(np.float64)
    print("[DRY RUN] Synthetic Y (fake objectives, negated for min):", y_fake[:8], "...")
    print("=" * 80 + "\n")

    return x_meas, y_fake


def create_db_objective_wrapper(objective_fn, dimensions: int, num_lines: int = 100, device="cuda", dry_run: bool = False):
    """Bridge ZoMBIHop objective interface to DB-backed line objective."""

    def wrapper(
        x_tell: torch.Tensor,
        bounds: torch.Tensor | None = None,
        acquisition_function=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_tries = 20  # resample directions until we get ≥2 non-degenerate segments (e.g. when at vertex)
        for _ in range(max_tries):
            directions = zero_sum_dirs(num_lines, dimensions, device=device, dtype=torch.float64)
            x_left, x_right, t_min, t_max, mask = batch_line_simplex_segments(x_tell, directions)
            if x_left is not None and x_left.shape[0] >= 2:
                break
        else:
            # Still < 2 valid lines (e.g. x_tell at vertex); try a smaller batch
            fallback_dirs = zero_sum_dirs(max(2, num_lines), dimensions, device=device, dtype=torch.float64)
            x_left, x_right, t_min, t_max, mask = batch_line_simplex_segments(x_tell, fallback_dirs)
            if x_left is None or x_left.shape[0] < 2:
                print("⚠️ Warning: Not enough non-degenerate lines (e.g. at vertex), using single-point fallback")
                x_left = x_tell.unsqueeze(0).repeat(2, 1)
                x_right = x_tell.unsqueeze(0).repeat(2, 1)

        x_left = x_left[:2]
        x_right = x_right[:2]
        if x_left.shape[0] < 2:
            x_left = torch.cat([x_left, x_left], dim=0)
            x_right = torch.cat([x_right, x_right], dim=0)

        ordered_endpoints = np.stack([x_left.cpu().numpy(), x_right.cpu().numpy()], axis=1)

        if dry_run:
            print("\n" + "=" * 80)
            print("[DRY RUN] LineBO step (up to communication)")
            print("=" * 80)
            print("[DRY RUN] x_tell (current best point, d-dim):", x_tell.cpu().numpy())
            print("[DRY RUN] num_lines sampled:", num_lines, "| dimensions:", dimensions)
            print("[DRY RUN] First 3 zero-sum directions:\n", directions[:3].cpu().numpy())
            print("[DRY RUN] t_min, t_max (line segment params):", t_min[:2].cpu().numpy(), t_max[:2].cpu().numpy())
            print("[DRY RUN] Line 0 left  (boundary):", x_left[0].cpu().numpy())
            print("[DRY RUN] Line 0 right (boundary):", x_right[0].cpu().numpy())
            print("[DRY RUN] Line 1 left  (boundary):", x_left[1].cpu().numpy())
            print("[DRY RUN] Line 1 right (boundary):", x_right[1].cpu().numpy())
            print("[DRY RUN] ordered_endpoints shape:", ordered_endpoints.shape)
            print("=" * 80)

        x_meas, y = objective_fn(ordered_endpoints)

        X_actual = torch.tensor(x_meas, device=device, dtype=torch.float64)
        X_expected = X_actual.clone()
        Y = torch.tensor(y, device=device, dtype=torch.float64).reshape(-1)
        return X_actual, X_expected, Y

    return wrapper


def initial_lines_on_boundary(
    num_lines: int,
    bounds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    max_retries: int = 10,
) -> np.ndarray:
    """
    Generate num_lines with endpoints on the simplex boundary.

    Sample num_lines interior points on the simplex and num_lines random
    zero-sum directions; for each point + direction, extrapolate to the
    simplex boundary (as in linebo.line_simplex_segment) to get (x_left, x_right).
    Returns ordered_endpoints of shape (num_lines, 2, d) where [i, 0] is left
    and [i, 1] is right endpoint on the boundary.
    """
    d = bounds.shape[1]
    low, high = bounds[0], bounds[1]
    # Interior points on simplex (sum=1, in [0,1])
    points = ZoMBIHop.random_simplex(num_lines, low, high, device=device)
    endpoints_list = []
    for i in range(num_lines):
        x0 = points[i]
        for _ in range(max_retries):
            direction = zero_sum_dirs(1, d, device=device, dtype=dtype).squeeze(0)
            seg = line_simplex_segment(x0, direction)
            if seg is not None:
                _t_min, _t_max, x_left, x_right = seg
                endpoints_list.append([x_left.cpu().numpy(), x_right.cpu().numpy()])
                break
        else:
            # Fallback: segment between two vertices (boundary edges)
            ei = torch.zeros(d, device=device, dtype=dtype)
            ej = torch.zeros(d, device=device, dtype=dtype)
            ei[i % d] = 1.0
            ej[(i + 1) % d] = 1.0
            endpoints_list.append([ei.cpu().numpy(), ej.cpu().numpy()])
    return np.array(endpoints_list)


def _load_bounds_from_run(run_dir: Path, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Load bounds tensor from a saved run directory (for resuming)."""
    current_state_file = run_dir / "current_state.txt"
    if not current_state_file.exists():
        raise FileNotFoundError(f"No current_state.txt in {run_dir}; cannot load bounds for resume.")
    label = current_state_file.read_text().strip()
    state_dir = run_dir / "states" / label
    if not state_dir.exists():
        raise FileNotFoundError(f"State directory {state_dir} not found; cannot load bounds.")
    tensors_path = state_dir / "tensors.pt"
    if not tensors_path.exists():
        raise FileNotFoundError(f"No tensors.pt in {state_dir}; cannot load bounds.")
    tensors = torch.load(tensors_path, map_location=device, weights_only=False)
    if "bounds" not in tensors:
        raise KeyError(f"tensors.pt in {state_dir} has no 'bounds' key.")
    return tensors["bounds"].to(device=device, dtype=dtype)


def run_zombi_main(resume_uuid: str | None = None, dry_run: bool = False):
    """Run DB-driven ZoMBI-Hop loop (new or resume). If dry_run=True, no apparatus communication; synthetic Y only."""
    if dry_run and resume_uuid is None:
        raise ValueError("dry_run requires resume_uuid (resume a run without talking to apparatus).")
    if resume_uuid is None:
        communication.reset_objective()

    dimensions = len(OPTIMIZING_DIMS)
    device = torch.device(DEVICE)
    dtype = torch.float64
    bounds = torch.zeros((2, dimensions), device=device, dtype=dtype)
    bounds[0] = 0.0
    bounds[1] = 1.0

    base_dir = Path("actual_runs")
    base_dir.mkdir(exist_ok=True)
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    if dry_run:
        print("\n" + "=" * 80)
        print("[DRY RUN] No communication with apparatus. Will print suggested endpoints/compositions and use synthetic Y.")
        print("=" * 80 + "\n")

    objective_wrapper = create_db_objective_wrapper(
        objective_function_dry if dry_run else objective_function,
        dimensions=dimensions,
        num_lines=100,
        device=device,
        dry_run=dry_run,
    )

    if resume_uuid is None:
        print("=" * 80)
        print("STARTING NEW ZOMBIHOP TRIAL (DATABASE-DRIVEN)")
        print("=" * 80)
        print(f"Dimensions: {dimensions} (from OPTIMIZING_DIMS: {OPTIMIZING_DIMS})")
        print(f"Device: {device}")
        print("=" * 80 + "\n")

        print("Generating initial data via database...")
        ordered_endpoints = initial_lines_on_boundary(
            NUM_INIT_DATA, bounds, device, dtype=torch.float64
        )

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

        print(f"✅ Starting new trial with UUID: {optimizer.run_uuid}")
    else:
        run_dir = checkpoint_dir / f"run_{resume_uuid}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found for UUID {resume_uuid} (expected {run_dir})")

        bounds_resumed = _load_bounds_from_run(run_dir, device, dtype=dtype)
        optimizer = ZoMBIHop(
            objective=objective_wrapper,
            bounds=bounds_resumed,
            X_init_actual=None,
            X_init_expected=None,
            Y_init=None,
            run_uuid=resume_uuid,
            device=device,
            dtype=dtype,
            checkpoint_dir=str(checkpoint_dir),
            verbose=True,
        )
        print(
            f"✅ Resumed from activation={optimizer.current_activation}, "
            f"zoom={optimizer.current_zoom}, iteration={optimizer.current_iteration}\n"
        )

    print("=" * 80)
    print("STARTING OPTIMIZATION" + (" (DRY RUN - no apparatus)" if dry_run else ""))
    print("=" * 80 + "\n")

    optimizer.run(max_activations=float("inf"), time_limit_hours=None)


if __name__ == "__main__":
    import sys
    resume_uuid = None
    dry_run = False
    if len(sys.argv) >= 2:
        a1 = sys.argv[1].strip().lower()
        if a1 in ("-h", "--help", "help"):
            print("Usage: python -m scripts.run_zombi_main [UUID] [--dry-run]")
            print("  UUID     Resume this run (e.g. 6877).")
            print("  --dry-run  No apparatus communication; print suggested endpoints/compositions and use synthetic Y.")
            print("Example: python -m scripts.run_zombi_main 6877 --dry-run")
            sys.exit(0)
        if a1 == "--dry-run":
            print("Usage: provide resume UUID first, e.g. python -m scripts.run_zombi_main 6877 --dry-run")
            sys.exit(1)
        resume_uuid = sys.argv[1]
    if len(sys.argv) >= 3 and sys.argv[2].strip().lower() == "--dry-run":
        dry_run = True
    run_zombi_main(resume_uuid=resume_uuid, dry_run=dry_run)
