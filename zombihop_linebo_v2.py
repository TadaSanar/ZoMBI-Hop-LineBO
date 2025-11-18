import torch
import json
import time
from pathlib import Path

import numpy as np
import communication
from zombihop_linebo_final import ZoMBIHop, LineBO


# Default Configuration (aligned with test_24hour_variable_minima.py)
DEFAULT_DIMENSIONS = 10
DEFAULT_NUM_MINIMA = 3
DEFAULT_TIME_LIMIT_HOURS = 24.0
NUM_EXPERIMENTS = 24
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Noise parameters (same as test file)
OUTPUT_SIGMA2 = 6.5e-2
INPUT_SIGMA2 = 4.3036e-3

# Landscape parameters (same as the EASY/HARD params used in test file)
HARD_PARAMS = {
    'global_scale': 20.0,
    'exp_scale': 0.2,
    'sharpness': 5.0,
}


# Database interfacing settings (mirror v1 behavior)
OPTIMIZING_DIMS = [0,1,8]


def normalize_last_axis(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    sums = a.sum(axis=-1, keepdims=True)
    return a / sums


def get_y_measurements(x, db="./sql/objective.db", verbose=False, ready_for_objectives=False):
    import sqlite3
    import time as _time
    import os as _os
    consecutive_errors = 0
    max_consecutive_errors = 10

    if ready_for_objectives:
        while True:
            try:
                conn = sqlite3.connect(db, timeout=10.0)
                cur = conn.cursor()
                cur.execute('''CREATE TABLE IF NOT EXISTS handshake (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    new_objective_available INTEGER DEFAULT 0
                )''')
                cur.execute('INSERT OR IGNORE INTO handshake (id, new_objective_available) VALUES (1, 0)')
                cur.execute('SELECT new_objective_available FROM handshake WHERE id = 1')
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
            from communication import _objective_db_lock, _objective_writing
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
                if hasattr(y_all, "mask"):
                    valid_mask &= ~y_all.mask
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) == 0:
                    conn.close()
                    _time.sleep(1)
                    continue
                y = y_all[valid_indices].reshape(-1)
                if len(y) == 0:
                    conn.close()
                    _time.sleep(1)
                    continue
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='compositions'"
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
            cur.execute('UPDATE handshake SET new_objective_available = 0 WHERE id = 1')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[get_y_measurements] Error clearing handshake flag: {e}")
        try:
            mem_db_path = "./sql/objective_memory.db"
            mem_conn = sqlite3.connect(mem_db_path, timeout=10.0)
            mem_cur = mem_conn.cursor()
            mem_cur.execute("DROP TABLE IF EXISTS objective")
            mem_cur.execute("CREATE TABLE objective (val REAL)")
            for val in y:
                mem_cur.execute("INSERT INTO objective (val) VALUES (?)", (float(val),))
            mem_conn.commit()
            mem_conn.close()
            if verbose:
                print(f"[get_y_measurements] Memory DB updated with {len(y)} objectives.")
        except Exception as e:
            print(f"[get_y_measurements] Error updating memory DB: {e}")
    return y, x_meas


def _pad_to_10d(arr):
    arr = np.atleast_2d(arr)
    out = np.zeros((arr.shape[0], 10), dtype=arr.dtype)
    out[:, OPTIMIZING_DIMS] = arr
    return out


def objective_function_init(ordered_endpoints, num_experiments=24):
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
        timestamp=time.time()
    )

    y, x_meas = get_y_measurements(x, verbose=True, ready_for_objectives=False)
    return x_meas, y.ravel()


def objective_function(ordered_endpoints, num_experiments=24):
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
        timestamp=time.time()
    )

    y, x_meas = get_y_measurements(x, verbose=True, ready_for_objectives=True)
    return x_meas, y.ravel()


def generate_distinguishable_minima(dimensions, num_minima, min_distance=0.3, device='cuda'):
    print(f"Generating {num_minima} distinguishable minima in {dimensions}D with min_distance={min_distance}...")
    minima = []
    max_attempts = 10000

    for i in range(num_minima):
        attempts = 0
        while attempts < max_attempts:
            x = torch.rand(dimensions, device=device, dtype=torch.float64)
            x = x / x.sum()

            if len(minima) == 0:
                minima.append(x)
                break

            distances = torch.stack([torch.norm(x - m) for m in minima])
            if torch.all(distances >= min_distance):
                minima.append(x)
                print(f"  Minimum {i+1}/{num_minima}: min_dist to others = {distances.min().item():.4f}")
                break
            attempts += 1

        if attempts >= max_attempts:
            raise ValueError(f"Could not generate {num_minima} distinguishable minima with min_distance={min_distance}")

    minima_tensor = torch.stack(minima)

    print("\nVerifying pairwise distances:")
    for i in range(len(minima)):
        for j in range(i+1, len(minima)):
            dist = torch.norm(minima[i] - minima[j]).item()
            print(f"  Distance between minimum {i+1} and {j+1}: {dist:.4f}")

    print(f"\nSuccessfully generated {num_minima} distinguishable minima!")
    return minima_tensor


def create_multi_minima_objective(minima_locs, params, device='cuda'):
    from test_functions_torch import MultiMinimaAckley

    device = torch.device(device)
    num_minima = minima_locs.shape[0]

    amplitudes = torch.ones(num_minima, device=device, dtype=torch.float64)
    sharpness = params['sharpness'] * torch.ones(num_minima, device=device, dtype=torch.float64)

    func = MultiMinimaAckley(
        minima_locs,
        amplitudes=amplitudes,
        sharpness=sharpness,
        global_scale=params['global_scale'],
        exp_scale=params['exp_scale']
    )

    def _line_objective(endpoints):
        if not torch.is_tensor(endpoints):
            endpoints = torch.tensor(endpoints, device=device, dtype=torch.float64)
        if endpoints.dim() == 2:
            endpoints = endpoints.unsqueeze(0)

        t_values = torch.linspace(0, 1, NUM_EXPERIMENTS, device=device, dtype=torch.float64)
        x_requested = (1 - t_values).unsqueeze(1) * endpoints[0, 0].unsqueeze(0) + \
                     t_values.unsqueeze(1) * endpoints[0, 1].unsqueeze(0)

        input_noise = torch.randn_like(x_requested) * torch.sqrt(torch.tensor(INPUT_SIGMA2, dtype=torch.float64, device=device))
        x_actual = x_requested + input_noise

        x_actual = torch.clamp(x_actual, min=0.0)
        x_actual = x_actual / x_actual.sum(dim=1, keepdim=True)

        y = -func.evaluate(x_actual)
        output_noise = torch.randn_like(y) * torch.sqrt(torch.tensor(OUTPUT_SIGMA2, dtype=torch.float64, device=device))
        y = y + output_noise

        return x_actual, y

    return _line_objective


def _find_trial_directory_by_uuid(base_dir: Path, uuid: str) -> Path | None:
    for td in base_dir.iterdir():
        if not td.is_dir():
            continue
        checkpoints_dir = td / 'checkpoints'
        if checkpoints_dir.exists():
            run_dir = checkpoints_dir / f'run_{uuid}'
            if run_dir.exists():
                return td
    return None


def run_actual_trial(num_minima: int | None = None,
                     dimensions: int | None = None,
                     time_limit_hours: float | None = None,
                     resume_uuid: str | None = None,
                     runs_root: str = 'actual_runs'):
    if num_minima is None:
        num_minima = DEFAULT_NUM_MINIMA
    if dimensions is None:
        dimensions = DEFAULT_DIMENSIONS
    if time_limit_hours is None:
        time_limit_hours = DEFAULT_TIME_LIMIT_HOURS

    device = torch.device(DEVICE)

    base_dir = Path(runs_root)
    base_dir.mkdir(exist_ok=True)

    if resume_uuid is None:
        print("="*80)
        print(f"ZOMBIHOP RUN: {num_minima} Minima in {dimensions}D")
        print("="*80)
        print(f"Dimensions: {dimensions}")
        print(f"Number of minima: {num_minima}")
        print(f"Time limit: {time_limit_hours} hours")
        print(f"Device: {DEVICE}")
        print("="*80 + "\n")

        trial_dir = base_dir / f'trial_{num_minima}minima_{dimensions}d_{int(time_limit_hours)}h'
        trial_dir.mkdir(exist_ok=True)

        minima_locs = generate_distinguishable_minima(dimensions, num_minima, min_distance=0.3, device=device)
        torch.save(minima_locs, trial_dir / 'minima_locations.pt')

        objective_fn = create_multi_minima_objective(minima_locs, HARD_PARAMS, device=device)

        linebo = LineBO(
            objective_fn,
            dimensions=dimensions,
            num_points_per_line=50,
            num_lines=100,
            device=device
        )

        bounds = torch.zeros((2, dimensions), device=device, dtype=torch.float64)
        bounds[0] = 0.0
        bounds[1] = 1.0

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

        print("Initializing ZoMBIHop optimizer...")
        zombihop = ZoMBIHop(
            objective=linebo.sampler,
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
            n_restarts=100,
            raw=10000,
            penalty_num_directions=100,
            penalty_max_radius=0.3,
            penalty_radius_step=0.01,
            max_gp_points=3000,
            device=device,
            checkpoint_dir=str(trial_dir / 'checkpoints')
        )

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

        print(f"Starting new run with UUID: {zombihop.run_uuid}\n")

    else:
        print(f"Resuming run from UUID: {resume_uuid}\n")
        found_dir = _find_trial_directory_by_uuid(base_dir, resume_uuid)
        if found_dir is None:
            print(f"Error: Could not find trial directory containing UUID {resume_uuid}")
            print("Available trials:")
            for td in sorted(base_dir.iterdir()):
                if td.is_dir():
                    print(f"  {td.name}")
            return

        trial_dir = found_dir
        metadata_file = trial_dir / 'trial_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            dimensions = metadata.get('dimensions', dimensions)
            num_minima = metadata.get('num_minima', num_minima)
            time_limit_hours = metadata.get('time_limit_hours', time_limit_hours)
            print(f"Loaded original parameters: {num_minima} minima, {dimensions}D, {time_limit_hours}h")

        minima_locs = torch.load(trial_dir / 'minima_locations.pt', map_location=device)
        objective_fn = create_multi_minima_objective(minima_locs, HARD_PARAMS, device=device)
        linebo = LineBO(
            objective_fn,
            dimensions=dimensions,
            num_points_per_line=50,
            num_lines=100,
            device=device
        )

        zombihop = ZoMBIHop(
            objective=linebo.sampler,
            bounds=None,
            X_init_actual=None,
            X_init_expected=None,
            Y_init=None,
            run_uuid=resume_uuid,
            device=device,
            checkpoint_dir=str(trial_dir / 'checkpoints')
        )

        print(f"Resumed from activation={zombihop.current_activation}, "
              f"zoom={zombihop.current_zoom}, iteration={zombihop.current_iteration}\n")

    print("="*80)
    print("STARTING OPTIMIZATION")
    print("="*80 + "\n")

    trial_start = time.time()
    needles_results, needles, needle_vals, X_all_actual, Y_all = zombihop.run(
        max_activations=float('inf'),
        time_limit_hours=time_limit_hours
    )
    trial_end = time.time()
    trial_duration = (trial_end - trial_start) / 3600.0

    print("\n" + "="*80)
    print("RUN COMPLETE")
    print("="*80)
    print(f"Trial duration: {trial_duration:.2f} hours")
    print(f"Total points sampled: {X_all_actual.shape[0]}")
    print(f"Needles found: {needles.shape[0]}")
    print(f"Run UUID: {zombihop.run_uuid}")
    print("="*80 + "\n")

    results = {
        'run_uuid': zombihop.run_uuid,
        'trial_duration_hours': trial_duration,
        'total_points': X_all_actual.shape[0],
        'num_needles': needles.shape[0],
        'best_value': Y_all.max().item() if Y_all.numel() > 0 else None,
    }
    results_file = trial_dir / f'results_{zombihop.run_uuid}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")


def run_zombi_main_v2():
    communication.reset_objective()

    dimensions = len(OPTIMIZING_DIMS)
    n_experiments = 24

    # Create bounds for the 3D simplex subspace that maps to OPTIMIZING_DIMS
    device = torch.device(DEVICE)
    bounds = torch.zeros((2, dimensions), device=device, dtype=torch.float64)
    bounds[0] = 0.0
    bounds[1] = 1.0

    # LineBO for initialization using DB (no handshake)
    linebo_init = LineBO(
        objective_function_init,
        dimensions=dimensions,
        num_points_per_line=100,
        num_lines=50,
        device=device
    )

    # Generate one or more random simplex start points and collect initial data via DB
    num_start_points = 1
    start_points = ZoMBIHop.random_simplex(num_start_points, bounds[0], bounds[1], device=device)

    X_init_actual = torch.empty((0, dimensions), device=device, dtype=torch.float64)
    X_init_expected = torch.empty((0, dimensions), device=device, dtype=torch.float64)
    Y_init = torch.empty((0, 1), device=device, dtype=torch.float64)

    for point in start_points:
        x_requested, x_actual, y = linebo_init.sampler(point, bounds=bounds)
        X_init_actual = torch.cat([X_init_actual, x_actual], dim=0)
        X_init_expected = torch.cat([X_init_expected, x_requested], dim=0)
        Y_init = torch.cat([Y_init, y.unsqueeze(1)], dim=0)

    # Main LineBO using DB handshake objective
    linebo = LineBO(
        objective_function,
        dimensions=dimensions,
        num_points_per_line=100,
        num_lines=100,
        device=device
    )

    # Initialize ZoMBIHop (final implementation) with DB-backed LineBO sampler
    optimizer = ZoMBIHop(
        objective=linebo.sampler,
        bounds=bounds,
        X_init_actual=X_init_actual,
        X_init_expected=X_init_expected,
        Y_init=Y_init,
        penalization_threshold=1e-3,
        improvement_threshold_noise_mult=1.5,
        input_noise_threshold_mult=3.4,
        n_consecutive_no_improvements=5,
        top_m_points=3,
        max_zooms=2,
        max_iterations=10,
        n_restarts=50,
        raw=5000,
        penalty_num_directions=100,
        penalty_max_radius=0.3,
        penalty_radius_step=0.01,
        max_gp_points=3000,
        device=device,
        checkpoint_dir=str(Path('actual_runs') / 'checkpoints')
    )

    # Run indefinitely (or until external stop) with no synthetic data
    optimizer.run(max_activations=float('inf'), time_limit_hours=None)


