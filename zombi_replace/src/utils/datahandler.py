"""
Data Handler for ZoMBI-Hop
==========================

Stores ALL data, control variables, and iteration state for ZoMBI-Hop.
All hyperparameters live here as plain attributes and are accessed directly.

Call take_snapshot(label) at any time to save a full checkpoint to disk.
Call load_state() to resume from the latest snapshot.

Backward-compat note: push_checkpoint() is kept as an alias for take_snapshot()
so that existing code (zombihop_exp.py) continues to work unchanged.
"""

import re
import torch
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

from .dataclasses import ZoMBIHopConfig


class DataHandler:
    """
    Stores all data, state, and control variables for ZoMBI-Hop.

    All hyperparameters are plain attributes (e.g. self.max_zooms).
    Call take_snapshot(label) to save everything to disk at any point.

    Parameters
    ----------
    max_zooms, max_iterations, top_m_points, n_restarts, raw,
    convergence_pi_threshold, input_noise_threshold_mult,
    output_noise_threshold_mult, n_consecutive_converged, max_gp_points,
    repulsion_lambda, acquisition_type, ucb_beta, nat_grad_step,
    nat_grad_max_steps : control variables
    directory : str, optional
        Base directory for snapshots. If None, no saving occurs.
    run_uuid : str, optional
        UUID to resume an existing run.
    max_saved_recent_checkpoints : int, optional
        Kept for backward compatibility. Use max_snapshots instead.
    max_snapshots : int, optional
        Max snapshots to keep (None = keep all).
    device, dtype : compute settings
    config : ZoMBIHopConfig or dict, optional
        If provided, values override the individual keyword arguments.
        Kept for backward compatibility.
    d : int, optional
        Dimensionality (auto-set in save_init).
    """

    def __init__(
        self,
        # --- Control variables ---
        max_zooms: int = 3,
        max_iterations: int = 10,
        top_m_points: Optional[int] = None,
        n_restarts: int = 30,
        raw: int = 500,
        convergence_pi_threshold: float = 0.01,
        input_noise_threshold_mult: float = 2.0,
        output_noise_threshold_mult: float = 2.0,
        n_consecutive_converged: int = 2,
        max_gp_points: int = 3000,
        repulsion_lambda: Optional[float] = None,
        acquisition_type: str = "ucb",
        ucb_beta: float = 0.1,
        nat_grad_step: float = 0.02,
        nat_grad_max_steps: int = 50,
        # --- Storage settings ---
        directory: Optional[str] = None,
        run_uuid: Optional[str] = None,
        max_saved_recent_checkpoints: Optional[int] = 50,  # backward compat
        max_snapshots: Optional[int] = None,
        # --- Compute settings ---
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64,
        # --- Over-penalization escape hatch ---
        old_needle_radius_mult: float = 3.0,
        retry_raw_scale: float = 2.0,
        retry_step_scale: float = 0.5,
        # --- Backward compat ---
        config: Optional[Union[ZoMBIHopConfig, Dict[str, Any]]] = None,
        d: Optional[int] = None,
    ):
        # If a config object/dict is provided, extract values from it
        if config is not None:
            cfg = config if isinstance(config, ZoMBIHopConfig) else ZoMBIHopConfig.from_dict(config)
            max_zooms = cfg.max_zooms
            max_iterations = cfg.max_iterations
            if cfg.top_m_points is not None:
                top_m_points = cfg.top_m_points
            n_restarts = cfg.n_restarts
            raw = cfg.raw
            convergence_pi_threshold = cfg.convergence_pi_threshold
            input_noise_threshold_mult = cfg.input_noise_threshold_mult
            output_noise_threshold_mult = cfg.output_noise_threshold_mult
            n_consecutive_converged = cfg.n_consecutive_converged
            max_gp_points = cfg.max_gp_points
            repulsion_lambda = cfg.repulsion_lambda
            acquisition_type = getattr(cfg, 'acquisition_type', acquisition_type)
            ucb_beta = getattr(cfg, 'ucb_beta', ucb_beta)
            nat_grad_step = getattr(cfg, 'nat_grad_step', nat_grad_step)
            nat_grad_max_steps = getattr(cfg, 'nat_grad_max_steps', nat_grad_max_steps)

        # --- Store all control variables as plain attributes ---
        self.max_zooms = max_zooms
        self.max_iterations = max_iterations
        self.top_m_points = top_m_points
        self.n_restarts = n_restarts
        self.raw = raw
        self.convergence_pi_threshold = convergence_pi_threshold
        self.input_noise_threshold_mult = input_noise_threshold_mult
        self.output_noise_threshold_mult = output_noise_threshold_mult
        self.n_consecutive_converged = n_consecutive_converged
        self.max_gp_points = max_gp_points
        self.repulsion_lambda = repulsion_lambda
        self.acquisition_type = acquisition_type
        self.ucb_beta = ucb_beta
        self.nat_grad_step = nat_grad_step
        self.nat_grad_max_steps = nat_grad_max_steps

        # Over-penalization escape hatch params
        self.old_needle_radius_mult = old_needle_radius_mult
        self.retry_raw_scale = retry_raw_scale
        self.retry_step_scale = retry_step_scale

        # Compute settings
        self.device = torch.device(device)
        self.dtype = dtype
        self.d = d

        # Storage settings
        self.max_snapshots = max_snapshots
        self.save_enabled = directory is not None
        self._snapshot_count = 0

        # Backward compat: max_saved_recent_checkpoints drives cleanup if max_snapshots not set
        if self.max_snapshots is None and max_saved_recent_checkpoints:
            self.max_snapshots = max_saved_recent_checkpoints
        self.max_saved_recent_checkpoints = max_saved_recent_checkpoints or 0

        # Initialize in-memory state
        self._init_storage()

        # Checkpoint tracking (kept for backward compat with push_checkpoint)
        self.checkpoint_history: List[Tuple[str, bool]] = []

        # Set up run directory and UUID
        if self.save_enabled:
            base_dir = Path(directory)
            base_dir.mkdir(exist_ok=True)
            if run_uuid is not None:
                self.run_uuid = run_uuid
                self.run_dir = base_dir / f"run_{run_uuid}"
                if not self.run_dir.exists():
                    raise ValueError(f"Run directory {self.run_dir} does not exist!")
            else:
                self.run_uuid = str(uuid.uuid4())[:4]
                self.run_dir = base_dir / f"run_{self.run_uuid}"
                self.run_dir.mkdir(exist_ok=True)
        else:
            self.run_uuid = run_uuid or str(uuid.uuid4())[:4]
            self.checkpoint_base_dir = None
            self.run_dir = None

    def _init_storage(self):
        """Initialize all in-memory tensors and state to defaults."""
        self.X_all_actual: Optional[torch.Tensor] = None
        self.X_all_expected: Optional[torch.Tensor] = None
        self.Y_all: Optional[torch.Tensor] = None
        self.X_init_actual: Optional[torch.Tensor] = None
        self.X_init_expected: Optional[torch.Tensor] = None
        self.Y_init: Optional[torch.Tensor] = None
        self.bounds: Optional[torch.Tensor] = None
        # Tracks the zoomed-in bounds at the current zoom level (may differ from
        # self.bounds which holds the activation-start / full-simplex bounds).
        self.current_zoom_bounds: Optional[torch.Tensor] = None

        self.needles: Optional[torch.Tensor] = None
        self.needle_vals: Optional[torch.Tensor] = None
        self.needle_indices: Optional[torch.Tensor] = None
        self.needle_penalty_radii: Optional[torch.Tensor] = None
        self.needles_results: List[Dict[str, Any]] = []

        # Per-needle ellipsoid parameters (None entry = fall back to sphere radius).
        # needle_B is shared across all needles (same simplex tangent space).
        self.needle_M_list: List[Optional[torch.Tensor]] = []  # each (d-1, d-1)
        self.needle_B: Optional[torch.Tensor] = None           # (d, d-1)

        # Old needles: converted from regular needles when an activation fails to
        # converge. They use uniform-sphere penalisation of radius
        # old_needle_radius_mult * input_noise, and are treated as boundary
        # constraints (not exclusion zones) during hyperrectangle computation.
        self.old_needles: Optional[torch.Tensor] = None        # (k, d)
        self.old_needle_vals: Optional[torch.Tensor] = None    # (k, 1)
        self.old_needle_radii: Optional[torch.Tensor] = None   # (k, 1)
        self.old_needle_radius_mult: float = 3.0               # configurable

        self._penalty_mask: Optional[torch.Tensor] = None

        # Iteration state
        self.current_activation: int = 0
        self.current_zoom: int = 0
        self.current_iteration: int = 0
        self.no_improvements: int = 0  # kept for backward compat

        # Logging
        self.log_ei_history: List[float] = []

    def save_init(
        self,
        X_init_actual: torch.Tensor,
        X_init_expected: torch.Tensor,
        Y_init: torch.Tensor,
        bounds: torch.Tensor,
    ):
        """
        Set up data storage with initial observations.
        Call once before optimization starts (not needed when resuming).
        Auto-computes top_m_points if not already set.
        """
        self.d = X_init_actual.shape[1]
        self.bounds = bounds.clone().to(device=self.device, dtype=self.dtype)

        if self.top_m_points is None:
            self.top_m_points = max(self.d + 1, 4)

        self.X_init_actual = X_init_actual.clone().to(device=self.device, dtype=self.dtype)
        self.X_init_expected = X_init_expected.clone().to(device=self.device, dtype=self.dtype)
        self.Y_init = Y_init.clone().to(device=self.device, dtype=self.dtype)

        self.X_all_actual = self.X_init_actual.clone()
        self.X_all_expected = self.X_init_expected.clone()
        self.Y_all = self.Y_init.clone()

        self.needles = torch.empty((0, self.d), device=self.device, dtype=self.dtype)
        self.needle_vals = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.needle_indices = torch.empty((0, 1), device=self.device, dtype=torch.int64)
        self.needle_penalty_radii = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.needle_M_list = []
        self.needle_B = None

        self._update_penalty_mask()

        if self.save_enabled:
            self._save_config()
            self.take_snapshot("init", permanent=True)

    def _save_config(self):
        """Write config.json once at init (human-readable, for inspection)."""
        config = {
            'run_uuid': self.run_uuid,
            'd': self.d,
            'max_zooms': self.max_zooms,
            'max_iterations': self.max_iterations,
            'top_m_points': self.top_m_points,
            'n_restarts': self.n_restarts,
            'raw': self.raw,
            'convergence_pi_threshold': self.convergence_pi_threshold,
            'input_noise_threshold_mult': self.input_noise_threshold_mult,
            'output_noise_threshold_mult': self.output_noise_threshold_mult,
            'n_consecutive_converged': self.n_consecutive_converged,
            'max_gp_points': self.max_gp_points,
            'repulsion_lambda': self.repulsion_lambda,
            'acquisition_type': self.acquisition_type,
            'ucb_beta': self.ucb_beta,
            'nat_grad_step': self.nat_grad_step,
            'nat_grad_max_steps': self.nat_grad_max_steps,
            'device': str(self.device),
            'dtype': str(self.dtype),
        }
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # =========================================================================
    # Snapshotting (new, simple system)
    # =========================================================================

    def take_snapshot(
        self,
        label: str = "",
        permanent: bool = False,
        activation: Optional[int] = None,
        zoom: Optional[int] = None,
        iteration: Optional[int] = None,
    ):
        """
        Save complete state to disk.

        Optionally updates current_activation/zoom/iteration before saving so
        a single call replaces the old update_iteration_state + take_snapshot pair.

        Saves all tensors, needle results, iteration state, and a summary.
        Snapshots are numbered sequentially under run_dir/snapshots/.
        Removes oldest non-permanent snapshots if max_snapshots is set.
        Permanent snapshots are never cleaned up.
        """
        if activation is not None:
            self.current_activation = activation
        if zoom is not None:
            self.current_zoom = zoom
        if iteration is not None:
            self.current_iteration = iteration

        if not self.save_enabled:
            return

        self._snapshot_count += 1
        name = f"{self._snapshot_count:04d}_{label}" if label else f"{self._snapshot_count:04d}"
        snapshot_dir = self.run_dir / "snapshots" / name
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        if permanent:
            (snapshot_dir / 'permanent').touch()

        # Serialise ellipsoid M matrices (list of Optional tensors) as a stacked
        # tensor + boolean has_M flag so None entries survive serialisation.
        if self.needle_M_list:
            # Infer (d-1) from first non-None entry or from needle_B
            dm1 = None
            for m in self.needle_M_list:
                if m is not None:
                    dm1 = m.shape[0]; break
            if dm1 is None and self.needle_B is not None:
                dm1 = self.needle_B.shape[1]
            if dm1 is None:
                dm1 = max(self.d - 1, 1)
            needle_has_M = torch.tensor(
                [m is not None for m in self.needle_M_list], dtype=torch.bool, device=self.device
            )
            needle_M_stack = torch.stack([
                m.to(device=self.device, dtype=self.dtype) if m is not None
                else torch.zeros(dm1, dm1, device=self.device, dtype=self.dtype)
                for m in self.needle_M_list
            ], dim=0)
        else:
            needle_has_M = torch.zeros(0, dtype=torch.bool, device=self.device)
            needle_M_stack = torch.zeros(0, 1, 1, device=self.device, dtype=self.dtype)

        # All tensors in one file
        torch.save({
            'bounds': self.bounds,
            'current_zoom_bounds': self.current_zoom_bounds,
            'X_init_actual': self.X_init_actual,
            'X_init_expected': self.X_init_expected,
            'Y_init': self.Y_init,
            'X_all_actual': self.X_all_actual,
            'X_all_expected': self.X_all_expected,
            'Y_all': self.Y_all,
            'needles': self.needles,
            'needle_vals': self.needle_vals,
            'needle_indices': self.needle_indices,
            'needle_penalty_radii': self.needle_penalty_radii,
            'needle_M_stack': needle_M_stack,
            'needle_has_M': needle_has_M,
            'needle_B': self.needle_B,
            'old_needles': self.old_needles,
            'old_needle_vals': self.old_needle_vals,
            'old_needle_radii': self.old_needle_radii,
            'penalty_mask': self._penalty_mask,
        }, snapshot_dir / 'tensors.pt')

        # Needle results as JSON
        needles_json = [
            {
                'point': r['point'].cpu().tolist(),
                'value': r['value'],
                'activation': r['activation'],
                'zoom': r['zoom'],
                'iteration': r['iteration'],
            }
            for r in self.needles_results
        ]
        with open(snapshot_dir / 'needles.json', 'w') as f:
            json.dump(needles_json, f, indent=2)

        # Human-readable summary
        summary = {
            'label': label,
            'timestamp': time.time(),
            'activation': self.current_activation,
            'zoom': self.current_zoom,
            'iteration': self.current_iteration,
            'n_points': self.X_all_actual.shape[0] if self.X_all_actual is not None else 0,
            'n_needles': self.needles.shape[0] if self.needles is not None else 0,
            'best_y': self.Y_all.max().item() if self.Y_all is not None and self.Y_all.numel() > 0 else None,
            'best_y_unpenalized': (
                self.Y_all[self._penalty_mask].max().item()
                if self._penalty_mask is not None and self._penalty_mask.any() else None
            ),
        }
        with open(snapshot_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Track latest
        with open(self.run_dir / 'latest.txt', 'w') as f:
            f.write(name)

        if self.max_snapshots is not None:
            self._cleanup_old_snapshots()

    def _cleanup_old_snapshots(self):
        """Remove oldest non-permanent snapshots, keeping the last max_snapshots."""
        snapshots_dir = self.run_dir / "snapshots"
        if not snapshots_dir.exists():
            return
        all_snapshots = sorted(snapshots_dir.iterdir())
        non_permanent = [s for s in all_snapshots if not (s / 'permanent').exists()]
        for old in non_permanent[:-self.max_snapshots]:
            shutil.rmtree(old, ignore_errors=True)

    # =========================================================================
    # Checkpointing (kept for backward compatibility with zombihop_exp.py)
    # =========================================================================

    def push_checkpoint(self, label: str, is_permanent: bool = False):
        """Backward-compat wrapper: calls take_snapshot(label, permanent=is_permanent)."""
        self.take_snapshot(label, permanent=is_permanent)

    # =========================================================================
    # State loading
    # =========================================================================

    def load_state(self) -> Tuple[int, int, int, int]:
        """
        Load state from the latest snapshot on disk.
        Returns (current_activation, current_zoom, current_iteration, no_improvements).
        """
        if not self.save_enabled or not self.run_dir.exists():
            return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

        # Load config
        config_path = self.run_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            self.d = cfg.get('d', self.d)
            self.max_zooms = cfg.get('max_zooms', self.max_zooms)
            self.max_iterations = cfg.get('max_iterations', self.max_iterations)
            self.top_m_points = cfg.get('top_m_points', self.top_m_points)
            self.n_restarts = cfg.get('n_restarts', self.n_restarts)
            self.raw = cfg.get('raw', self.raw)
            self.convergence_pi_threshold = cfg.get('convergence_pi_threshold', self.convergence_pi_threshold)
            self.input_noise_threshold_mult = cfg.get('input_noise_threshold_mult', self.input_noise_threshold_mult)
            self.output_noise_threshold_mult = cfg.get('output_noise_threshold_mult', self.output_noise_threshold_mult)
            self.n_consecutive_converged = cfg.get('n_consecutive_converged', self.n_consecutive_converged)
            self.max_gp_points = cfg.get('max_gp_points', self.max_gp_points)
            self.repulsion_lambda = cfg.get('repulsion_lambda', self.repulsion_lambda)
            self.acquisition_type = cfg.get('acquisition_type', self.acquisition_type)
            self.ucb_beta = cfg.get('ucb_beta', self.ucb_beta)
            self.nat_grad_step = cfg.get('nat_grad_step', self.nat_grad_step)
            self.nat_grad_max_steps = cfg.get('nat_grad_max_steps', self.nat_grad_max_steps)

        # Try new snapshot format first (latest.txt -> snapshots/)
        latest_path = self.run_dir / 'latest.txt'
        if latest_path.exists():
            return self._load_from_snapshot(latest_path.read_text().strip())

        # Fall back to old checkpoint format (current_state.txt -> states/)
        current_state_file = self.run_dir / 'current_state.txt'
        if current_state_file.exists():
            return self._load_from_old_checkpoint(current_state_file.read_text().strip())

        return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

    def _load_from_snapshot(self, snapshot_name: str) -> Tuple[int, int, int, int]:
        """Load state from a snapshot directory (new format)."""
        snapshot_dir = self.run_dir / 'snapshots' / snapshot_name
        if not snapshot_dir.exists():
            return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

        tensors = torch.load(snapshot_dir / 'tensors.pt', map_location=self.device)
        self._load_tensors(tensors)

        summary_path = snapshot_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            self.current_activation = summary.get('activation', 0)
            self.current_zoom = summary.get('zoom', 0)
            self.current_iteration = summary.get('iteration', 0)

        self._load_needles_json(snapshot_dir / 'needles.json')

        # Set snapshot count from existing snapshots
        snapshots_dir = self.run_dir / 'snapshots'
        if snapshots_dir.exists():
            existing = sorted(snapshots_dir.iterdir())
            if existing:
                try:
                    self._snapshot_count = int(existing[-1].name.split('_')[0])
                except ValueError:
                    self._snapshot_count = len(existing)

        return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

    def _load_from_old_checkpoint(self, iteration_label: str) -> Tuple[int, int, int, int]:
        """Load state from old checkpoint format (states/ directory)."""
        state_dir = self.run_dir / 'states' / iteration_label
        if not state_dir.exists():
            return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

        tensors = torch.load(state_dir / 'tensors.pt', map_location=self.device)
        self._load_tensors(tensors)

        tracking_path = state_dir / 'tracking.json'
        if tracking_path.exists():
            with open(tracking_path) as f:
                tracking = json.load(f)
            self.current_activation = tracking['current_activation']
            self.current_zoom = tracking['current_zoom']
            self.current_iteration = tracking['current_iteration']
            self.no_improvements = tracking.get('no_improvements', 0)

        # State label is authoritative for position
        match = re.match(r'act(\d+)_zoom(\d+)_iter(\d+)', iteration_label)
        if match:
            self.current_activation = int(match.group(1))
            self.current_zoom = int(match.group(2))
            self.current_iteration = int(match.group(3))

        self._load_needles_json(state_dir / 'needles_results.json')

        # Rebuild checkpoint history
        states_dir = self.run_dir / 'states'
        if states_dir.exists():
            for state_subdir in sorted(states_dir.iterdir()):
                if state_subdir.is_dir():
                    label = state_subdir.name
                    is_permanent = any(keyword in label for keyword in
                                       ['init', 'needle', 'complete', 'finished', 'timeout', 'final'])
                    self.checkpoint_history.append((label, is_permanent))

        return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

    def _load_tensors(self, tensors: dict):
        """Load all tensors from a checkpoint dict."""
        self.bounds = tensors['bounds'].to(device=self.device, dtype=self.dtype)
        czb = tensors.get('current_zoom_bounds', None)
        self.current_zoom_bounds = (
            czb.to(device=self.device, dtype=self.dtype)
            if czb is not None
            else self.bounds.clone()
        )
        self.X_init_actual = tensors['X_init_actual'].to(device=self.device, dtype=self.dtype)
        self.X_init_expected = tensors['X_init_expected'].to(device=self.device, dtype=self.dtype)
        self.Y_init = tensors['Y_init'].to(device=self.device, dtype=self.dtype)
        self.X_all_actual = tensors['X_all_actual'].to(device=self.device, dtype=self.dtype)
        self.X_all_expected = tensors['X_all_expected'].to(device=self.device, dtype=self.dtype)
        self.Y_all = tensors['Y_all'].to(device=self.device, dtype=self.dtype)
        self.needles = tensors['needles'].to(device=self.device, dtype=self.dtype)
        self.needle_vals = tensors['needle_vals'].to(device=self.device, dtype=self.dtype)
        self.needle_indices = tensors['needle_indices'].to(device=self.device, dtype=torch.int64)
        self.needle_penalty_radii = tensors['needle_penalty_radii'].to(device=self.device, dtype=self.dtype)
        self._penalty_mask = tensors['penalty_mask'].to(device=self.device)

        # Restore ellipsoid data (absent in old checkpoints → safe defaults)
        needle_M_stack = tensors.get('needle_M_stack', None)
        needle_has_M = tensors.get('needle_has_M', None)
        if needle_M_stack is not None and needle_has_M is not None and needle_has_M.numel() > 0:
            needle_M_stack = needle_M_stack.to(device=self.device, dtype=self.dtype)
            needle_has_M = needle_has_M.to(device=self.device)
            self.needle_M_list = [
                needle_M_stack[i].clone() if needle_has_M[i].item() else None
                for i in range(needle_has_M.shape[0])
            ]
        else:
            self.needle_M_list = [None] * self.needles.shape[0]

        nb = tensors.get('needle_B', None)
        self.needle_B = nb.to(device=self.device, dtype=self.dtype) if nb is not None else None

        # Restore old_needles (absent in old checkpoints → empty)
        on = tensors.get('old_needles', None)
        self.old_needles = on.to(device=self.device, dtype=self.dtype) if on is not None else None
        onv = tensors.get('old_needle_vals', None)
        self.old_needle_vals = onv.to(device=self.device, dtype=self.dtype) if onv is not None else None
        onr = tensors.get('old_needle_radii', None)
        self.old_needle_radii = onr.to(device=self.device, dtype=self.dtype) if onr is not None else None

    def _load_needles_json(self, path: Path):
        """Load needle results from a JSON file (handles both old and new key names)."""
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self.needles_results = [
            {
                'point': torch.tensor(r['point'], device=self.device, dtype=self.dtype),
                'value': r['value'],
                'activation': r['activation'],
                'zoom': r['zoom'],
                'iteration': r['iteration'],
            }
            for r in data
        ]

    # =========================================================================
    # Iteration state helpers (kept for backward compat with zombihop_exp.py)
    # =========================================================================

    def update_iteration_state(self, activation: int, zoom: int, iteration: int, no_improvements: int):
        """Update iteration tracking state."""
        self.current_activation = activation
        self.current_zoom = zoom
        self.current_iteration = iteration
        self.no_improvements = no_improvements

    def get_iteration_state(self) -> Tuple[int, int, int, int]:
        """Get current iteration state."""
        return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

    # =========================================================================
    # Data management
    # =========================================================================

    def add_all_points(
        self,
        new_X_actual: torch.Tensor,
        new_X_expected: torch.Tensor,
        new_Y: torch.Tensor,
    ) -> torch.Tensor:
        """Add new observations. Returns penalty mask for the new points (True = not penalized)."""
        new_X_actual = new_X_actual.to(device=self.device, dtype=self.dtype)
        new_X_expected = new_X_expected.to(device=self.device, dtype=self.dtype)
        new_Y = new_Y.to(device=self.device, dtype=self.dtype)

        if new_Y.ndim == 1:
            new_Y = new_Y.unsqueeze(1)

        new_penalty_mask = self._compute_penalty_mask(new_X_actual)

        self.X_all_actual = torch.cat([self.X_all_actual, new_X_actual], dim=0)
        self.X_all_expected = torch.cat([self.X_all_expected, new_X_expected], dim=0)
        self.Y_all = torch.cat([self.Y_all, new_Y], dim=0)
        self._penalty_mask = torch.cat([self._penalty_mask, new_penalty_mask], dim=0)

        return new_penalty_mask

    def get_all_points(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (X_all_actual, X_all_expected, Y_all)."""
        return self.X_all_actual, self.X_all_expected, self.Y_all

    def add_needle(
        self,
        needle: torch.Tensor,
        needle_value: float,
        needle_penalty_radius: float,
        activation: int,
        zoom: int,
        iteration: int,
        M: Optional[torch.Tensor] = None,
        B: Optional[torch.Tensor] = None,
    ):
        """Record a discovered needle (local optimum) and update penalty mask.

        If M and B are provided (the tangent-space Hessian ellipsoid), they are
        stored and used for the ellipsoid penalty mask instead of the sphere.
        """
        needle = needle.to(device=self.device, dtype=self.dtype)

        distances = torch.norm(self.X_all_actual - needle.unsqueeze(0), dim=1)
        global_idx = distances.argmin()

        self.needles = torch.cat([self.needles, needle.unsqueeze(0)], dim=0)
        self.needle_vals = torch.cat([
            self.needle_vals,
            torch.tensor([[needle_value]], device=self.device, dtype=self.dtype),
        ], dim=0)
        self.needle_indices = torch.cat([self.needle_indices, global_idx.reshape(1, 1)], dim=0)
        self.needle_penalty_radii = torch.cat([
            self.needle_penalty_radii,
            torch.tensor([[needle_penalty_radius]], device=self.device, dtype=self.dtype),
        ], dim=0)

        # Store ellipsoid (or None for sphere fallback)
        self.needle_M_list.append(M.to(device=self.device, dtype=self.dtype) if M is not None else None)
        if B is not None:
            self.needle_B = B.to(device=self.device, dtype=self.dtype)

        self.needles_results.append({
            'point': needle.clone(),
            'value': needle_value,
            'activation': activation,
            'zoom': zoom,
            'iteration': iteration,
        })

        self._update_penalty_mask()

    def get_needle_locations(self) -> torch.Tensor:
        """Return (num_needles, d) tensor of needle locations."""
        return self.needles

    def get_needle_results(self) -> List[Dict[str, Any]]:
        """Return list of needle result dicts."""
        return self.needles_results

    def get_needles_and_penalty_radii(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (needles, penalty_radii) tensors."""
        return self.needles, self.needle_penalty_radii

    def get_needle_ellipsoids(self) -> Tuple[List[Optional[torch.Tensor]], Optional[torch.Tensor]]:
        """Return (needle_M_list, needle_B) for use in RepulsiveAcquisition."""
        return self.needle_M_list, self.needle_B

    # =========================================================================
    # Penalty mask
    # =========================================================================

    def get_penalty_mask(self, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return penalty mask (True = not penalized). If X is None, returns mask for all stored points."""
        if X is None:
            return self._penalty_mask
        return self._compute_penalty_mask(X)

    def _compute_penalty_mask(self, X: torch.Tensor) -> torch.Tensor:
        """Compute penalty mask for given points. True = not inside any penalty region.

        For needles with an ellipsoid (M, B): uses u^T M u <= 1 membership test.
        For needles with only a radius: uses Euclidean sphere test.
        Also applies sphere tests for old_needles (if any).
        """
        is_2d = X.ndim == 2
        if is_2d:
            n = X.shape[0]
            X_flat = X  # (n, d)
        elif X.ndim == 3:
            n, l, d_x = X.shape
            X_flat = X.reshape(-1, d_x)
        else:
            raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")

        num_pts = X_flat.shape[0]
        penalized = torch.zeros(num_pts, dtype=torch.bool, device=X.device)

        # --- Regular needles ---
        if self.needles is not None and self.needles.shape[0] > 0:
            for idx in range(self.needles.shape[0]):
                needle = self.needles[idx]  # (d,)
                diff = X_flat - needle.unsqueeze(0)  # (num_pts, d)

                M = self.needle_M_list[idx] if idx < len(self.needle_M_list) else None
                if M is not None and self.needle_B is not None:
                    u = diff @ self.needle_B           # (num_pts, d-1)
                    quad = (u @ M * u).sum(dim=-1)     # (num_pts,)
                    inside = quad <= 1.0
                else:
                    r = self.needle_penalty_radii[idx].squeeze()
                    inside = torch.norm(diff, dim=-1) <= r
                penalized = penalized | inside

        # --- Old needles (sphere only) ---
        if self.old_needles is not None and self.old_needles.shape[0] > 0:
            old_diff = X_flat.unsqueeze(1) - self.old_needles.unsqueeze(0)  # (num_pts, k, d)
            old_dist = torch.norm(old_diff, dim=-1)                          # (num_pts, k)
            old_radii = self.old_needle_radii.view(1, -1)                    # (1, k)
            old_inside = (old_dist <= old_radii).any(dim=1)                  # (num_pts,)
            penalized = penalized | old_inside

        if not is_2d:
            penalized = penalized.reshape(n, l)

        return ~penalized

    def _update_penalty_mask(self):
        """Recompute cached penalty mask for all stored points."""
        if self.X_all_actual is not None and self.X_all_actual.shape[0] > 0:
            self._penalty_mask = self._compute_penalty_mask(self.X_all_actual)
        else:
            self._penalty_mask = torch.ones(0, dtype=torch.bool, device=self.device)

    # =========================================================================
    # GP helpers
    # =========================================================================

    def get_gp_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (X, Y) for GP fitting.

        Prefers the top ``max_gp_points`` unpenalized points.  Falls back to
        ALL stored data when every point is penalized (e.g. at the start of a
        new activation after a large penalty radius has absorbed the previous
        activation's observations).  The repulsive acquisition already steers
        the optimizer away from needles, so training on all data is safe.
        """
        mask = self._penalty_mask
        if mask.any():
            X = self.X_all_actual[mask]
            Y = self.Y_all[mask]
        else:
            # All points penalized — fall back to full dataset
            X = self.X_all_actual
            Y = self.Y_all
        sorted_idx = torch.argsort(Y.squeeze(), descending=True)
        n = min(self.max_gp_points, len(sorted_idx))
        top_idx = sorted_idx[:n]
        return X[top_idx], Y[top_idx]

    def get_best_unpenalized(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:
        """Return (X_best, Y_best, global_index) for the best unpenalized point."""
        if not self._penalty_mask.any():
            return None, None, None
        Y_masked = self.Y_all[self._penalty_mask]
        max_idx = Y_masked.argmax()
        unpenalized_indices = torch.where(self._penalty_mask)[0]
        global_idx = unpenalized_indices[max_idx]
        return self.X_all_actual[global_idx], self.Y_all[global_idx], global_idx.item()

    def determine_new_bounds(self) -> torch.Tensor:
        """Compute new bounds around the top_m_points best unpenalized points."""
        Y_masked = self.Y_all[self._penalty_mask].squeeze(-1)
        k = min(self.top_m_points, Y_masked.numel())
        top_idx = torch.topk(Y_masked, k).indices
        X_top = self.X_all_actual[self._penalty_mask][top_idx]
        return torch.stack([X_top.min(dim=0).values, X_top.max(dim=0).values], dim=0)

    # =========================================================================
    # Over-penalization escape hatch
    # =========================================================================

    def move_needles_to_old(self):
        """
        Demote all current needles to 'old_needles' with uniform sphere radius
        old_needle_radius_mult * input_noise, then reset the regular needle lists.

        Old needles:
        - Still act as exclusion zones in _compute_penalty_mask (sphere).
        - Are treated as boundary constraints (not exclusion zones) in
          determine_new_bounds_constrained.
        """
        if self.needles is None or self.needles.shape[0] == 0:
            return

        sigma = self.get_input_noise()
        r = self.old_needle_radius_mult * sigma

        r_tensor = torch.full(
            (self.needles.shape[0], 1), r, device=self.device, dtype=self.dtype
        )

        if self.old_needles is None:
            self.old_needles = self.needles.clone()
            self.old_needle_vals = self.needle_vals.clone()
            self.old_needle_radii = r_tensor
        else:
            self.old_needles = torch.cat([self.old_needles, self.needles], dim=0)
            self.old_needle_vals = torch.cat([self.old_needle_vals, self.needle_vals], dim=0)
            self.old_needle_radii = torch.cat([self.old_needle_radii, r_tensor], dim=0)

        # Reset regular needles
        self.needles = torch.empty((0, self.d), device=self.device, dtype=self.dtype)
        self.needle_vals = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.needle_indices = torch.empty((0, 1), device=self.device, dtype=torch.int64)
        self.needle_penalty_radii = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.needle_M_list = []
        self.needles_results = []

        self._update_penalty_mask()

    def determine_new_bounds_constrained(
        self,
        converged_point: torch.Tensor,
        global_bounds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Largest axis-aligned hyperrectangle R containing *converged_point* such that:

        1. R does not intersect any regular-needle ellipsoid (or sphere fallback).
        2. No old_needle is strictly interior to R (may sit on boundary or outside).

        Algorithm (per-dimension, O(n_needles * d)):
        - Initialise lo[i] = 0, hi[i] = 1 (clamped to global_bounds if given).
        - For each regular needle, compute its axis-aligned bounding box (AABB).
          If the needle AABB overlaps the converged_point's side, tighten the bound.
        - For each old_needle that is strictly interior, tighten the bound in the
          dimension with the smallest margin to push it to the boundary.

        Parameters
        ----------
        converged_point : torch.Tensor  shape (d,)
        global_bounds   : torch.Tensor  shape (2, d) or None (defaults to [0,1]^d)
        """
        d = converged_point.shape[0]
        cp = converged_point.to(device=self.device, dtype=self.dtype)

        if global_bounds is not None:
            lo = global_bounds[0].clone().to(device=self.device, dtype=self.dtype)
            hi = global_bounds[1].clone().to(device=self.device, dtype=self.dtype)
        else:
            lo = torch.zeros(d, device=self.device, dtype=self.dtype)
            hi = torch.ones(d, device=self.device, dtype=self.dtype)

        # --- Constraint 1: exclude regular-needle ellipsoids / spheres ---
        if self.needles is not None and self.needles.shape[0] > 0:
            for idx in range(self.needles.shape[0]):
                needle = self.needles[idx]  # (d,)
                M = self.needle_M_list[idx] if idx < len(self.needle_M_list) else None

                if M is not None and self.needle_B is not None:
                    # AABB extent in each dim: sqrt((B M^{-1} B^T)[i,i])
                    M_inv = torch.linalg.inv(M)
                    amb_cov = self.needle_B @ M_inv @ self.needle_B.T  # (d, d)
                    extent = torch.sqrt(torch.clamp(torch.diag(amb_cov), min=0.0))
                else:
                    r = self.needle_penalty_radii[idx].squeeze()
                    extent = r.expand(d)

                n_lo = needle - extent   # AABB lower
                n_hi = needle + extent   # AABB upper

                for i in range(d):
                    if cp[i] < n_lo[i]:
                        # converged_point is BELOW the AABB → tighten hi[i]
                        hi[i] = torch.min(hi[i], n_lo[i])
                    elif cp[i] > n_hi[i]:
                        # converged_point is ABOVE the AABB → tighten lo[i]
                        lo[i] = torch.max(lo[i], n_hi[i])
                    # else: converged_point is inside AABB — no feasible constraint
                    # in this dimension; leave bounds as-is (AABB is approximate)

        # --- Constraint 2: old_needles must not be strictly interior ---
        if self.old_needles is not None and self.old_needles.shape[0] > 0:
            for k_idx in range(self.old_needles.shape[0]):
                on = self.old_needles[k_idx]  # (d,)
                # Check if strictly interior
                if torch.all((lo < on) & (on < hi)):
                    # Find dimension with smallest margin (cheapest to tighten)
                    margin_lo = on - lo   # how far from lower bound
                    margin_hi = hi - on   # how far from upper bound
                    # Stack: row 0 = push lower up, row 1 = push upper down
                    margins = torch.stack([margin_lo, margin_hi], dim=0)  # (2, d)
                    min_margin, dim_idx = margins.min(dim=1)  # per direction
                    best_dir = min_margin.argmin().item()  # 0 or 1
                    best_dim = dim_idx[best_dir].item()
                    if best_dir == 0:
                        lo[best_dim] = on[best_dim]  # push lower bound up to old_needle
                    else:
                        hi[best_dim] = on[best_dim]  # push upper bound down to old_needle

        # Ensure converged_point is still inside (clamp lo/hi to cp)
        lo = torch.min(lo, cp)
        hi = torch.max(hi, cp)

        return torch.stack([lo, hi], dim=0)

    # =========================================================================
    # Input noise helpers
    # =========================================================================

    def get_normalized_input_noise(self) -> float:
        """Median normalized distance between expected and actual points."""
        if self.X_all_expected is None or self.X_all_expected.shape[0] == 0:
            return 0.0
        distances = torch.norm(self.X_all_expected - self.X_all_actual, dim=1)
        normalized = distances / torch.sqrt(torch.tensor(self.d, dtype=self.dtype, device=self.device))
        return torch.median(normalized).item()

    def get_input_noise(self) -> float:
        """Median distance between expected and actual points."""
        if self.X_all_expected is None or self.X_all_expected.shape[0] == 0:
            return 0.0
        distances = torch.norm(self.X_all_expected - self.X_all_actual, dim=1)
        return torch.median(distances).item()
