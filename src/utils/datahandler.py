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
    penalization_threshold, penalty_num_directions, penalty_max_radius,
    penalty_radius_step, convergence_pi_threshold, input_noise_threshold_mult,
    output_noise_threshold_mult, n_consecutive_converged, max_gp_points,
    repulsion_lambda, acquisition_type, ucb_beta : control variables
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
        penalization_threshold: float = 1e-3,
        penalty_num_directions: Optional[int] = None,
        penalty_max_radius: float = 0.3,
        penalty_radius_step: Optional[float] = None,
        convergence_pi_threshold: float = 0.01,
        input_noise_threshold_mult: float = 2.0,
        output_noise_threshold_mult: float = 2.0,
        n_consecutive_converged: int = 2,
        max_gp_points: int = 3000,
        repulsion_lambda: Optional[float] = None,
        acquisition_type: str = "ucb",
        ucb_beta: float = 0.1,
        # --- Storage settings ---
        directory: Optional[str] = None,
        run_uuid: Optional[str] = None,
        max_saved_recent_checkpoints: Optional[int] = 50,  # backward compat
        max_snapshots: Optional[int] = None,
        # --- Compute settings ---
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64,
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
            penalization_threshold = cfg.penalization_threshold
            if cfg.penalty_num_directions is not None:
                penalty_num_directions = cfg.penalty_num_directions
            penalty_max_radius = cfg.penalty_max_radius
            penalty_radius_step = cfg.penalty_radius_step
            convergence_pi_threshold = cfg.convergence_pi_threshold
            input_noise_threshold_mult = cfg.input_noise_threshold_mult
            output_noise_threshold_mult = cfg.output_noise_threshold_mult
            n_consecutive_converged = cfg.n_consecutive_converged
            max_gp_points = cfg.max_gp_points
            repulsion_lambda = cfg.repulsion_lambda
            acquisition_type = getattr(cfg, 'acquisition_type', acquisition_type)
            ucb_beta = getattr(cfg, 'ucb_beta', ucb_beta)

        # --- Store all control variables as plain attributes ---
        self.max_zooms = max_zooms
        self.max_iterations = max_iterations
        self.top_m_points = top_m_points
        self.n_restarts = n_restarts
        self.raw = raw
        self.penalization_threshold = penalization_threshold
        self.penalty_num_directions = penalty_num_directions
        self.penalty_max_radius = penalty_max_radius
        self.penalty_radius_step = penalty_radius_step
        self.convergence_pi_threshold = convergence_pi_threshold
        self.input_noise_threshold_mult = input_noise_threshold_mult
        self.output_noise_threshold_mult = output_noise_threshold_mult
        self.n_consecutive_converged = n_consecutive_converged
        self.max_gp_points = max_gp_points
        self.repulsion_lambda = repulsion_lambda
        self.acquisition_type = acquisition_type
        self.ucb_beta = ucb_beta

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

        self.needles: Optional[torch.Tensor] = None
        self.needle_vals: Optional[torch.Tensor] = None
        self.needle_indices: Optional[torch.Tensor] = None
        self.needle_penalty_radii: Optional[torch.Tensor] = None
        self.needles_results: List[Dict[str, Any]] = []

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
        Auto-computes top_m_points and penalty_num_directions if not already set.
        """
        self.d = X_init_actual.shape[1]
        self.bounds = bounds.clone().to(device=self.device, dtype=self.dtype)

        if self.top_m_points is None:
            self.top_m_points = max(self.d + 1, 4)
        if self.penalty_num_directions is None:
            self.penalty_num_directions = 10 * self.d

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
            'penalization_threshold': self.penalization_threshold,
            'penalty_num_directions': self.penalty_num_directions,
            'penalty_max_radius': self.penalty_max_radius,
            'penalty_radius_step': self.penalty_radius_step,
            'convergence_pi_threshold': self.convergence_pi_threshold,
            'input_noise_threshold_mult': self.input_noise_threshold_mult,
            'output_noise_threshold_mult': self.output_noise_threshold_mult,
            'n_consecutive_converged': self.n_consecutive_converged,
            'max_gp_points': self.max_gp_points,
            'repulsion_lambda': self.repulsion_lambda,
            'acquisition_type': self.acquisition_type,
            'ucb_beta': self.ucb_beta,
            'device': str(self.device),
            'dtype': str(self.dtype),
        }
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # =========================================================================
    # Snapshotting (new, simple system)
    # =========================================================================

    def take_snapshot(self, label: str = "", permanent: bool = False):
        """
        Save complete state to disk.

        Saves all tensors, needle results, iteration state, and a summary.
        Snapshots are numbered sequentially under run_dir/snapshots/.
        Removes oldest non-permanent snapshots if max_snapshots is set.
        Permanent snapshots are never cleaned up.
        """
        if not self.save_enabled:
            return

        self._snapshot_count += 1
        name = f"{self._snapshot_count:04d}_{label}" if label else f"{self._snapshot_count:04d}"
        snapshot_dir = self.run_dir / "snapshots" / name
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        if permanent:
            (snapshot_dir / 'permanent').touch()

        # All tensors in one file
        torch.save({
            'bounds': self.bounds,
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
            self.penalization_threshold = cfg.get('penalization_threshold', self.penalization_threshold)
            self.penalty_num_directions = cfg.get('penalty_num_directions', self.penalty_num_directions)
            self.penalty_max_radius = cfg.get('penalty_max_radius', self.penalty_max_radius)
            self.penalty_radius_step = cfg.get('penalty_radius_step', self.penalty_radius_step)
            self.convergence_pi_threshold = cfg.get('convergence_pi_threshold', self.convergence_pi_threshold)
            self.input_noise_threshold_mult = cfg.get('input_noise_threshold_mult', self.input_noise_threshold_mult)
            self.output_noise_threshold_mult = cfg.get('output_noise_threshold_mult', self.output_noise_threshold_mult)
            self.n_consecutive_converged = cfg.get('n_consecutive_converged', self.n_consecutive_converged)
            self.max_gp_points = cfg.get('max_gp_points', self.max_gp_points)
            self.repulsion_lambda = cfg.get('repulsion_lambda', self.repulsion_lambda)
            self.acquisition_type = cfg.get('acquisition_type', self.acquisition_type)
            self.ucb_beta = cfg.get('ucb_beta', self.ucb_beta)

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
    ):
        """Record a discovered needle (local optimum) and update penalty mask."""
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

    # =========================================================================
    # Penalty mask
    # =========================================================================

    def get_penalty_mask(self, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return penalty mask (True = not penalized). If X is None, returns mask for all stored points."""
        if X is None:
            return self._penalty_mask
        return self._compute_penalty_mask(X)

    def _compute_penalty_mask(self, X: torch.Tensor) -> torch.Tensor:
        """Compute penalty mask for given points. True = not inside any penalty ball."""
        if X.ndim == 2:
            X_reshaped = X.unsqueeze(1)
            n = X.shape[0]
            l = 1
        elif X.ndim == 3:
            X_reshaped = X
            n, l, _ = X.shape
        else:
            raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")

        if self.needles is None or self.needles.shape[0] == 0:
            if X.ndim == 2:
                return torch.ones(n, dtype=torch.bool, device=X.device)
            else:
                return torch.ones((n, l), dtype=torch.bool, device=X.device)

        X_expanded = X_reshaped.unsqueeze(2)                       # (n, l, 1, d)
        needles_expanded = self.needles.unsqueeze(0).unsqueeze(0)  # (1, 1, M, d)
        radii_expanded = self.needle_penalty_radii.view(1, 1, -1)  # (1, 1, M)

        distances = torch.norm(X_expanded - needles_expanded, dim=-1)  # (n, l, M)
        penalized = (distances <= radii_expanded).any(dim=2)            # (n, l)

        if X.ndim == 2:
            penalized = penalized.squeeze(1)

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
        """Return (X, Y) of the top max_gp_points unpenalized points for GP fitting."""
        X = self.X_all_actual[self._penalty_mask]
        Y = self.Y_all[self._penalty_mask]
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
