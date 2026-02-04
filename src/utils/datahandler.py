"""
Data Handler for ZoMBI-Hop
==========================

Manages all data storage and retrieval for the ZoMBI-Hop optimization algorithm.
Supports both persistent (file-based) and in-memory modes.
"""

import torch
import json
import csv
import time
import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

from .dataclasses import ZoMBIHopConfig


class DataHandler:
    """
    Handles data storage, checkpointing, and retrieval for ZoMBI-Hop.

    Parameters
    ----------
    directory : str or Path, optional
        Base directory for saving checkpoints. If None, operates in memory-only mode.
    run_uuid : str, optional
        UUID for resuming a saved run. If None, generates a new one.
    max_saved_recent_checkpoints : int, optional
        Maximum number of recent checkpoints to keep. If None or 0, no saving occurs.
    device : str
        Torch device for tensors.
    dtype : torch.dtype
        Data type for tensors.
    config : ZoMBIHopConfig or dict, optional
        Configuration parameters to save. Can be a ZoMBIHopConfig dataclass or dict.
    d : int
        Dimensionality of the search space.
    top_m_points : int
        Number of top points for determining zoom bounds.
    max_gp_points : int
        Maximum points to use for GP fitting.
    """

    def __init__(
        self,
        directory: Optional[str] = None,
        run_uuid: Optional[str] = None,
        max_saved_recent_checkpoints: Optional[int] = 50,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64,
        config: Optional[Union[ZoMBIHopConfig, Dict[str, Any]]] = None,
        d: int = None,
        top_m_points: Optional[int] = None,
        max_gp_points: int = 3000,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.d = d
        self._top_m_points_input = top_m_points  # Store original input (may be None)
        self.top_m_points = top_m_points  # Will be auto-computed in save_init if None
        self.max_gp_points = max_gp_points
        
        # Convert config to dataclass if it's a dict
        if config is None:
            self.config = ZoMBIHopConfig()
        elif isinstance(config, dict):
            self.config = ZoMBIHopConfig.from_dict(config)
        else:
            self.config = config

        # Determine if we should save to disk
        self.save_enabled = (
            directory is not None and
            max_saved_recent_checkpoints is not None and
            max_saved_recent_checkpoints > 0
        )
        self.max_saved_recent_checkpoints = max_saved_recent_checkpoints or 0

        # Initialize in-memory storage
        self._init_memory_storage()

        # Checkpoint tracking
        self.checkpoint_history: List[Tuple[str, bool]] = []

        # Handle directory and UUID
        if self.save_enabled:
            self.checkpoint_base_dir = Path(directory)
            self.checkpoint_base_dir.mkdir(exist_ok=True)

            if run_uuid is not None:
                self.run_uuid = run_uuid
                self.run_dir = self.checkpoint_base_dir / f"run_{run_uuid}"
                if not self.run_dir.exists():
                    raise ValueError(f"Checkpoint directory {self.run_dir} does not exist!")
            else:
                self.run_uuid = str(uuid.uuid4())[:4]
                self.run_dir = self.checkpoint_base_dir / f"run_{self.run_uuid}"
                self.run_dir.mkdir(exist_ok=True)
        else:
            self.run_uuid = run_uuid or str(uuid.uuid4())[:4]
            self.checkpoint_base_dir = None
            self.run_dir = None

    def _init_memory_storage(self):
        """Initialize in-memory storage tensors."""
        # All points storage
        self.X_all_actual: Optional[torch.Tensor] = None
        self.X_all_expected: Optional[torch.Tensor] = None
        self.Y_all: Optional[torch.Tensor] = None

        # Initial data reference
        self.X_init_actual: Optional[torch.Tensor] = None
        self.X_init_expected: Optional[torch.Tensor] = None
        self.Y_init: Optional[torch.Tensor] = None

        # Needle tracking
        self.needles: Optional[torch.Tensor] = None
        self.needle_vals: Optional[torch.Tensor] = None
        self.needle_indices: Optional[torch.Tensor] = None
        self.needle_penalty_radii: Optional[torch.Tensor] = None
        self.needles_results: List[Dict[str, Any]] = []

        # Cached penalty mask
        self._penalty_mask: Optional[torch.Tensor] = None

        # Iteration tracking
        self.current_activation = 0
        self.current_zoom = 0
        self.current_iteration = 0
        self.no_improvements = 0

    def save_init(
        self,
        X_init_actual: torch.Tensor,
        X_init_expected: torch.Tensor,
        Y_init: torch.Tensor,
        bounds: torch.Tensor,
    ):
        """
        Save initial data and set up storage.

        Parameters
        ----------
        X_init_actual : torch.Tensor
            Initial observed locations (n, d).
        X_init_expected : torch.Tensor
            Initial expected/requested locations (n, d).
        Y_init : torch.Tensor
            Initial observed values (n, 1).
        bounds : torch.Tensor
            Search bounds (2, d).
        """
        self.d = X_init_actual.shape[1]
        self.bounds = bounds.clone().to(device=self.device, dtype=self.dtype)

        # Auto-compute top_m_points if not provided: max(d + 1, 4)
        if self.top_m_points is None:
            self.top_m_points = max(self.d + 1, 4)

        # Store initial data
        self.X_init_actual = X_init_actual.clone().to(device=self.device, dtype=self.dtype)
        self.X_init_expected = X_init_expected.clone().to(device=self.device, dtype=self.dtype)
        self.Y_init = Y_init.clone().to(device=self.device, dtype=self.dtype)

        # Initialize all points with initial data
        self.X_all_actual = self.X_init_actual.clone()
        self.X_all_expected = self.X_init_expected.clone()
        self.Y_all = self.Y_init.clone()

        # Initialize needles as empty
        self.needles = torch.empty((0, self.d), device=self.device, dtype=self.dtype)
        self.needle_vals = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.needle_indices = torch.empty((0, 1), device=self.device, dtype=torch.int64)
        self.needle_penalty_radii = torch.empty((0, 1), device=self.device, dtype=self.dtype)

        # Initialize penalty mask
        self._update_penalty_mask()

        # Save config if enabled
        if self.save_enabled:
            self._save_config()
            self.push_checkpoint("init", is_permanent=True)

    def _save_config(self):
        """Save configuration to disk."""
        if not self.save_enabled:
            return

        # Convert config dataclass to dict and add metadata
        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
        config_dict.update({
            'run_uuid': self.run_uuid,
            'd': self.d,
            'top_m_points': self.top_m_points,
            'max_gp_points': self.max_gp_points,
        })

        config_path = self.run_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def load_state(self) -> Tuple[int, int, int, int]:
        """
        Load state from disk (if saving enabled and UUID provided).

        Returns
        -------
        tuple
            (current_activation, current_zoom, current_iteration, no_improvements)
        """
        if not self.save_enabled or not self.run_dir.exists():
            return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

        # Load config
        config_path = self.run_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Extract metadata fields
            self.d = config_dict.pop('d', self.d)
            self.top_m_points = config_dict.pop('top_m_points', self.top_m_points)
            self.max_gp_points = config_dict.pop('max_gp_points', self.max_gp_points)
            run_uuid = config_dict.pop('run_uuid', None)
            
            # Reconstruct config dataclass from remaining fields
            self.config = ZoMBIHopConfig.from_dict(config_dict)

        # Find current state
        current_state_file = self.run_dir / 'current_state.txt'
        if not current_state_file.exists():
            return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

        with open(current_state_file, 'r') as f:
            iteration_label = f.read().strip()

        state_dir = self.run_dir / 'states' / iteration_label
        if not state_dir.exists():
            return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

        # Load tensors
        tensors = torch.load(state_dir / 'tensors.pt', map_location=self.device)
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

        # Load needles results
        needles_results_path = state_dir / 'needles_results.json'
        if needles_results_path.exists():
            with open(needles_results_path, 'r') as f:
                needles_results_loaded = json.load(f)

            self.needles_results = []
            for result_dict in needles_results_loaded:
                self.needles_results.append({
                    'point': torch.tensor(result_dict['point'], device=self.device, dtype=self.dtype),
                    'value': result_dict['value'],
                    'activation': result_dict['activation'],
                    'zoom': result_dict['zoom'],
                    'iteration': result_dict['iteration'],
                })

        # Load tracking
        tracking_path = state_dir / 'tracking.json'
        if tracking_path.exists():
            with open(tracking_path, 'r') as f:
                tracking = json.load(f)

            self.current_activation = tracking['current_activation']
            self.current_zoom = tracking['current_zoom']
            self.current_iteration = tracking['current_iteration']
            self.no_improvements = tracking['no_improvements']

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

    def push_checkpoint(
        self,
        label: str,
        is_permanent: bool = False,
    ):
        """
        Save a checkpoint (if saving is enabled).

        Parameters
        ----------
        label : str
            Label for the checkpoint.
        is_permanent : bool
            If True, checkpoint won't be deleted during cleanup.
        """
        if not self.save_enabled:
            return

        state_dir = self.run_dir / 'states' / label
        state_dir.mkdir(parents=True, exist_ok=True)

        # Calculate distances
        distances = torch.norm(self.X_all_expected - self.X_all_actual, dim=1)

        # Save tensors
        torch.save({
            'bounds': self.bounds,
            'X_init_actual': self.X_init_actual,
            'X_init_expected': self.X_init_expected,
            'Y_init': self.Y_init,
            'X_all_actual': self.X_all_actual,
            'X_all_expected': self.X_all_expected,
            'Y_all': self.Y_all,
            'distances': distances,
            'needles': self.needles,
            'needle_vals': self.needle_vals,
            'needle_indices': self.needle_indices,
            'needle_penalty_radii': self.needle_penalty_radii,
            'penalty_mask': self._penalty_mask,
        }, state_dir / 'tensors.pt')

        # Save needles results
        needles_results_serializable = []
        for needle_result in self.needles_results:
            result_dict = {
                'point': needle_result['point'].cpu().tolist(),
                'value': needle_result['value'],
                'activation': needle_result['activation'],
                'zoom': needle_result['zoom'],
                'iteration': needle_result['iteration'],
            }
            needles_results_serializable.append(result_dict)

        with open(state_dir / 'needles_results.json', 'w') as f:
            json.dump(needles_results_serializable, f, indent=2)

        # Save tracking
        tracking = {
            'current_activation': self.current_activation,
            'current_zoom': self.current_zoom,
            'current_iteration': self.current_iteration,
            'no_improvements': self.no_improvements,
        }
        with open(state_dir / 'tracking.json', 'w') as f:
            json.dump(tracking, f, indent=2)

        # Update current state link
        current_state_file = self.run_dir / 'current_state.txt'
        with open(current_state_file, 'w') as f:
            f.write(label)

        # Save statistics
        stats = {
            'iteration_label': label,
            'timestamp': time.time(),
            'num_points_total': self.X_all_actual.shape[0],
            'num_needles': self.needles.shape[0],
            'best_value': self.Y_all.max().item() if self.Y_all.numel() > 0 else None,
            'best_value_unpenalized': (
                self.Y_all[self._penalty_mask].max().item()
                if (self.Y_all.numel() > 0 and self._penalty_mask.any()) else None
            ),
            'input_noise': self.get_normalized_input_noise(),
            'mean_distance': distances.mean().item() if distances.numel() > 0 else 0.0,
            'median_distance': distances.median().item() if distances.numel() > 0 else 0.0,
            'max_distance': distances.max().item() if distances.numel() > 0 else 0.0,
        }
        with open(state_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        # Save CSV
        if self.X_all_actual.shape[0] > 0:
            with open(state_dir / 'all_points.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['index', 'y_value']
                header.extend([f'x_actual_{i}' for i in range(self.d)])
                header.extend([f'x_expected_{i}' for i in range(self.d)])
                header.append('distance')
                header.append('penalized')
                writer.writerow(header)

                for idx in range(self.X_all_actual.shape[0]):
                    row = [idx, self.Y_all[idx].item()]
                    row.extend(self.X_all_actual[idx].cpu().tolist())
                    row.extend(self.X_all_expected[idx].cpu().tolist())
                    row.append(distances[idx].item())
                    row.append(not self._penalty_mask[idx].item())
                    writer.writerow(row)

        # Track and cleanup
        self.checkpoint_history.append((label, is_permanent))
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping recent + permanent ones."""
        if not self.save_enabled:
            return
        if len(self.checkpoint_history) <= self.max_saved_recent_checkpoints:
            return

        keep_labels = set()

        # Keep permanent checkpoints
        for label, is_permanent in self.checkpoint_history:
            if is_permanent:
                keep_labels.add(label)

        # Keep recent checkpoints
        recent_labels = [label for label, _ in self.checkpoint_history[-self.max_saved_recent_checkpoints:]]
        keep_labels.update(recent_labels)

        # Delete old non-permanent checkpoints
        states_dir = self.run_dir / 'states'
        deleted_count = 0

        for label, _ in self.checkpoint_history:
            if label not in keep_labels:
                state_dir = states_dir / label
                if state_dir.exists():
                    try:
                        shutil.rmtree(state_dir)
                        deleted_count += 1
                    except Exception as e:
                        pass  # Silently continue

        # Update history
        self.checkpoint_history = [(label, perm) for label, perm in self.checkpoint_history
                                   if label in keep_labels]

    def add_all_points(
        self,
        new_X_actual: torch.Tensor,
        new_X_expected: torch.Tensor,
        new_Y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add new points to the dataset.

        Parameters
        ----------
        new_X_actual : torch.Tensor
            Actual observed locations.
        new_X_expected : torch.Tensor
            Expected/requested locations.
        new_Y : torch.Tensor
            Observed values.

        Returns
        -------
        torch.Tensor
            Penalty mask for the new points.
        """
        new_X_actual = new_X_actual.to(device=self.device, dtype=self.dtype)
        new_X_expected = new_X_expected.to(device=self.device, dtype=self.dtype)
        new_Y = new_Y.to(device=self.device, dtype=self.dtype)

        # Ensure proper shapes
        if new_Y.ndim == 1:
            new_Y = new_Y.unsqueeze(1)

        # Calculate penalty mask for new points
        new_penalty_mask = self._compute_penalty_mask(new_X_actual)

        # Concatenate
        self.X_all_actual = torch.cat([self.X_all_actual, new_X_actual], dim=0)
        self.X_all_expected = torch.cat([self.X_all_expected, new_X_expected], dim=0)
        self.Y_all = torch.cat([self.Y_all, new_Y], dim=0)
        self._penalty_mask = torch.cat([self._penalty_mask, new_penalty_mask], dim=0)

        return new_penalty_mask

    def get_all_points(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all collected points.

        Returns
        -------
        tuple
            (X_all_actual, X_all_expected, Y_all)
        """
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
        """
        Add a discovered needle (local optimum).

        Parameters
        ----------
        needle : torch.Tensor
            Location of the needle (d,).
        needle_value : float
            Function value at the needle.
        needle_penalty_radius : float
            Penalty radius for this needle.
        activation : int
            Current activation number.
        zoom : int
            Current zoom level.
        iteration : int
            Current iteration.
        """
        needle = needle.to(device=self.device, dtype=self.dtype)

        # Find the global index
        unpenalized_mask = self._penalty_mask
        distances = torch.norm(self.X_all_actual - needle.unsqueeze(0), dim=1)
        global_idx = distances.argmin()

        # Add to storage
        self.needles = torch.cat([self.needles, needle.unsqueeze(0)], dim=0)
        self.needle_vals = torch.cat([
            self.needle_vals,
            torch.tensor([[needle_value]], device=self.device, dtype=self.dtype)
        ], dim=0)
        self.needle_indices = torch.cat([
            self.needle_indices,
            global_idx.reshape(1, 1)
        ], dim=0)
        self.needle_penalty_radii = torch.cat([
            self.needle_penalty_radii,
            torch.tensor([[needle_penalty_radius]], device=self.device, dtype=self.dtype)
        ], dim=0)

        # Add to results list
        self.needles_results.append({
            'point': needle.clone(),
            'value': needle_value,
            'activation': activation,
            'zoom': zoom,
            'iteration': iteration,
        })

        # Update penalty mask
        self._update_penalty_mask()

    def get_needle_locations(self) -> torch.Tensor:
        """
        Get locations of all needles.

        Returns
        -------
        torch.Tensor
            (num_needles, d) tensor of needle locations.
        """
        return self.needles

    def get_needle_results(self) -> List[Dict[str, Any]]:
        """
        Get detailed results for all needles.

        Returns
        -------
        list
            List of dicts with needle info.
        """
        return self.needles_results

    def get_needles_and_penalty_radii(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get needle locations and their penalty radii.

        Returns
        -------
        tuple
            (needles, penalty_radii) tensors.
        """
        return self.needles, self.needle_penalty_radii

    def _compute_penalty_mask(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty mask for given points.

        Parameters
        ----------
        X : torch.Tensor
            Points to check (n, d) or (n, l, d).

        Returns
        -------
        torch.Tensor
            Boolean mask where True = not penalized.
        """
        if X.ndim == 2:
            X_reshaped = X.unsqueeze(1)
            n = X.shape[0]
            l = 1
        elif X.ndim == 3:
            X_reshaped = X
            n, l, d = X.shape
        else:
            raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")

        if self.needles is None or self.needles.shape[0] == 0:
            if X.ndim == 2:
                return torch.ones(n, dtype=torch.bool, device=X.device)
            else:
                return torch.ones((n, l), dtype=torch.bool, device=X.device)

        # Compute distances to all needles
        X_expanded = X_reshaped.unsqueeze(2)  # (n, l, 1, d)
        needles_expanded = self.needles.unsqueeze(0).unsqueeze(0)  # (1, 1, num_needles, d)
        penalty_radii_expanded = self.needle_penalty_radii.view(1, 1, -1)  # (1, 1, num_needles)

        distances = torch.norm(X_expanded - needles_expanded, dim=-1)  # (n, l, num_needles)
        within_radius = distances <= penalty_radii_expanded
        penalized = within_radius.any(dim=2)  # (n, l)

        if X.ndim == 2:
            penalized = penalized.squeeze(1)

        return ~penalized

    def get_penalty_mask(self, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get penalty mask for points.

        Parameters
        ----------
        X : torch.Tensor, optional
            Points to check. If None, returns mask for all stored points.

        Returns
        -------
        torch.Tensor
            Boolean mask where True = not penalized.
        """
        if X is None:
            return self._penalty_mask
        return self._compute_penalty_mask(X)

    def _update_penalty_mask(self):
        """Update the cached penalty mask for all points."""
        if self.X_all_actual is not None and self.X_all_actual.shape[0] > 0:
            self._penalty_mask = self._compute_penalty_mask(self.X_all_actual)
        else:
            self._penalty_mask = torch.ones(0, dtype=torch.bool, device=self.device)

    def get_normalized_input_noise(self) -> float:
        """
        Calculate normalized input noise level.

        Returns
        -------
        float
            Median normalized distance between expected and actual points.
        """
        if self.X_all_expected is None or self.X_all_expected.shape[0] == 0:
            return 0.0
        distances = torch.norm(self.X_all_expected - self.X_all_actual, dim=1)
        normalized_distances = distances / torch.sqrt(torch.tensor(self.d, dtype=self.dtype, device=self.device))
        return torch.median(normalized_distances).item()

    def get_input_noise(self) -> float:
        """
        Calculate raw input noise level.

        Returns
        -------
        float
            Median distance between expected and actual points.
        """
        if self.X_all_expected is None or self.X_all_expected.shape[0] == 0:
            return 0.0
        distances = torch.norm(self.X_all_expected - self.X_all_actual, dim=1)
        return torch.median(distances).item()

    def determine_new_bounds(self) -> torch.Tensor:
        """
        Determine new bounds based on top performing unpenalized points.

        Returns
        -------
        torch.Tensor
            New bounds (2, d) with [min_bounds, max_bounds].
        """
        Y_masked = self.Y_all[self._penalty_mask].squeeze(-1)
        k = min(self.top_m_points, Y_masked.numel())
        top_idx = torch.topk(Y_masked, k).indices
        X_top = self.X_all_actual[self._penalty_mask][top_idx]

        min_bounds = X_top.min(dim=0).values
        max_bounds = X_top.max(dim=0).values
        return torch.stack([min_bounds, max_bounds], dim=0)

    def get_gp_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get data for GP fitting (top unpenalized points).

        Returns
        -------
        tuple
            (X_gp, Y_gp) tensors of top unpenalized points.
        """
        X_non_penalized = self.X_all_actual[self._penalty_mask]
        Y_non_penalized = self.Y_all[self._penalty_mask]

        sorted_indices = torch.argsort(Y_non_penalized.squeeze(), descending=True)
        n_points = min(self.max_gp_points, len(sorted_indices))
        top_indices = sorted_indices[:n_points]

        return X_non_penalized[top_indices], Y_non_penalized[top_indices]

    def get_best_unpenalized(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get the best unpenalized point.

        Returns
        -------
        tuple
            (X_best, Y_best, global_index)
        """
        if not self._penalty_mask.any():
            return None, None, None

        Y_masked = self.Y_all[self._penalty_mask]
        max_idx = Y_masked.argmax()

        unpenalized_indices = torch.where(self._penalty_mask)[0]
        global_idx = unpenalized_indices[max_idx]

        return self.X_all_actual[global_idx], self.Y_all[global_idx], global_idx.item()

    def update_iteration_state(
        self,
        activation: int,
        zoom: int,
        iteration: int,
        no_improvements: int,
    ):
        """Update iteration tracking state."""
        self.current_activation = activation
        self.current_zoom = zoom
        self.current_iteration = iteration
        self.no_improvements = no_improvements

    def get_iteration_state(self) -> Tuple[int, int, int, int]:
        """Get current iteration state."""
        return (
            self.current_activation,
            self.current_zoom,
            self.current_iteration,
            self.no_improvements,
        )
