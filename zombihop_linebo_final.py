import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.autograd import grad
from typing import Callable, Tuple, Optional, List, Union

from scipy.optimize import differential_evolution

import json
import os
import uuid
import time
import shutil
import csv
from pathlib import Path

# CUDA optimization settings
if torch.cuda.is_available():
    # Set memory fraction to avoid memory issues
    torch.cuda.set_per_process_memory_fraction(0.95)
    # Enable cuDNN benchmarking for optimal performance
    torch.backends.cudnn.benchmark = True
    # Enable cuDNN deterministic mode for reproducibility
    torch.backends.cudnn.deterministic = False
    # Set default tensor type to CUDA
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class ZoMBIHop:
    def __init__(self,
                 objective,
                 bounds: torch.Tensor,
                 X_init_actual: torch.Tensor,
                 X_init_expected: torch.Tensor,
                 Y_init: torch.Tensor,
                 proj_fn = None,
                 random_sampler = None,
                 random_direction_sampler = None,
                 max_zooms: int = 3,
                 max_iterations: int = 10,
                 top_m_points: int = 4,
                 n_restarts: int = 30,
                 raw: int = 500,
                 penalization_threshold: float = 1e-3,
                 penalty_num_directions: int = 100,
                 penalty_max_radius: float = 0.3,
                 penalty_radius_step: float = 0.01,
                 improvement_threshold_noise_mult: float = 2.0,
                 input_noise_threshold_mult: float = 3.0,
                 n_consecutive_no_improvements: int = 5,
                 max_gp_points: int = 3000,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float64,
                 run_uuid: str = None,
                 checkpoint_dir: str = 'zombihop_checkpoints'):
        """
        Initialize ZoMBIHop optimizer.
        
        Args:
            run_uuid: 4-digit UUID to resume from a saved run. If None, creates a new run.
            checkpoint_dir: Base directory for storing checkpoints
        """
        # computer parameters
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_gp_points = max_gp_points

        # CUDA optimization
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()  # Clear GPU memory before starting
            print(f"Initialized ZoMBIHop on CUDA device: {torch.cuda.get_device_name()}")
            print(f"Initial CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        self.objective = objective
        self.proj_fn = proj_fn if proj_fn is not None else ZoMBIHop.proj_simplex
        self.random_sampler = random_sampler if random_sampler is not None else ZoMBIHop.random_simplex
        self.random_direction_sampler = random_direction_sampler if random_direction_sampler is not None else ZoMBIHop.random_zero_sum_directions

        # Checkpoint management
        self.checkpoint_base_dir = Path(checkpoint_dir)
        self.checkpoint_base_dir.mkdir(exist_ok=True)
        
        # Rolling checkpoint configuration
        self.max_recent_iterations = 50  # Keep last 50 iterations
        self.checkpoint_history = []  # Track all checkpoints: (label, is_permanent)
        
        # Check if resuming from saved run
        if run_uuid is not None:
            print(f"Resuming from saved run: {run_uuid}")
            self.run_uuid = run_uuid
            self.run_dir = self.checkpoint_base_dir / f"run_{run_uuid}"
            if not self.run_dir.exists():
                raise ValueError(f"Checkpoint directory {self.run_dir} does not exist!")
            self._load_state()
            print(f"Successfully loaded state from {self.run_dir}")
        else:
            # Generate new UUID for this run
            self.run_uuid = str(uuid.uuid4())[:4]
            self.run_dir = self.checkpoint_base_dir / f"run_{self.run_uuid}"
            self.run_dir.mkdir(exist_ok=True)
            print(f"Starting new run with UUID: {self.run_uuid}")
            print(f"Checkpoint directory: {self.run_dir}")

            # bounds parameters
            self.d = bounds.shape[1]
            self.bounds = bounds.clone().to(device=self.device, dtype=self.dtype) # (2, d) torch tensor
            assert self.bounds.shape == (2, self.d), "bounds must be a (2, d) torch tensor"

            self.X_init_actual = X_init_actual.clone().to(device=self.device, dtype=self.dtype) # (n, d) torch tensor
            assert self.X_init_actual.shape[1] == self.d, "X_init_actual must be a (n, d) torch tensor"
            self.X_init_expected = X_init_expected.clone().to(device=self.device, dtype=self.dtype) # (n, d) torch tensor
            assert self.X_init_expected.shape[1] == self.d, "X_init_expected must be a (n, d) torch tensor"
            self.Y_init = Y_init.clone().to(device=self.device, dtype=self.dtype) # (n, 1) torch tensor
            assert self.Y_init.shape[1] == 1, "Y_init must be a (n, 1) torch tensor"
            assert self.X_init_actual.shape[0] == self.X_init_expected.shape[0] == self.Y_init.shape[0], "X_init_actual, X_init_expected, and Y_init must have the same number of rows"

            # all points found
            self.X_all_actual = X_init_actual.clone().to(device=self.device, dtype=self.dtype) # (n, d) torch tensor
            self.X_all_expected = X_init_expected.clone().to(device=self.device, dtype=self.dtype) # for calculating input noise. (n, d) torch tensor
            self.Y_all = Y_init.clone().to(device=self.device, dtype=self.dtype) # (n, 1) torch tensor

            # needle tracking parameters
            self.needles_results = []  # List of dicts with 'point' as tensor, 'value' as scalar, and metadata
            self.needles = torch.empty((0, self.d), device=self.device, dtype=self.dtype)  # (0, d) tensor
            self.needle_vals = torch.empty((0, 1), device=self.device, dtype=self.dtype)  # (0, 1) tensor
            self.needle_indices = torch.empty((0, 1), device=self.device, dtype=torch.int64)  # (0, 1) tensor
            self.needle_penalty_radii = torch.empty((0, 1), device=self.device, dtype=self.dtype)  # (0, 1) tensor
            self._set_penalty_mask()

            # zombihop parameters
            self.max_zooms = max_zooms
            self.max_iterations = max_iterations
            self.top_m_points = top_m_points # for determining zoom

            # finding next point parameters
            self.n_restarts = n_restarts
            self.raw = raw

            # penalization parameters
            self.penalization_threshold = penalization_threshold
            self.penalty_num_directions = penalty_num_directions
            self.penalty_max_radius = penalty_max_radius
            self.penalty_radius_step = penalty_radius_step

            # convergence parameters
            self.improvement_threshold_noise_mult = improvement_threshold_noise_mult
            self.input_noise_threshold_mult = input_noise_threshold_mult
            self.n_consecutive_no_improvements = n_consecutive_no_improvements
            
            # Iteration tracking
            self.current_activation = 0
            self.current_zoom = 0
            self.current_iteration = 0
            self.no_improvements = 0
            
            # Save initial state (permanent)
            self._save_config()
            self._save_state(iteration_label="init", is_permanent=True)

    def _save_config(self):
        """Save the configuration parameters of the run."""
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
            'improvement_threshold_noise_mult': self.improvement_threshold_noise_mult,
            'input_noise_threshold_mult': self.input_noise_threshold_mult,
            'n_consecutive_no_improvements': self.n_consecutive_no_improvements,
            'max_gp_points': self.max_gp_points,
            'device': str(self.device),
            'dtype': str(self.dtype),
        }
        
        config_path = self.run_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")

    def _save_state(self, iteration_label: str = "init", is_permanent: bool = False):
        """
        Save the current state of the optimizer.
        
        Args:
            iteration_label: Label for this state
            is_permanent: If True, this checkpoint will never be deleted (zoom ends, needle discoveries)
        """
        state_dir = self.run_dir / 'states' / iteration_label
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate distances between expected and actual
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
            'penalty_mask': self.penalty_mask,
        }, state_dir / 'tensors.pt')
        
        # Save needles_results (convert tensors to lists for JSON serialization)
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
        
        # Save iteration tracking
        tracking = {
            'current_activation': self.current_activation,
            'current_zoom': self.current_zoom,
            'current_iteration': self.current_iteration,
            'no_improvements': self.no_improvements,
        }
        with open(state_dir / 'tracking.json', 'w') as f:
            json.dump(tracking, f, indent=2)
        
        # Save current state link (always point to the latest state)
        current_state_file = self.run_dir / 'current_state.txt'
        with open(current_state_file, 'w') as f:
            f.write(iteration_label)
        
        # Calculate and save statistics
        stats = {
            'iteration_label': iteration_label,
            'timestamp': time.time(),
            'num_points_total': self.X_all_actual.shape[0],
            'num_needles': self.needles.shape[0],
            'best_value': self.Y_all.max().item() if self.Y_all.numel() > 0 else None,
            # Track best among CURRENTLY unpenalized points as well
            'best_value_unpenalized': (
                self.Y_all[self.penalty_mask].max().item()
                if (self.Y_all.numel() > 0 and self.penalty_mask.any()) else None
            ),
            'input_noise': self._calculate_normalized_input_noise(),
            'mean_distance': distances.mean().item() if distances.numel() > 0 else 0.0,
            'median_distance': distances.median().item() if distances.numel() > 0 else 0.0,
            'max_distance': distances.max().item() if distances.numel() > 0 else 0.0,
        }
        with open(state_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save human-readable CSV of all points and distances
        if self.X_all_actual.shape[0] > 0:
            with open(state_dir / 'all_points.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                header = ['index', 'y_value']
                header.extend([f'x_actual_{i}' for i in range(self.d)])
                header.extend([f'x_expected_{i}' for i in range(self.d)])
                header.append('distance')
                header.append('penalized')
                writer.writerow(header)
                
                # Data rows
                for idx in range(self.X_all_actual.shape[0]):
                    row = [idx, self.Y_all[idx].item()]
                    row.extend(self.X_all_actual[idx].cpu().tolist())
                    row.extend(self.X_all_expected[idx].cpu().tolist())
                    row.append(distances[idx].item())
                    row.append(not self.penalty_mask[idx].item())
                    writer.writerow(row)
        
        # Track this checkpoint
        self.checkpoint_history.append((iteration_label, is_permanent))
        
        # Clean up old checkpoints (keep last 50 + permanent ones)
        self._cleanup_old_checkpoints()
        
        print(f"Saved state to {state_dir}" + (" [PERMANENT]" if is_permanent else ""))

    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints, keeping:
        1. The last 50 iterations (rolling window)
        2. All permanent checkpoints (zoom ends, needle discoveries, etc.)
        """
        if len(self.checkpoint_history) <= self.max_recent_iterations:
            # Not enough checkpoints to clean up yet
            return
        
        # Identify checkpoints to keep
        keep_labels = set()
        
        # Keep all permanent checkpoints
        for label, is_permanent in self.checkpoint_history:
            if is_permanent:
                keep_labels.add(label)
        
        # Keep the last N iterations
        recent_labels = [label for label, _ in self.checkpoint_history[-self.max_recent_iterations:]]
        keep_labels.update(recent_labels)
        
        # Delete checkpoints not in keep_labels
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
                        print(f"Warning: Could not delete {state_dir}: {e}")
        
        # Update checkpoint history to only include kept checkpoints
        self.checkpoint_history = [(label, perm) for label, perm in self.checkpoint_history 
                                   if label in keep_labels]
        
        if deleted_count > 0:
            print(f"  Cleaned up {deleted_count} old checkpoints (keeping last {self.max_recent_iterations} + permanents)")

    def _load_state(self):
        """Load state from the checkpoint directory."""
        # Load config
        config_path = self.run_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Restore config parameters
        self.d = config['d']
        self.max_zooms = config['max_zooms']
        self.max_iterations = config['max_iterations']
        self.top_m_points = config['top_m_points']
        self.n_restarts = config['n_restarts']
        self.raw = config['raw']
        self.penalization_threshold = config['penalization_threshold']
        self.penalty_num_directions = config['penalty_num_directions']
        self.penalty_max_radius = config['penalty_max_radius']
        self.penalty_radius_step = config['penalty_radius_step']
        self.improvement_threshold_noise_mult = config['improvement_threshold_noise_mult']
        self.input_noise_threshold_mult = config['input_noise_threshold_mult']
        self.n_consecutive_no_improvements = config['n_consecutive_no_improvements']
        self.max_gp_points = config['max_gp_points']
        
        # Find the latest state
        current_state_file = self.run_dir / 'current_state.txt'
        with open(current_state_file, 'r') as f:
            iteration_label = f.read().strip()
        
        state_dir = self.run_dir / 'states' / iteration_label
        print(f"Loading state from {state_dir}")
        
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
        self.penalty_mask = tensors['penalty_mask'].to(device=self.device)
        
        # Load needles_results
        with open(state_dir / 'needles_results.json', 'r') as f:
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
        
        # Load iteration tracking
        with open(state_dir / 'tracking.json', 'r') as f:
            tracking = json.load(f)
        
        self.current_activation = tracking['current_activation']
        self.current_zoom = tracking['current_zoom']
        self.current_iteration = tracking['current_iteration']
        self.no_improvements = tracking['no_improvements']
        
        # Reconstruct checkpoint history from existing states
        states_dir = self.run_dir / 'states'
        self.checkpoint_history = []
        if states_dir.exists():
            for state_dir in sorted(states_dir.iterdir()):
                if state_dir.is_dir():
                    label = state_dir.name
                    # Mark as permanent if it's a special checkpoint
                    is_permanent = any(keyword in label for keyword in 
                                      ['init', 'needle', 'complete', 'finished', 'timeout', 'final'])
                    self.checkpoint_history.append((label, is_permanent))
        
        print(f"Loaded state: activation={self.current_activation}, zoom={self.current_zoom}, iteration={self.current_iteration}")
        print(f"Checkpoint history: {len(self.checkpoint_history)} states found")

    def _determine_penalty_radius(
        self,
        needle: torch.Tensor,
        acq: torch.nn.Module
    ) -> float:
        # build random unit directions
        dirs = self._random_direction_sampler_wrapper(self.penalty_num_directions)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)

        r = self.penalty_radius_step
        while r <= self.penalty_max_radius:
            X_shell = needle.unsqueeze(0) + r * dirs  # (num_directions, d)
            # find worst‐case gradient on this shell
            max_grad = 0.0
            for x in X_shell:
                # compute gradient magnitude
                x = x.unsqueeze(0).clone().detach().requires_grad_(True)
                y = acq(x)
                grads = grad(y.sum(), x)[0]
                gm = grads.norm().item()
                # update max_grad if this gradient is greater than the current max_grad
                if gm > max_grad:
                    max_grad = gm
                    if max_grad > self.penalization_threshold:
                        break
            if max_grad <= self.penalization_threshold:
                return max(r, float(3*self._calculate_input_noise())) # want it to be a minimum of 3*input noise
            r += self.penalty_radius_step

        return self.penalty_max_radius

    def _random_sampler_wrapper(self, n: int, bounds: torch.Tensor) -> torch.Tensor:
        out = self.random_sampler(n, bounds[0], bounds[1])
        assert out.shape == (n, self.d), "out must be a (n, d) torch tensor"
        return out

    def _random_direction_sampler_wrapper(self, n: int) -> torch.Tensor:
        out = self.random_direction_sampler(n, self.d)
        assert out.shape == (n, self.d), "out must be a (n, d) torch tensor"
        return out


    def _set_penalty_mask(self):
        print("X_all_actual shape: ", self.X_all_actual.shape)
        self.penalty_mask = self._get_penalty_mask(self.X_all_actual)
        print("penalty_mask shape:", self.penalty_mask.shape)

    def _get_gp_data(self, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.penalty_mask.ndim == 1, "penalty mask not 1D. Got penalty mask of shape: " + str(self.penalty_mask.shape)
        # Get non-penalized X and Y values using the boolean mask
        X_non_penalized = self.X_all_actual[self.penalty_mask]
        Y_non_penalized = self.Y_all[self.penalty_mask]

        # Sort by Y values to get indices of top points
        sorted_indices = torch.argsort(Y_non_penalized.squeeze(), descending=True)

        # Take top max_gp_points (or all if less than max_gp_points)
        n_points = min(self.max_gp_points, len(sorted_indices))
        top_indices = sorted_indices[:n_points]

        # Get final X and Y tensors for GP
        X_gp = X_non_penalized[top_indices]
        Y_gp = Y_non_penalized[top_indices]

        return X_gp, Y_gp

    def _get_penalty_mask(self, X: torch.Tensor) -> torch.Tensor:
        # X: (n, d) or (n, l, d)
        if X.ndim == 2:
            # (n, d) -> (n, 1, d)
            X_reshaped = X.unsqueeze(1)
            n = X.shape[0]
            l = 1
        elif X.ndim == 3:
            X_reshaped = X
            n, l, d = X.shape
        else:
            raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")

        # If no needles, all are unpenalized
        if self.needles.shape[0] == 0:
            return torch.ones(n, dtype=torch.bool, device=X.device) if X.ndim == 2 else torch.ones((n, l), dtype=torch.bool, device=X.device)

        # Expand for broadcasting
        # X_reshaped: (n, l, d) -> (n, l, 1, d)
        X_expanded = X_reshaped.unsqueeze(2)
        # needles: (m, d) -> (1, 1, m, d)
        needles_expanded = self.needles.unsqueeze(0).unsqueeze(0)
        # penalty radii: (m, 1) or (m,) -> (1, 1, m)
        penalty_radii_expanded = self.needle_penalty_radii.view(1, 1, -1)

        # Compute distances: (n, l, m)
        distances = torch.norm(X_expanded - needles_expanded, dim=-1)
        within_radius = distances <= penalty_radii_expanded  # (n, l, m)
        penalized = within_radius.any(dim=2)  # (n, l)

        # For 2D input, squeeze to (n,)
        if X.ndim == 2:
            penalized = penalized.squeeze(1)
            assert penalized.shape == (n,)

        return ~penalized

    def _determine_new_bounds(self) -> torch.Tensor:
        # pick top m on squeezed Y to avoid that bogus extra dim
        Y_masked = self.Y_all[self.penalty_mask].squeeze(-1)          # (n,)
        k = min(self.top_m_points, Y_masked.numel())
        top_idx = torch.topk(Y_masked, k).indices                     # (k,)
        X_top = self.X_all_actual[self.penalty_mask][top_idx]         # (k, d)

        min_bounds = X_top.min(dim=0).values                          # (d,)
        max_bounds = X_top.max(dim=0).values                          # (d,)
        return torch.stack([min_bounds, max_bounds], dim=0)           # (2, d)


    def _objective_wrapper(self, X: torch.Tensor, bounds: torch.Tensor, acquisition_function) -> torch.Tensor:
        assert X.shape == (self.d,), "X must be a (d,) torch tensor, got shape " + str(X.shape)
        X_expected, X_actual, Y = self.objective(X, bounds, acquisition_function)

        X_expected = X_expected.to(device=self.device, dtype=self.dtype)
        X_actual = X_actual.to(device=self.device, dtype=self.dtype)
        Y = Y.to(device=self.device, dtype=self.dtype)

        assert isinstance(X_expected, torch.Tensor), "X_expected must be a torch tensor"
        assert isinstance(X_actual, torch.Tensor), "X_actual must be a torch tensor"
        assert isinstance(Y, torch.Tensor), "Y must be a torch tensor"
        # make sure that X_actual is on the simplex
        # assert torch.all(X_actual.sum(dim=1) - 1.0 < 1e-10), "X_actual must be on the simplex, got sum " + str(X_actual.sum(dim=1))

        assert X_expected.shape[1] == self.d, "X_expected must be a (n, d) torch tensor"
        assert X_actual.shape[1] == self.d, "X_actual must be a (n, d) torch tensor"
        assert Y.ndim == 1, "Y must be a (n,) torch tensor"
        assert X_expected.shape[0] == X_actual.shape[0] == Y.shape[0], "X_expected, X_actual, and Y must have the same number of rows"

        # 1) determine penalty mask
        penalty_mask = self._get_penalty_mask(X_actual)

        self.X_all_actual = torch.cat([self.X_all_actual, X_actual], dim=0)
        self.X_all_expected = torch.cat([self.X_all_expected, X_expected], dim=0)
        self.Y_all = torch.cat([self.Y_all, Y.unsqueeze(1)], dim=0)
        self.penalty_mask = torch.cat([self.penalty_mask, penalty_mask], dim=0)

        # Return max of unpenalized values
        unpenalized_Y = Y[penalty_mask]
        unpenalized_X = X_actual[penalty_mask]
        return unpenalized_X, unpenalized_Y

    def _calculate_input_noise(self) -> float:
        """
        Calculate input noise as the average distance between expected and actual points.
        Returns raw (unnormalized) noise level for use in penalty radius calculations.
        """
        if self.X_all_expected.shape[0] == 0:
            return 0.0

        # Calculate Euclidean distances (raw, not normalized)
        distances = torch.norm(self.X_all_expected - self.X_all_actual, dim=1)

        # Use median for robustness against outliers
        noise_level = torch.median(distances).item()

        return noise_level

    def _calculate_normalized_input_noise(self) -> float:
        """
        Calculate input noise as the average distance between expected and actual points.
        Uses normalized Euclidean distance to account for dimensionality.
        """
        if self.X_all_expected.shape[0] == 0:
            return 0.0

        # Calculate Euclidean distances
        distances = torch.norm(self.X_all_expected - self.X_all_actual, dim=1)

        # Normalize by sqrt(dimension) to account for dimensionality
        normalized_distances = distances / torch.sqrt(torch.tensor(self.d, dtype=self.dtype))

        # Use median for robustness against outliers
        noise_level = torch.median(normalized_distances).item()

        return noise_level

    def run(self, max_activations: int = 5, time_limit_hours: float = None):
        """
        Run ZoMBIHop optimization.
        
        Args:
            max_activations: Maximum number of activations (default 5, use float('inf') for unlimited)
            time_limit_hours: Time limit in hours (default None for no limit)
        """
        # CUDA memory management
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"Starting optimization. CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Track start time if time limit is set
        start_time = time.time() if time_limit_hours is not None else None
        
        finished = False
        # Start from saved state if resuming
        start_activation = self.current_activation
        
        activation = start_activation
        while activation < max_activations and not finished:
            self.current_activation = activation
            print(f"{"-"*10} Activation {activation+1} {"-"*10}")
            
            # Check time limit
            if time_limit_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= time_limit_hours:
                    print(f"Time limit of {time_limit_hours} hours reached. Stopping optimization.")
                    finished = True
                    iteration_label = f"act{activation}_timeout"
                    self._save_state(iteration_label, is_permanent=True)
                    break
                print(f"Elapsed time: {elapsed_hours:.2f} / {time_limit_hours:.2f} hours")

            # CUDA memory cleanup between activations
            if self.device.type == 'cuda' and activation > 0:
                torch.cuda.empty_cache()
                print(f"Activation {activation} - CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

            # Reset for new activation if not resuming mid-activation
            if activation > start_activation or self.current_zoom == 0:
                self.no_improvements = 0
            needle = None
            bounds = self.bounds.clone()
            activation_failed = False
            
            start_zoom = self.current_zoom if activation == start_activation else 0
            for zoom in range(start_zoom, self.max_zooms):
                self.current_zoom = zoom
                print(f"{"-"*10} Zoom {zoom+1}/{self.max_zooms}  {"-"*10}")
                print(f"Bounds: {bounds}")
                # 1) get GP data
                X, Y = self._get_gp_data(bounds)

                # 2) fit GP
                self.gp = SingleTaskGP(X, Y)
                mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
                fit_gpytorch_mll(mll)

                start_iteration = self.current_iteration if (activation == start_activation and zoom == start_zoom) else 0
                for iteration in range(start_iteration, self.max_iterations):
                    # Check time limit at each iteration
                    if time_limit_hours is not None:
                        elapsed_hours = (time.time() - start_time) / 3600.0
                        if elapsed_hours >= time_limit_hours:
                            print(f"Time limit of {time_limit_hours} hours reached during iteration. Stopping optimization.")
                            finished = True
                            iteration_label = f"act{activation}_zoom{zoom}_iter{iteration}_timeout"
                            self._save_state(iteration_label, is_permanent=True)
                            break
                    
                    self.current_iteration = iteration
                    print(f"{"-"*10} Iteration {iteration+1}/{self.max_iterations}  {"-"*10}")
                    print(f"Time: {time.time()}")
                    if time_limit_hours is not None:
                        elapsed_hours = (time.time() - start_time) / 3600.0
                        print(f"Elapsed: {elapsed_hours:.2f}h / {time_limit_hours:.2f}h ({elapsed_hours/time_limit_hours*100:.1f}%)")

                    # choose acquisition
                    acq_fn = LogExpectedImprovement(self.gp, best_f=Y.max().item())

                    # wrap it so every test‐point is proj_fn'd and penalized properly
                    class ProjectedAcq(torch.nn.Module):
                        def __init__(self, base, proj, penalty_mask_fn, penalty_value: float = 0.0):
                            """
                            base: The underlying acquisition function (already batched)
                            proj: A projection function that maps Xq to the space expected by `base`
                            penalty_mask_fn: A callable(Xq) -> (N, 1) bool mask where True = not penalized
                            penalty_value: Value to assign to penalized points (typically 0 or very negative)
                            """
                            super().__init__()
                            self.base = base
                            self.proj = proj
                            self.penalty_mask_fn = penalty_mask_fn
                            self.penalty_value = penalty_value

                        def forward(self, Xq: torch.Tensor) -> torch.Tensor:
                            """
                            Xq: (n, q, d) where q=1 for LogExpectedImprovement
                            """
                            X_proj = self.proj(Xq)
                            mask = self.penalty_mask_fn(X_proj)

                            # Get base acquisition values
                            base_acq = self.base(X_proj)

                            # Apply penalty using torch.where (differentiable!)
                            penalized_acq = torch.where(
                                mask.squeeze(-1),
                                base_acq,
                                torch.full_like(base_acq, self.penalty_value)
                            )

                            return penalized_acq

                    
                    acq = ProjectedAcq(
                        base=acq_fn,
                        proj=self.proj_fn,
                        penalty_mask_fn=self._get_penalty_mask,
                        penalty_value=-100.0  # or 0.0 if you want ties
                    )
                    
                    # Generate many candidate starting points
                    ic_candidates = self._random_sampler_wrapper(self.raw, bounds)  # (500, d)
                    ic_candidates_3d = ic_candidates.unsqueeze(1)  # (500, 1, d)

                    # Evaluate acquisition function on all candidates
                    with torch.no_grad():  # No need for gradients during acquisition evaluation
                        acq_values = acq(ic_candidates_3d).squeeze()  # (500,)

                    # Get mask for unpenalized points
                    unpenalized_mask = self._get_penalty_mask(ic_candidates)  # (500, 1)
                    unpenalized_indices = torch.where(unpenalized_mask.squeeze())[0]

                    # Try up to 5 times to get enough unpenalized points
                    attempt = 0
                    max_attempts = 5
                    current_ic_candidates = ic_candidates
                    current_ic_candidates_3d = ic_candidates_3d
                    current_acq_values = acq_values
                    current_unpenalized_indices = unpenalized_indices

                    while len(current_unpenalized_indices) < self.n_restarts and attempt < max_attempts:
                        attempt += 1
                        print(f"Attempt {attempt}: Only {len(current_unpenalized_indices)} unpenalized points found, need {self.n_restarts}")

                        # Generate additional random points
                        additional_points = self._random_sampler_wrapper(self.raw, bounds)  # (raw, d)
                        additional_points_3d = additional_points.unsqueeze(1)  # (raw, 1, d)

                        # Evaluate acquisition function on additional points
                        with torch.no_grad():
                            additional_acq_values = acq(additional_points_3d).squeeze()  # (raw,)

                        # Get mask for additional unpenalized points
                        additional_unpenalized_mask = self._get_penalty_mask(additional_points)  # (raw,)
                        additional_unpenalized_indices = torch.where(additional_unpenalized_mask.squeeze())[0]

                        # Combine with existing points
                        current_ic_candidates = torch.cat([current_ic_candidates, additional_points], dim=0)
                        current_ic_candidates_3d = torch.cat([current_ic_candidates_3d, additional_points_3d], dim=0)
                        current_acq_values = torch.cat([current_acq_values, additional_acq_values], dim=0)

                        # Update unpenalized indices (need to offset the additional indices)
                        offset = len(current_unpenalized_indices)
                        additional_unpenalized_indices_offset = additional_unpenalized_indices + len(current_ic_candidates) - len(additional_points)
                        current_unpenalized_indices = torch.cat([current_unpenalized_indices, additional_unpenalized_indices_offset], dim=0)

                        print(f"After attempt {attempt}: {len(current_unpenalized_indices)} total unpenalized points")

                    if len(current_unpenalized_indices) < self.n_restarts:
                        print(f"Warning: After {max_attempts} attempts, only got {len(current_unpenalized_indices)} unpenalized points, need {self.n_restarts}")
                        if len(current_unpenalized_indices) < 0.1 * self.n_restarts:
                            print("No unpenalized points found, breaking")
                            activation_failed = True
                            break
                        print("Using all available unpenalized points")
                        num_restarts_to_use = len(current_unpenalized_indices)
                    else:
                        num_restarts_to_use = self.n_restarts

                    # Get acquisition values for unpenalized points only
                    unpenalized_acq_values = current_acq_values[current_unpenalized_indices]

                    # Select the top num_restarts_to_use candidates from unpenalized points
                    top_unpenalized_indices = torch.argsort(unpenalized_acq_values, descending=True)[:num_restarts_to_use]
                    selected_indices = current_unpenalized_indices[top_unpenalized_indices]
                    initial_conditions = current_ic_candidates_3d[selected_indices]  # (num_restarts_to_use, 1, d)

                    # Clean up large tensors to free GPU memory
                    if self.device.type == 'cuda':
                        del current_ic_candidates, current_ic_candidates_3d, current_acq_values
                        del current_unpenalized_indices, unpenalized_acq_values, top_unpenalized_indices
                        torch.cuda.empty_cache()

                                        # Now optimize with exactly the number of initial conditions we have
                    try:
                        candidate, acq_val = optimize_acqf(
                            acq_function=acq,
                            bounds=bounds,
                            q=1,
                            num_restarts=num_restarts_to_use,
                            raw_samples=None,  # Don't generate any new samples
                            batch_initial_conditions=initial_conditions,
                            options={"maxiter": 50},
                        )
                    except Exception as e:
                        print(f"Warning: optimize_acqf failed with error: {e}")
                        print("Falling back to using the best initial condition directly")
                        # Fallback: use the best initial condition directly
                        best_idx = torch.argmax(unpenalized_acq_values).item()
                        candidate = current_ic_candidates_3d[current_unpenalized_indices[best_idx]].unsqueeze(0)  # Add batch dimension
                        acq_val = unpenalized_acq_values[best_idx].unsqueeze(0)  # Add batch dimension
                    # Get previous best point and value
                    prev_max_idx = self.Y_all[self.penalty_mask].argmax()
                    prev_max_Y = self.Y_all[self.penalty_mask][prev_max_idx]
                    prev_max_X = self.X_all_actual[self.penalty_mask][prev_max_idx]

                    # Project and evaluate candidate
                    candidate = self.proj_fn(candidate).squeeze(0)  # Convert from (1,d) to (d,)
                    print(f"Sampling candidate: {candidate}")
                    unpenalized_X, unpenalized_Y = self._objective_wrapper(candidate, bounds, acq)

                    # update gp with new data
                    X, Y = self._get_gp_data(bounds)
                    print(f"Number of points in GP: {X.shape[0]}")
                    self.gp = SingleTaskGP(X, Y)
                    mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
                    fit_gpytorch_mll(mll)

                    if unpenalized_Y.shape[0] == 0:
                        print("No unpenalized Y values, breaking")
                        activation_failed = True
                        # Save state before breaking (permanent - marks end of zoom)
                        iteration_label = f"act{activation}_zoom{zoom}_iter{iteration}_failed"
                        self._save_state(iteration_label, is_permanent=True)
                        break

                    # Get current best point and value
                    curr_max_idx = unpenalized_Y.argmax()
                    curr_max_Y = unpenalized_Y[curr_max_idx]
                    curr_max_X = unpenalized_X[curr_max_idx]

                    # Calculate noise thresholds
                    average_output_noise = self.gp.likelihood.noise_covar.noise.mean().item()
                    input_noise = self._calculate_normalized_input_noise()
                    output_improvement_threshold = self.improvement_threshold_noise_mult * average_output_noise
                    input_change_threshold = self.input_noise_threshold_mult * input_noise

                    # Check if no improvement: output difference small AND input points close
                    input_distance = torch.norm(curr_max_X - prev_max_X)
                    if (curr_max_Y - prev_max_Y < output_improvement_threshold and
                        input_distance < input_change_threshold):
                        self.no_improvements += 1
                    else:
                        self.no_improvements = 0
                    
                    # Save state after each iteration (not permanent - will be cleaned up)
                    iteration_label = f"act{activation}_zoom{zoom}_iter{iteration}"
                    self._save_state(iteration_label, is_permanent=False)

                    self.penalty_mask = self._get_penalty_mask(self.X_all_actual)

                    print(f"No improvements: {self.no_improvements}")
                    print(f"Current max Y: {curr_max_Y}")
                    print(f"Previous max Y: {prev_max_Y}")
                    print(f"Input distance: {input_distance}")
                    print(f"Overall max Y: {self.Y_all[self.penalty_mask].max()}")
                    print(f"Overall max X: {self.X_all_actual[self.penalty_mask][self.Y_all[self.penalty_mask].argmax()]}")

                    if self.no_improvements >= self.n_consecutive_no_improvements:
                        # max_idx is within the masked (unpenalized) view
                        max_idx = self.Y_all[self.penalty_mask].argmax()
                        # Map masked index to global index so values/indices are consistent
                        unpenalized_indices = torch.where(self.penalty_mask)[0]
                        global_idx = unpenalized_indices[max_idx]
                        needle = self.X_all_actual[global_idx]
                        assert isinstance(needle, torch.Tensor), "needle must be a torch tensor"
                        assert needle.shape == (self.d,), "needle must be a (d,) torch tensor"

                        print(f"No improvements! Found needle {needle} with value {self.Y_all[global_idx]}")

                        # get global gp data and fit it
                        X, Y = self._get_gp_data(self.bounds)
                        self.gp = SingleTaskGP(X, Y)
                        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
                        fit_gpytorch_mll(mll)

                        acq_fn = LogExpectedImprovement(self.gp, best_f=Y.max().item())

                        acq = ProjectedAcq(
                            base=acq_fn,
                            proj=self.proj_fn,
                            penalty_mask_fn=self._get_penalty_mask,
                            penalty_value=-1e6  # or 0.0 if you want ties
                        )

                        # 1) Determine penalty radius using global gp
                        penalty_radius = self._determine_penalty_radius(needle, acq)
                        print(f"Penalizing with penalty radius {penalty_radius}")

                        # 2) add needle to needles
                        self.needles_results.append({
                            'point': needle.clone(),  # Keep as tensor
                            'value': self.Y_all[global_idx].item(),
                            'activation': activation,
                            'zoom': zoom,
                            'iteration': iteration
                        })
                        self.needles = torch.cat([self.needles, needle.unsqueeze(0)], dim=0)
                        self.needle_vals = torch.cat([self.needle_vals, self.Y_all[global_idx].reshape(1, 1)], dim=0)
                        self.needle_indices = torch.cat([self.needle_indices, global_idx.reshape(1, 1)], dim=0)
                        self.needle_penalty_radii = torch.cat([self.needle_penalty_radii, torch.tensor([[penalty_radius]], device=self.device, dtype=self.dtype)], dim=0)

                        # 4) update penalty mask
                        self._set_penalty_mask()
                        
                        # Save state after finding needle (permanent - marks end of zoom)
                        iteration_label = f"act{activation}_zoom{zoom}_iter{iteration}_needle"
                        self._save_state(iteration_label, is_permanent=True)
                        break
                
                # Check if we need to break due to time limit
                if finished:
                    break

                if needle is not None or activation_failed:
                    # Generate many candidate starting points
                    ic_candidates = self._random_sampler_wrapper(self.raw, self.bounds)  # (500, d)
                    ic_candidates_3d = ic_candidates.unsqueeze(1)  # (500, 1, d)

                    # Evaluate acquisition function on all candidates
                    with torch.no_grad():  # No need for gradients during acquisition evaluation
                        acq_values = acq(ic_candidates_3d).squeeze()  # (500,)

                    # Get mask for unpenalized points
                    unpenalized_mask = self._get_penalty_mask(ic_candidates)  # (500, 1)
                    # Calculate percentage of penalized points
                    total_points = unpenalized_mask.numel()
                    unpenalized_points = unpenalized_mask.sum().item()
                    penalized_points = total_points - unpenalized_points
                    penalized_percentage = (penalized_points / total_points) * 100
                    if penalized_percentage > 90:
                        print(f"Too much area penalized: {penalized_percentage:.2f}%. Ending optimization.")
                        finished = True
                        # Save state before ending (permanent - optimization complete)
                        iteration_label = f"act{activation}_zoom{zoom}_finished"
                        self._save_state(iteration_label, is_permanent=True)
                    break

                if finished:
                    break
                if zoom < self.max_zooms - 1:
                    bounds = self._determine_new_bounds()
                    # Save state after zoom bounds update (permanent - marks end of zoom)
                    iteration_label = f"act{activation}_zoom{zoom}_complete"
                    self._save_state(iteration_label, is_permanent=True)
            
            # Increment activation counter
            activation += 1

        # 5) return the needles

        # Final CUDA memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"Optimization complete. Final CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Save final state (permanent)
        final_label = "final"
        self._save_state(final_label, is_permanent=True)
        print(f"Optimization complete. Run UUID: {self.run_uuid}")

        return self.needles_results, self.needles, self.needle_vals, self.X_all_actual, self.Y_all

    # your helpers
    @staticmethod
    def _subset_sums_and_signs(caps: torch.Tensor) -> tuple:
        """
        Compute subset sums and inclusion-exclusion signs for CFS algorithm.
        """
        m = caps.numel()
        device, dtype = caps.device, caps.dtype
        subset_sums = torch.zeros(1 << m, dtype=dtype, device=device)
        for mask in range(1, 1 << m):  # DP peel lowbit
            lsb = mask & -mask
            j = (lsb.bit_length() - 1)
            subset_sums[mask] = subset_sums[mask ^ lsb] + caps[j]
        # signs:  +1 for even |J|, ‑1 for odd |J|
        signs = torch.tensor([1 - 2 * (mask.bit_count() & 1) for mask in range(1 << m)],
                                dtype=dtype, device=device)
        return subset_sums, signs

    @staticmethod
    def _polytope_volume(S: torch.Tensor, subset_sums: torch.Tensor,
                        signs: torch.Tensor, power: int, denom: int) -> torch.Tensor:
        """
        Compute polytope volume or antiderivative using inclusion-exclusion principle.
        """
        # Input validation
        if torch.any(torch.isnan(S)) or torch.any(torch.isinf(S)):
            print("🐛 Warning: NaN/Inf in S input to _polytope_volume")
            S = torch.clamp(S, min=0.0, max=1e6)

        # Shape (B, 2**m)
        shifted = (S.unsqueeze(1) - subset_sums)
        positive = torch.clamp(shifted, min=0.0)

        # Numerical stability: clamp power to avoid overflow
        if power > 50:  # Very high powers can cause overflow
            positive = torch.clamp(positive, max=10.0)

        powered = positive.pow(power)

        # Check for numerical issues in intermediate results
        if torch.any(torch.isnan(powered)) or torch.any(torch.isinf(powered)):
            print(f"🐛 Warning: NaN/Inf in powered terms (power={power})")
            print(f"   positive range: [{torch.min(positive):.2e}, {torch.max(positive):.2e}]")
            powered = torch.nan_to_num(powered, nan=0.0, posinf=1e6, neginf=0.0)

        result = (signs * powered).sum(dim=1) / denom

        # Final safety check
        result = torch.clamp(result, min=0.0)  # Volume should be non-negative
        if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
            print(f"🐛 Warning: NaN/Inf in _polytope_volume result")
            result = torch.nan_to_num(result, nan=1e-15, posinf=1e6, neginf=1e-15)

        return result

    @staticmethod
    def random_simplex(
        num_samples: int,
        a: torch.Tensor,
        b: torch.Tensor,
        S: float = 1.0,
        max_batch: int = None,
        debug: bool = False,
        device: str = 'cuda',
        torch_dtype: torch.dtype = torch.float64,
        **ignored,
    ) -> torch.Tensor:
        """
        Generate highly-parallel CFS samples from bounded simplex.
        Returns a 2D tensor of shape (num_samples, dimension).
        """
        import math
        # Use float64 for high precision calculations, convert back to self.torch_dtype at the end
        a = a.to(device=device, dtype=torch.float64)
        b = b.to(device=device, dtype=torch.float64)
        d = a.numel()

        a = a.flatten()
        b = b.flatten()
        caps_full = (b - a).flatten()
        assert caps_full.ndim == 1

        if d > 20:
            raise ValueError("Analytic CFS variant supports dimension ≤ 20")
        if not torch.all(b >= a):
            raise ValueError("Each upper bound must exceed the lower bound")

        S = torch.as_tensor(S, dtype=torch.float64, device=device)
        if not (a.sum() - 1e-12 <= S <= b.sum() + 1e-12):
            raise ValueError("Sum S outside feasible range")

        # Calculate optimal batch size if not provided
        if max_batch is None:
            if device == 'cuda':
                # GPU: Memory-constrained batching
                estimated_bytes_per_point = d * 8 * 32  # Conservative estimate
                gpu_memory = torch.cuda.get_device_properties(device).total_memory
                available_memory = gpu_memory - torch.cuda.memory_allocated(device)
                safe_memory = available_memory * 0.8  # Fixed 0.8 instead of undefined max_gpu_percentage
                max_batch = max(int(safe_memory / estimated_bytes_per_point), 1000)
            else:
                # CPU: Large batches
                max_batch = min(num_samples, 10_000_000)

        if num_samples > 10_000:
            print(f"🚀 Generating {num_samples:,} samples using highly-parallel CFS")
            print(f"   Max batch size: {max_batch}")
            print(f"   Device: {device}")

        caps_full = b - a           # (d,)
        S0 = S - a.sum()            # remaining sum after shift to 0‑based caps

        # Create RNG
        rng = torch.Generator(device=device)

        # Pre‑compute subset‑sums / signs for each suffix caps[k:]
        precomp = []
        for k in range(d - 1):  # last coordinate is deterministic
            subset_sums, signs = ZoMBIHop._subset_sums_and_signs(caps_full[k+1:])
            precomp.append((subset_sums, signs))

        out = torch.empty((num_samples, d), dtype=torch.float64, device=device)
        written = 0

        while written < num_samples:
            B = min(max_batch, num_samples - written)

            # Running arrays for the *entire* batch
            S_rem = S0.expand(B).clone()      # (B,)
            caps_rem = caps_full.clone()      # (d,)
            y = torch.empty((B, d), dtype=torch.float64, device=device)

            # Iterate over coordinates – small loop (≤20)
            for k in range(d - 1):
                subset_sums, signs = precomp[k]
                m = d - k - 1
                denom_vol = math.factorial(m - 1)
                denom_int = math.factorial(m)

                # Feasible interval per sample
                sum_tail_caps = caps_full[k+1:].sum()  # Use caps_full, not caps_rem
                t_low = torch.clamp(S_rem - sum_tail_caps, min=0.0)
                t_high = torch.minimum(caps_full[k].expand_as(S_rem), S_rem)

                # Short‑circuit deterministic cases (interval of length 0)
                deterministic_mask = (t_high - t_low) < 1e-15
                yk = torch.zeros_like(S_rem)  # Initialize with zeros
                yk[deterministic_mask] = t_low[deterministic_mask]

                # Stochastic sub‑batch
                stochastic_mask = ~deterministic_mask
                n_stochastic = stochastic_mask.sum().item()

                if n_stochastic > 0:
                    # Gather vectors for active samples
                    S_todo = S_rem[stochastic_mask]
                    tl = t_low[stochastic_mask]
                    th = t_high[stochastic_mask]

                    # Debug: Check for problematic intervals
                    interval_sizes = th - tl
                    if debug and torch.any(interval_sizes <= 0):
                        print(f"🐛 Debug: Zero or negative intervals detected in coord {k}")
                        print(f"   Min interval: {torch.min(interval_sizes):.2e}")
                        print(f"   Max interval: {torch.max(interval_sizes):.2e}")

                    # Skip samples with very small intervals to avoid numerical issues
                    valid_mask = interval_sizes > 1e-12
                    if torch.sum(valid_mask) == 0:
                        # All intervals too small, use midpoint
                        yk[stochastic_mask] = (tl + th) / 2
                        continue

                    # Only process valid samples
                    S_todo = S_todo[valid_mask]
                    tl_valid = tl[valid_mask]
                    th_valid = th[valid_mask]

                    # Helper lambdas (vectorised) with enhanced stability
                    def _volume(t: torch.Tensor) -> torch.Tensor:
                        vol = ZoMBIHop._polytope_volume(S_todo - t, subset_sums, signs, m - 1, denom_vol)
                        # Clamp volume to avoid negative values due to numerical errors
                        return torch.clamp(vol, min=1e-15)

                    def _cdf(t: torch.Tensor) -> torch.Tensor:
                        shifted = S_todo - t
                        I_high = ZoMBIHop._polytope_volume(shifted, subset_sums, signs, m, denom_int)
                        shifted_low = S_todo - tl_valid
                        I_low = ZoMBIHop._polytope_volume(shifted_low, subset_sums, signs, m, denom_int)

                        # Ensure I_low >= I_high to avoid negative CDFs
                        I_low = torch.clamp(I_low, min=0.0)
                        I_high = torch.clamp(I_high, min=0.0)
                        I_high = torch.minimum(I_high, I_low)

                        cdf_val = I_low - I_high
                        return torch.clamp(cdf_val, min=1e-15)

                    # Normalisation constant Z ≡ integral from tl to th
                    Z = _cdf(th_valid)

                    # Debug: Check Z values
                    if torch.any(Z <= 0) or torch.any(torch.isnan(Z)) or torch.any(torch.isinf(Z)):
                        if debug:
                            print(f"🐛 Debug: Invalid Z values in coord {k}")
                            print(f"   Z min: {torch.min(Z):.2e}, max: {torch.max(Z):.2e}")
                            print(f"   NaN count: {torch.sum(torch.isnan(Z))}, Inf count: {torch.sum(torch.isinf(Z))}")
                        # Use fallback for all invalid samples
                        Z = torch.clamp(Z, min=1e-15)

                    U = torch.rand(len(tl_valid), generator=rng, device=device, dtype=torch.float64)
                    target = U * Z

                    # Newton iterations with enhanced stability and debugging
                    t = tl_valid + U * (th_valid - tl_valid)

                    for iteration in range(12):  # Increased iterations for better convergence
                        f = _cdf(t) - target
                        fp = _volume(t)

                        # Enhanced safe division
                        fp_safe = torch.clamp(fp.abs(), min=1e-15)
                        fp_sign = torch.sign(fp)

                        # Check for problematic derivatives
                        if debug and torch.any(fp_safe < 1e-12):
                            print(f"🐛 Debug: Very small derivatives in coord {k}, iter {iteration}")
                            print(f"   fp min: {torch.min(fp):.2e}, max: {torch.max(fp):.2e}")

                        delta = f / (fp_safe * fp_sign)

                        # Limit step size to prevent wild jumps
                        step_limit = 0.1 * (th_valid - tl_valid)
                        delta = torch.clamp(delta, -step_limit, step_limit)

                        t_new = t - delta
                        t = torch.clamp(t_new, tl_valid, th_valid)

                        # Check for NaN/Inf in intermediate results
                        if torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)):
                            if debug:
                                print(f"🐛 Debug: NaN/Inf in Newton iteration {iteration}, coord {k}")
                                print(f"   delta range: [{torch.min(delta):.2e}, {torch.max(delta):.2e}]")
                                print(f"   f range: [{torch.min(f):.2e}, {torch.max(f):.2e}]")
                                print(f"   fp range: [{torch.min(fp):.2e}, {torch.max(fp):.2e}]")
                            # Force fallback
                            t = tl_valid + torch.rand_like(tl_valid) * (th_valid - tl_valid)
                            break

                        # Check for convergence with relaxed tolerance
                        if torch.all(torch.abs(f) < 1e-10):
                            break

                    # Assign results back, handling the valid_mask
                    if torch.sum(valid_mask) > 0:
                        yk_temp = torch.zeros_like(S_rem[stochastic_mask])
                        yk_temp[valid_mask] = t
                        yk_temp[~valid_mask] = (tl[~valid_mask] + th[~valid_mask]) / 2  # Midpoint for invalid
                        yk[stochastic_mask] = yk_temp
                    else:
                        yk[stochastic_mask] = (tl + th) / 2

                # Validate coordinate values
                if torch.any(torch.isnan(yk)) or torch.any(torch.isinf(yk)):
                    print(f"⚠️  NaN/Inf detected in coordinate {k}, using enhanced fallback sampling")

                    if debug:
                        # Debug information about the problematic values
                        nan_count = torch.sum(torch.isnan(yk))
                        inf_count = torch.sum(torch.isinf(yk))
                        print(f"   NaN count: {nan_count}, Inf count: {inf_count}")
                        print(f"   S_rem range: [{torch.min(S_rem):.4f}, {torch.max(S_rem):.4f}]")
                        print(f"   t_low range: [{torch.min(t_low):.4f}, {torch.max(t_high):.4f}]")
                        print(f"   t_high range: [{torch.min(t_low):.4f}, {torch.max(t_high):.4f}]")

                    # Enhanced fallback: Use Beta distribution for more principled sampling
                    # Beta(2,2) gives a bell-shaped distribution on [0,1], more realistic than uniform
                    alpha, beta = 2.0, 2.0
                    U_beta = torch.distributions.Beta(alpha, beta).sample((len(S_rem),)).to(device=device, dtype=torch.float64)

                    # Ensure feasible intervals
                    interval_size = torch.clamp(t_high - t_low, min=1e-15)
                    yk = t_low + U_beta * interval_size

                    # Final safety check and clamp to valid range
                    yk = torch.clamp(yk, min=t_low, max=t_high)

                    if debug:
                        print(f"   Fallback yk range: [{torch.min(yk):.4f}, {torch.max(yk):.4f}]")

                # Write and update running state
                y[:, k] = yk
                S_rem -= yk

                # Ensure S_rem stays non-negative
                S_rem = torch.clamp(S_rem, min=0.0)

            # Last coordinate deterministic
            y[:, -1] = torch.minimum(torch.maximum(S_rem, torch.zeros_like(S_rem)),
                                   caps_full[-1].expand_as(S_rem))

            out[written: written+B] = y + a  # shift back
            written += B

            # Progress for large datasets
            if num_samples > 100_000 and written % max(max_batch, 100_000) == 0:
                progress = written / num_samples * 100
                print(f"CFS Progress: {progress:.1f}% ({written:,}/{num_samples:,})")

        if num_samples > 10_000:
            print(f"✅ Generated {len(out):,} highly-parallel CFS samples")

        # Ensure output is always a 2D tensor of shape (num_samples, d)
        out = out.reshape(num_samples, d)

        # Convert back to consistent dtype for compatibility with GP models
        return out.to(dtype=torch_dtype)

    @staticmethod
    def proj_simplex(X):
        # Handle input dimensions - either (n,d) or (n,l,d)
        original_dim = X.dim()

        # Reshape to 2D (n*l, d) if needed while preserving gradients
        if original_dim == 3:
            n, l, d = X.shape
            X_2d = X.reshape(-1, d)  # Reshape to (n*l, d)
        else:
            X_2d = X

        # Project each row onto simplex using differentiable operations
        u, _ = torch.sort(X_2d, descending=True, dim=-1)
        css = torch.cumsum(u, dim=-1)
        d = X_2d.size(-1)
        indices = torch.arange(1, d+1, device=X_2d.device, dtype=X_2d.dtype)
        rho = torch.sum((u * indices) > (css - 1), dim=-1) - 1

        # Compute theta using gathered cumsum values
        batch_indices = torch.arange(X_2d.size(0), device=X_2d.device)
        theta = (torch.gather(css, 1, rho.unsqueeze(-1)).squeeze(-1) - 1) / (rho + 1).to(X_2d.dtype)

        # Apply projection using broadcasting
        result = torch.maximum(X_2d - theta.unsqueeze(-1), torch.zeros_like(X_2d))

        # Restore original shape if input was 3D
        if original_dim == 3:
            result = result.reshape(n, l, d)

        return result

    @staticmethod
    def random_zero_sum_directions(n: int, d: int, device='cuda') -> torch.Tensor:
        """
        Sample `n` vectors of dimension `d` that each have zero sum and unit norm,
        using double‐precision floats for maximal numerical accuracy.

        Args:
            n (int): Number of vectors to sample.
            d (int): Dimensionality of each vector.
        Returns:
            torch.Tensor: (n, d) tensor of zero-sum, unit-norm vectors, dtype=torch.float64.
        """
        return zero_sum_dirs(n, d, device=device)

def zero_sum_dirs(k: int, d: int, device=None, dtype=torch.float64, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    v = torch.randn(k, d, device=device, dtype=dtype)
    v -= v.mean(dim=1, keepdim=True)                   # sum=0
    n = v.norm(dim=1, keepdim=True)
    # resample zero-norm rows
    mask = (n.squeeze() == 0)
    while mask.any():
        r = torch.randn(mask.sum(), d, device=device, dtype=dtype)
        r -= r.mean(dim=1, keepdim=True)
        v[mask] = r
        n[mask] = v[mask].norm(dim=1, keepdim=True)
        mask = (n.squeeze() == 0)
    return v / n

def line_simplex_segment(x0: torch.Tensor, d: torch.Tensor):
    """
    x0: (d,)
    d:  (d,)
    returns: t_min, t_max, x_left, x_right
    """
    neg = d < 0
    pos = d > 0

    # +inf / -inf helpers
    inf  = torch.tensor(float('inf'),  device=d.device, dtype=d.dtype)
    ninf = torch.tensor(float('-inf'), device=d.device, dtype=d.dtype)

    t_max = torch.min(torch.where(neg, -x0/ d, inf))
    t_min = torch.max(torch.where(pos, -x0/ d, ninf))

    # If infeasible (no intersection), you could return None
    if t_min > t_max:
        return None

    x_left  = x0 + t_min * d
    x_right = x0 + t_max * d
    return t_min, t_max, x_left, x_right

def batch_line_simplex_segments(x0: torch.Tensor, D: torch.Tensor):
    # x0: (d,), D: (k,d)
    neg = D < 0
    pos = D > 0

    inf  = torch.tensor(float('inf'),  device=D.device, dtype=D.dtype)
    ninf = torch.tensor(float('-inf'), device=D.device, dtype=D.dtype)

    t_max = torch.min(torch.where(neg, -x0.unsqueeze(0)/D, inf), dim=1).values
    t_min = torch.max(torch.where(pos, -x0.unsqueeze(0)/D, ninf), dim=1).values

    mask  = t_min <= t_max
    t_min = t_min[mask]
    t_max = t_max[mask]
    Dm    = D[mask]

    x_left  = x0.unsqueeze(0) + t_min.unsqueeze(1)*Dm
    x_right = x0.unsqueeze(0) + t_max.unsqueeze(1)*Dm

    return x_left, x_right, t_min, t_max, mask

class LineBO:
    """
    Simplified Line-based Bayesian Optimization for simplex-constrained problems.
    Uses float64 throughout for numerical stability.
    """

    def __init__(self, objective_function, dimensions: int, num_points_per_line: int = 100,
                 num_lines: int = 20, device: str = 'cuda'):
        """
        Initialize LineBO sampler.

        Args:
            objective_function: Function that accepts line endpoints and returns x_actual, y
            dimensions: Number of dimensions (d)
            num_points_per_line: Number of points to sample along each line for integration
            num_lines: Number of candidate lines to generate and evaluate
            device: Device to use for computations ('cuda' by default)
        """
        self.objective_function = objective_function
        self.d = dimensions
        self.num_points_per_line = num_points_per_line
        self.num_lines = num_lines
        self.device = torch.device(device)
        self.dtype = torch.float64  # Force float64 for numerical stability

    def _integrate_acquisition_along_lines(self, x_left: torch.Tensor, x_right: torch.Tensor,
                                         acquisition_function: nn.Module) -> torch.Tensor:
        """
        Integrate acquisition function along multiple line segments.

        Args:
            x_left: (k, d) left endpoints of line segments
            x_right: (k, d) right endpoints of line segments
            acquisition_function: Acquisition function to evaluate

        Returns:
            (k,) tensor of integrated acquisition values
        """
        k = x_left.shape[0]

        # Generate evenly spaced points along each line
        t_values = torch.linspace(0, 1, self.num_points_per_line,
                                device=self.device, dtype=torch.float64)

        # Create points along all lines: (k, num_points_per_line, d)
        points = (x_left.unsqueeze(1) * (1 - t_values).view(1, -1, 1) +
                 x_right.unsqueeze(1) * t_values.view(1, -1, 1))

        # Reshape for batch evaluation: (k * num_points_per_line, 1, d)
        points_flat = points.reshape(-1, 1, self.d)

        # Get acquisition values - Process in batches to avoid memory overflow
        batch_size = 500  # Process 500 points at a time
        num_points = points_flat.shape[0]
        acquisition_values_flat = torch.empty(num_points, device=self.device, dtype=torch.float64)
        
        with torch.no_grad():
            for i in range(0, num_points, batch_size):
                end_idx = min(i + batch_size, num_points)
                batch = points_flat[i:end_idx]
                
                try:
                    batch_values = acquisition_function(batch)
                    acquisition_values_flat[i:end_idx] = batch_values.squeeze()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Further reduce batch size for this problematic batch
                        smaller_batch_size = batch_size // 4
                        for j in range(i, end_idx, smaller_batch_size):
                            small_end_idx = min(j + smaller_batch_size, end_idx)
                            small_batch = points_flat[j:small_end_idx]
                            small_batch_values = acquisition_function(small_batch)
                            acquisition_values_flat[j:small_end_idx] = small_batch_values.squeeze()
                    else:
                        raise e
                        
                # Clean up GPU memory after each batch
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Reshape and integrate: (k, num_points_per_line) -> (k,)
        acquisition_values = acquisition_values_flat.reshape(k, self.num_points_per_line)
        integrated = acquisition_values.mean(dim=1)  # Use mean for stability

        return integrated

    def sampler(self, x_tell: torch.Tensor, bounds: torch.Tensor = None,
                acquisition_function: nn.Module = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points using simplified LineBO algorithm.

        Args:
            x_tell: Starting point (d,) - should be on simplex
            bounds: Bounds tensor (2, d) with lower/upper bounds
            acquisition_function: Acquisition function to optimize

        Returns:
            Tuple of (x_requested, x_actual, y)
        """
        # Ensure float64 precision
        x_tell = x_tell.to(device=self.device, dtype=self.dtype)
        if bounds is not None:
            bounds = bounds.to(device=self.device, dtype=self.dtype)

        # Validate x_tell is on simplex
        assert abs(x_tell.sum().item() - 1.0) < 1e-12, f"x_tell must sum to 1, got {x_tell.sum().item()}"

        # Generate zero-sum directions
        directions = zero_sum_dirs(self.num_lines * 2, self.d,
                                 device=self.device, dtype=self.dtype)

        # Find valid line segments in batch
        x_left, x_right, t_min, t_max, valid_mask = batch_line_simplex_segments(x_tell, directions)

        # Check if we have enough valid lines
        if x_left.shape[0] < self.num_lines:
            print(f"Warning: Only found {x_left.shape[0]} valid lines, needed {self.num_lines}")
            # Generate more directions if needed
            while x_left.shape[0] < self.num_lines:
                additional_dirs = zero_sum_dirs(self.num_lines, self.d,
                                              device=self.device, dtype=self.dtype)
                x_left_add, x_right_add, t_min_add, t_max_add, mask_add = batch_line_simplex_segments(x_tell, additional_dirs)

                if x_left_add.shape[0] > 0:
                    x_left = torch.cat([x_left, x_left_add], dim=0)
                    x_right = torch.cat([x_right, x_right_add], dim=0)
                    t_min = torch.cat([t_min, t_min_add], dim=0)
                    t_max = torch.cat([t_max, t_max_add], dim=0)
                else:
                    break

        # Take only the number of lines we need
        if x_left.shape[0] > self.num_lines:
            x_left = x_left[:self.num_lines]
            x_right = x_right[:self.num_lines]
            t_min = t_min[:self.num_lines]
            t_max = t_max[:self.num_lines]

        # Select best line based on acquisition function
        if acquisition_function is None:
            # Random selection
            best_idx = torch.randint(0, x_left.shape[0], (1,), device=self.device).item()
        else:
            # Integrate acquisition along all lines
            integrated_values = self._integrate_acquisition_along_lines(
                x_left, x_right, acquisition_function)
            best_idx = torch.argmax(integrated_values).item()

        # Get the selected line endpoints
        selected_left = x_left[best_idx]
        selected_right = x_right[best_idx]

        # Create line endpoints tensor for objective function
        endpoints = torch.stack([selected_left, selected_right], dim=0)  # (2, d)

        # Call objective function
        x_actual, y = self.objective_function(endpoints)

        # Validate returns and ensure float64
        assert torch.is_tensor(x_actual) and x_actual.shape[1] == self.d
        assert torch.is_tensor(y) and y.shape[0] == x_actual.shape[0]

        x_actual = x_actual.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        # Generate x_requested by fitting a line through x_actual points
        if x_actual.shape[0] > 1:
            # Use PCA to find best-fit line
            x_centered = x_actual - x_actual.mean(dim=0, keepdim=True)
            U, S, V = torch.svd(x_centered)
            direction = V[:, 0]  # First principal component

            # Project points onto this line
            projections = torch.matmul(x_centered, direction.unsqueeze(1)).squeeze(1)

            # Create evenly spaced projections
            t_vals = torch.linspace(projections.min(), projections.max(),
                                  x_actual.shape[0], device=self.device, dtype=torch.float64)

            # Reconstruct x_requested
            x_requested = x_actual.mean(dim=0).unsqueeze(0) + t_vals.unsqueeze(1) * direction.unsqueeze(0)
        else:
            # Single point case
            x_requested = x_actual.clone()

        return x_requested, x_actual, y