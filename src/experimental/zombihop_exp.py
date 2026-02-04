"""
ZoMBI-Hop: Zooming Multi-Basin Identification with Hopping
==========================================================

A novel Bayesian optimization algorithm for discovering multiple optima
in simplex-constrained spaces, designed for materials research applications.
"""

import torch
import time
from typing import Callable, Tuple, Optional, List

from ..utils.simplex import (
    proj_simplex,
    random_simplex,
    random_zero_sum_directions,
)
from ..utils.datahandler import DataHandler
from ..utils.gp_simplex import GPSimplex
from ..utils.dataclasses import ZoMBIHopConfig


# CUDA optimization settings
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class ZoMBIHop:
    """
    Zooming Multi-Basin Identification with Hopping.

    A Bayesian optimization algorithm that efficiently discovers multiple
    local optima in simplex-constrained spaces through:

    1. **Zooming**: Iteratively narrowing the search space around promising regions
    2. **Basin Identification**: Detecting when convergence to a local optimum occurs
    3. **Hopping**: Penalizing discovered optima to encourage exploration of new basins

    Designed for materials research where compositions must sum to 1 (simplex constraint).

    Parameters
    ----------
    objective : Callable
        Objective function that takes (X, bounds, acquisition_function) and returns
        (X_expected, X_actual, Y) tensors.
    bounds : torch.Tensor
        Search bounds as (2, d) tensor with [lower_bounds, upper_bounds].
    X_init_actual : torch.Tensor
        Initial observed locations (n, d).
    X_init_expected : torch.Tensor
        Initial expected/requested locations (n, d).
    Y_init : torch.Tensor
        Initial observed values (n, 1).
    proj_fn : Callable, optional
        Projection function to simplex. Default: proj_simplex.
    random_sampler : Callable, optional
        Random sampler for simplex. Default: random_simplex.
    random_direction_sampler : Callable, optional
        Sampler for zero-sum directions. Default: random_zero_sum_directions.
    max_zooms : int
        Maximum zoom levels per activation. Default: 3.
    max_iterations : int
        Maximum iterations per zoom level. Default: 10.
    top_m_points : int, optional
        Number of top points for determining zoom bounds. If None, auto-computed
        as max(d + 1, 4) where d is dimensionality. Default: None (auto).
    n_restarts : int
        Number of restarts for acquisition optimization. Default: 30.
    raw : int
        Number of raw samples for initial candidates. Default: 500.
    penalization_threshold : float
        Gradient threshold for penalty radius. Default: 1e-3.
    penalty_num_directions : int, optional
        Number of directions for penalty radius estimation. If None, auto-computed
        as 10 * d where d is dimensionality. Default: None (auto).
    penalty_max_radius : float
        Maximum penalty radius. Default: 0.3.
    penalty_radius_step : float, optional
        Step size for penalty radius search. If None, auto-computed based on
        input noise as max(3 * input_noise, 0.005). Default: None (auto).
    improvement_threshold_noise_mult : float
        Multiplier for output noise threshold. Default: 2.0.
    input_noise_threshold_mult : float
        Multiplier for input noise threshold. Default: 3.0.
    n_consecutive_no_improvements : int
        Consecutive iterations without improvement to trigger hopping. Default: 5.
    max_gp_points : int
        Maximum points for GP fitting. Default: 3000.
    repulsion_lambda : float, optional
        Lambda for repulsive acquisition. If None, auto-computed dynamically
        as 10 * median(|acquisition_values|) during optimization. Default: None (auto).
    device : str
        Torch device. Default: 'cuda'.
    dtype : torch.dtype
        Torch dtype. Default: torch.float64.
    run_uuid : str, optional
        4-digit UUID to resume from a saved run.
    checkpoint_dir : str, optional
        Base directory for checkpoints. If None, no checkpointing. Default: 'zombihop_checkpoints'.
    max_checkpoints : int, optional
        Maximum recent checkpoints to keep. If None or 0, no saving. Default: 50.
    verbose : bool
        Print progress information. Default: True.
    """

    def __init__(self,
                 objective,
                 bounds: torch.Tensor,
                 X_init_actual: torch.Tensor,
                 X_init_expected: torch.Tensor,
                 Y_init: torch.Tensor,
                 proj_fn: Optional[Callable] = None,
                 random_sampler: Optional[Callable] = None,
                 random_direction_sampler: Optional[Callable] = None,
                 max_zooms: int = 3,
                 max_iterations: int = 10,
                 top_m_points: Optional[int] = None,
                 n_restarts: int = 30,
                 raw: int = 500,
                 penalization_threshold: float = 1e-3,
                 penalty_num_directions: Optional[int] = None,
                 penalty_max_radius: float = 0.3,
                 penalty_radius_step: Optional[float] = None,
                 improvement_threshold_noise_mult: float = 2.0,
                 input_noise_threshold_mult: float = 3.0,
                 n_consecutive_no_improvements: int = 5,
                 max_gp_points: int = 3000,
                 repulsion_lambda: Optional[float] = None,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float64,
                 run_uuid: Optional[str] = None,
                 checkpoint_dir: Optional[str] = 'zombihop_checkpoints',
                 max_checkpoints: Optional[int] = 50,
                 verbose: bool = True):
        """Initialize ZoMBIHop optimizer."""
        # Computer parameters
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose

        # CUDA optimization
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if self.verbose:
                print(f"Initialized ZoMBIHop on CUDA device: {torch.cuda.get_device_name()}")
                print(f"Initial CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Store simplex utilities
        self.proj_fn = proj_fn if proj_fn is not None else proj_simplex
        self.random_sampler = random_sampler if random_sampler is not None else random_simplex
        self.random_direction_sampler = random_direction_sampler if random_direction_sampler is not None else random_zero_sum_directions

        self.objective = objective

        # Bounds and dimensionality
        self.d = bounds.shape[1]
        bounds = bounds.clone().to(device=self.device, dtype=self.dtype)
        assert bounds.shape == (2, self.d), "bounds must be a (2, d) torch tensor"

        # Auto-compute top_m_points if not provided: max(d + 1, 4)
        # Rationale: d+1 points define a simplex, minimum of 4 for stability
        if top_m_points is None:
            top_m_points = max(self.d + 1, 4)
            if self.verbose:
                print(f"Auto-computed top_m_points = {top_m_points} (based on d={self.d})")

        # ZoMBIHop parameters
        self.max_zooms = max_zooms
        self.max_iterations = max_iterations
        self.top_m_points = top_m_points
        self.max_gp_points = max_gp_points

        # Finding next point parameters
        self.n_restarts = n_restarts
        self.raw = raw

        # Penalization parameters
        self.penalization_threshold = penalization_threshold
        self.penalty_max_radius = penalty_max_radius
        
        # Auto-compute penalty_num_directions if not provided: 10 * d
        if penalty_num_directions is None:
            penalty_num_directions = 10 * self.d
            if self.verbose:
                print(f"Auto-computed penalty_num_directions = {penalty_num_directions} (based on d={self.d})")
        self.penalty_num_directions = penalty_num_directions
        
        # penalty_radius_step: None means auto-compute based on input noise at runtime
        # (input noise not available until we have data, so defer computation)
        self.penalty_radius_step = penalty_radius_step

        # Convergence parameters
        self.improvement_threshold_noise_mult = improvement_threshold_noise_mult
        self.input_noise_threshold_mult = input_noise_threshold_mult
        self.n_consecutive_no_improvements = n_consecutive_no_improvements

        # Acquisition parameters - repulsion_lambda=None means auto-compute dynamically
        self.repulsion_lambda = repulsion_lambda

        # Build config dataclass for saving
        self.config = ZoMBIHopConfig(
            max_zooms=max_zooms,
            max_iterations=max_iterations,
            top_m_points=top_m_points,  # May be None (auto-computed) or user-provided
            n_restarts=n_restarts,
            raw=raw,
            penalization_threshold=penalization_threshold,
            penalty_num_directions=penalty_num_directions,
            penalty_max_radius=penalty_max_radius,
            penalty_radius_step=penalty_radius_step,
            improvement_threshold_noise_mult=improvement_threshold_noise_mult,
            input_noise_threshold_mult=input_noise_threshold_mult,
            n_consecutive_no_improvements=n_consecutive_no_improvements,
            repulsion_lambda=repulsion_lambda,  # May be None (auto-computed) or user-provided
            device=str(self.device),
            dtype=str(self.dtype),
        )

        # Initialize data handler
        self.data_handler = DataHandler(
            directory=checkpoint_dir,
            run_uuid=run_uuid,
            max_saved_recent_checkpoints=max_checkpoints,
            device=str(self.device),
            dtype=self.dtype,
            config=self.config,
            d=self.d,
            top_m_points=top_m_points,
            max_gp_points=max_gp_points,
        )

        # Check if resuming from saved run
        if run_uuid is not None:
            if self.verbose:
                print(f"Resuming from saved run: {run_uuid}")
            activation, zoom, iteration, no_improvements = self.data_handler.load_state()
            if self.verbose:
                print(f"Loaded state: activation={activation}, zoom={zoom}, iteration={iteration}")
        else:
            if self.verbose:
                print(f"Starting new run with UUID: {self.data_handler.run_uuid}")
                if checkpoint_dir:
                    print(f"Checkpoint directory: {self.data_handler.run_dir}")

            # Initialize with provided data
            X_init_actual = X_init_actual.clone().to(device=self.device, dtype=self.dtype)
            X_init_expected = X_init_expected.clone().to(device=self.device, dtype=self.dtype)
            Y_init = Y_init.clone().to(device=self.device, dtype=self.dtype)

            assert X_init_actual.shape[1] == self.d, "X_init_actual must be a (n, d) torch tensor"
            assert X_init_expected.shape[1] == self.d, "X_init_expected must be a (n, d) torch tensor"
            assert Y_init.shape[1] == 1, "Y_init must be a (n, 1) torch tensor"
            assert X_init_actual.shape[0] == X_init_expected.shape[0] == Y_init.shape[0]

            self.data_handler.save_init(X_init_actual, X_init_expected, Y_init, bounds)

        # Store bounds reference
        self.bounds = self.data_handler.bounds

        # Initialize GP handler
        self.gp_handler = GPSimplex(
            data_handler=self.data_handler,
            proj_fn=self.proj_fn,
            random_sampler=self.random_sampler,
            num_restarts=self.n_restarts,
            raw_samples=self.raw,
            repulsion_lambda=self.repulsion_lambda,
            device=str(self.device),
            dtype=self.dtype,
        )

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _log_status(self, activation: int, zoom: int, iteration: int,
                    candidate: Optional[torch.Tensor], no_improvements: int):
        """Print current status."""
        if self.verbose:
            candidate_str = f"{candidate.cpu().numpy()}" if candidate is not None else "None"
            print(f"[A{activation+1}/Z{zoom+1}/I{iteration+1}] "
                  f"Candidate: {candidate_str} | "
                  f"No improvements: {no_improvements}")

    def _objective_wrapper(self, X: torch.Tensor, bounds: torch.Tensor, acquisition_function) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Call objective and update data handler.

        Returns
        -------
        tuple
            (unpenalized_X, unpenalized_Y) for points not in penalty regions.
        """
        assert X.shape == (self.d,)
        X_expected, X_actual, Y = self.objective(X, bounds, acquisition_function)

        X_expected = X_expected.to(device=self.device, dtype=self.dtype)
        X_actual = X_actual.to(device=self.device, dtype=self.dtype)
        Y = Y.to(device=self.device, dtype=self.dtype)

        assert isinstance(X_expected, torch.Tensor)
        assert isinstance(X_actual, torch.Tensor)
        assert isinstance(Y, torch.Tensor)

        assert X_expected.shape[1] == self.d
        assert X_actual.shape[1] == self.d
        assert Y.ndim == 1
        assert X_expected.shape[0] == X_actual.shape[0] == Y.shape[0]

        # Add to data handler and get penalty mask for new points
        penalty_mask = self.data_handler.add_all_points(X_actual, X_expected, Y.unsqueeze(1))

        unpenalized_Y = Y[penalty_mask]
        unpenalized_X = X_actual[penalty_mask]
        return unpenalized_X, unpenalized_Y

    def run(self, max_activations: int = 5, time_limit_hours: float = None):
        """
        Run ZoMBI-Hop optimization.

        Parameters
        ----------
        max_activations : int
            Maximum number of activations (hopping cycles).
        time_limit_hours : float, optional
            Time limit in hours.

        Returns
        -------
        tuple
            (needles_results, needles, needle_vals, X_all_actual, Y_all)
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self._log(f"Starting optimization. CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        start_time = time.time() if time_limit_hours is not None else None

        finished = False
        activation, zoom, iteration, no_improvements = self.data_handler.get_iteration_state()
        start_activation = activation

        while activation < max_activations and not finished:
            self._log(f"\n{'='*50}")
            self._log(f"ACTIVATION {activation+1}/{max_activations}")
            self._log(f"{'='*50}")

            if time_limit_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= time_limit_hours:
                    self._log(f"Time limit of {time_limit_hours} hours reached. Stopping.")
                    finished = True
                    self.data_handler.push_checkpoint(f"act{activation}_timeout", is_permanent=True)
                    break
                self._log(f"Elapsed time: {elapsed_hours:.2f} / {time_limit_hours:.2f} hours")

            if self.device.type == 'cuda' and activation > 0:
                torch.cuda.empty_cache()

            if activation > start_activation or zoom == 0:
                no_improvements = 0
            needle = None
            bounds = self.bounds.clone()
            activation_failed = False

            start_zoom = zoom if activation == start_activation else 0
            for zoom in range(start_zoom, self.max_zooms):
                self._log(f"\n--- Zoom {zoom+1}/{self.max_zooms} ---")
                self._log(f"Bounds: {bounds}")

                # Fit GP
                X, Y = self.data_handler.get_gp_data()
                self._log(f"GP data points: {X.shape[0]}")
                self.gp_handler.fit(X, Y)

                start_iteration = iteration if (activation == start_activation and zoom == start_zoom) else 0
                for iteration in range(start_iteration, self.max_iterations):
                    if time_limit_hours is not None:
                        elapsed_hours = (time.time() - start_time) / 3600.0
                        if elapsed_hours >= time_limit_hours:
                            self._log(f"Time limit reached during iteration.")
                            finished = True
                            self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_iter{iteration}_timeout", is_permanent=True)
                            break

                    # Get candidate
                    candidate = self.gp_handler.get_candidate(bounds, best_f=Y.max().item())

                    if candidate is None:
                        self._log("No valid candidate found (all in penalized regions)")
                        activation_failed = True
                        self._log_status(activation, zoom, iteration, None, no_improvements)
                        break

                    self._log_status(activation, zoom, iteration, candidate, no_improvements)

                    # Get previous best for comparison
                    prev_best_X, prev_best_Y, _ = self.data_handler.get_best_unpenalized()

                    # Sample the candidate
                    unpenalized_X, unpenalized_Y = self._objective_wrapper(
                        candidate, bounds, self.gp_handler.acq_fn
                    )

                    # Refit GP with new data
                    X, Y = self.data_handler.get_gp_data()
                    self.gp_handler.fit(X, Y)

                    if unpenalized_Y.shape[0] == 0:
                        self._log("No unpenalized Y values, breaking")
                        activation_failed = True
                        self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_iter{iteration}_failed", is_permanent=True)
                        break

                    # Get current best
                    curr_best_X, curr_best_Y, _ = self.data_handler.get_best_unpenalized()

                    # Calculate improvement thresholds
                    output_noise = self.gp_handler.get_output_noise()
                    input_noise = self.data_handler.get_normalized_input_noise()
                    output_improvement_threshold = self.improvement_threshold_noise_mult * output_noise
                    input_change_threshold = self.input_noise_threshold_mult * input_noise

                    # Check for improvement
                    input_distance = torch.norm(curr_best_X - prev_best_X)
                    if prev_best_Y is not None:
                        improvement = curr_best_Y.item() - prev_best_Y.item()
                        if improvement < output_improvement_threshold and input_distance < input_change_threshold:
                            no_improvements += 1
                        else:
                            no_improvements = 0
                    else:
                        no_improvements = 0

                    # Update state and checkpoint
                    self.data_handler.update_iteration_state(activation, zoom, iteration, no_improvements)
                    self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_iter{iteration}", is_permanent=False)

                    self._log(f"No improvements: {no_improvements} | "
                              f"Current max Y: {curr_best_Y.item():.4f} | "
                              f"Overall max: {self.data_handler.Y_all[self.data_handler.get_penalty_mask()].max().item():.4f}")

                    # Check if we've converged to a needle
                    if no_improvements >= self.n_consecutive_no_improvements:
                        needle_X, needle_Y, global_idx = self.data_handler.get_best_unpenalized()

                        self._log(f"\n*** Found needle at {needle_X.cpu().numpy()} with value {needle_Y.item():.4f} ***")

                        # Refit GP on full bounds for penalty radius determination
                        X, Y = self.data_handler.get_gp_data()
                        self.gp_handler.fit(X, Y)
                        self.gp_handler.create_acquisition(best_f=Y.max().item(), penalty_value=-1e6)

                        # Determine penalty radius
                        penalty_radius = self.gp_handler.determine_penalty_radius(
                            needle=needle_X,
                            penalization_threshold=self.penalization_threshold,
                            num_directions=self.penalty_num_directions,
                            max_radius=self.penalty_max_radius,
                            radius_step=self.penalty_radius_step,
                        )
                        self._log(f"Penalizing with radius {penalty_radius:.4f}")

                        # Add needle to data handler
                        self.data_handler.add_needle(
                            needle=needle_X,
                            needle_value=needle_Y.item(),
                            needle_penalty_radius=penalty_radius,
                            activation=activation,
                            zoom=zoom,
                            iteration=iteration,
                        )

                        self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_iter{iteration}_needle", is_permanent=True)
                        break

                if finished:
                    break

                if needle is not None or activation_failed:
                    # Check if too much area is penalized
                    test_samples = self.random_sampler(
                        self.raw, self.bounds[0], self.bounds[1],
                        device=str(self.device), torch_dtype=self.dtype
                    )
                    unpenalized_mask = self.data_handler.get_penalty_mask(test_samples)
                    penalized_percentage = (1 - unpenalized_mask.float().mean().item()) * 100

                    if penalized_percentage > 90:
                        self._log(f"Too much area penalized: {penalized_percentage:.2f}%. Ending.")
                        finished = True
                        self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_finished", is_permanent=True)
                    break

                if finished:
                    break
                if zoom < self.max_zooms - 1:
                    bounds = self.data_handler.determine_new_bounds()
                    self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_complete", is_permanent=True)

            activation += 1
            zoom = 0
            iteration = 0

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self._log(f"Optimization complete. Final CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        self.data_handler.push_checkpoint("final", is_permanent=True)
        self._log(f"\nOptimization complete. Run UUID: {self.data_handler.run_uuid}")
        self._log(f"Found {len(self.data_handler.needles_results)} needles")

        # Return results
        X_all_actual, _, Y_all = self.data_handler.get_all_points()
        return (
            self.data_handler.get_needle_results(),
            self.data_handler.get_needle_locations(),
            self.data_handler.needle_vals,
            X_all_actual,
            Y_all
        )

    # Keep static methods for backward compatibility, but they now delegate to simplex.py
    @staticmethod
    def proj_simplex(X):
        """Project points onto the simplex (differentiable)."""
        return proj_simplex(X)

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
        """Generate CFS samples from bounded simplex."""
        return random_simplex(num_samples, a, b, S, max_batch, debug, device, torch_dtype, **ignored)

    @staticmethod
    def random_zero_sum_directions(n: int, d: int, device='cuda') -> torch.Tensor:
        """Sample n vectors of dimension d with zero sum and unit norm."""
        return random_zero_sum_directions(n, d, device=device)
