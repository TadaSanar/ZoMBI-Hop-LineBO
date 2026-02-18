"""
ZoMBI-Hop: Zooming Multi-Basin Identification with Hopping
==========================================================

A novel Bayesian optimization algorithm for discovering multiple optima
in simplex-constrained spaces, designed for materials research applications.
"""

# PyTorch and standard library
import torch
import time
from typing import Callable, Tuple, Optional, List, Any

# Simplex utilities: projection, random sampling, zero-sum directions
from ..utils.simplex import (
    proj_simplex,
    random_simplex,
    random_zero_sum_directions,
)
from ..utils.datahandler import DataHandler
from ..utils.gp_simplex import GPSimplex
from ..utils.dataclasses import ZoMBIHopConfig


# --- CUDA optimization settings (when CUDA is available) ---
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)


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
    convergence_pi_threshold : float
        Probability of Improvement threshold; converge when PI < this. Default: 0.01.
    input_noise_threshold_mult : float
        Multiplier for input noise; converge when best X from batch is within this * input_noise of prev best. Default: 2.0.
    output_noise_threshold_mult : float
        Multiplier for output noise; converge when best Y from batch improves by less than this * output_noise. Default: 2.0.
    n_consecutive_converged : int
        Require this many consecutive iterations where convergence criteria are met before declaring a needle. Default: 2.
    max_gp_points : int
        Maximum points for GP fitting. Default: 3000.
    repulsion_lambda : float, optional
        Lambda for repulsive acquisition. If None, auto-computed dynamically
        as 10 * median(|acquisition_values|) during optimization. Default: None (auto).
    acquisition_type : str
        Base acquisition: "ucb" (Upper Confidence Bound) or "ei" (Expected Improvement).
        Both are wrapped with needle repulsion. Default: "ucb".
    ucb_beta : float
        Exploration weight for UCB (mean + beta * std). Only used when acquisition_type=="ucb". Default: 0.1.
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
                 bounds: torch.Tensor,           # (2, d): [lower_bounds, upper_bounds]
                 X_init_actual: torch.Tensor,   # (n, d): initial observed locations
                 X_init_expected: torch.Tensor, # (n, d): initial requested locations
                 Y_init: torch.Tensor,          # (n, 1): initial observed values
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
                 convergence_pi_threshold: float = 0.01,
                 input_noise_threshold_mult: float = 2.0,
                 output_noise_threshold_mult: float = 2.0,
                 n_consecutive_converged: int = 2,
                 max_gp_points: int = 3000,
                 repulsion_lambda: Optional[float] = None,
                 acquisition_type: str = "ucb",
                 ucb_beta: float = 0.1,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float64,
                 run_uuid: Optional[str] = None,
                 checkpoint_dir: Optional[str] = 'zombihop_checkpoints',
                 max_checkpoints: Optional[int] = 50,
                 verbose: bool = True,
                 needle_plot_points_ref: Optional[List[Any]] = None):
        """Initialize ZoMBIHop optimizer. If needle_plot_points_ref is provided, append {sample_idx, y, distance} when a needle is found (for live plot stars)."""
        # --- Compute/device parameters ---
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose
        self._needle_plot_points_ref = needle_plot_points_ref

        # --- CUDA optimization (clear cache, optional print) ---
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

        # --- Bounds and dimensionality ---
        # bounds: (2, d) — row0 = lower bounds, row1 = upper bounds
        self.d = bounds.shape[1]
        bounds = bounds.clone().to(device=self.device, dtype=self.dtype)
        assert bounds.shape == (2, self.d), "bounds must be a (2, d) torch tensor"

        # --- Auto-compute top_m_points if not provided: max(d + 1, 4) ---
        # Rationale: d+1 points define a simplex, minimum of 4 for stability
        if top_m_points is None:
            top_m_points = max(self.d + 1, 4)
            if self.verbose:
                print(f"Auto-computed top_m_points = {top_m_points} (based on d={self.d})")

        # --- ZoMBIHop parameters (zoom levels, iterations, GP size) ---
        self.max_zooms = max_zooms
        self.max_iterations = max_iterations
        self.top_m_points = top_m_points
        self.max_gp_points = max_gp_points

        # --- Finding next point parameters (acquisition optimization) ---
        self.n_restarts = n_restarts
        self.raw = raw

        # --- Penalization parameters (threshold, radius, directions) ---
        self.penalization_threshold = penalization_threshold
        self.penalty_max_radius = penalty_max_radius

        # --- Auto-compute penalty_num_directions if not provided: 10 * d ---
        if penalty_num_directions is None:
            penalty_num_directions = 10 * self.d
            if self.verbose:
                print(f"Auto-computed penalty_num_directions = {penalty_num_directions} (based on d={self.d})")
        self.penalty_num_directions = penalty_num_directions

        # penalty_radius_step: None = auto-compute from input noise at runtime
        self.penalty_radius_step = penalty_radius_step

        # --- Convergence parameters (PI + input/output noise thresholds) ---
        self.convergence_pi_threshold = convergence_pi_threshold
        self.input_noise_threshold_mult = input_noise_threshold_mult
        self.output_noise_threshold_mult = output_noise_threshold_mult
        self.n_consecutive_converged = n_consecutive_converged
        self.log_ei_history: List[float] = []

        # --- Acquisition: repulsion_lambda=None => auto-compute dynamically ---
        self.repulsion_lambda = repulsion_lambda
        self.acquisition_type = acquisition_type
        self.ucb_beta = ucb_beta

        # --- Build config dataclass for checkpointing ---
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
            convergence_pi_threshold=convergence_pi_threshold,
            input_noise_threshold_mult=input_noise_threshold_mult,
            output_noise_threshold_mult=output_noise_threshold_mult,
            n_consecutive_converged=n_consecutive_converged,
            repulsion_lambda=repulsion_lambda,  # May be None (auto-computed) or user-provided
            acquisition_type=acquisition_type,
            ucb_beta=ucb_beta,
            device=str(self.device),
            dtype=str(self.dtype),
        )

        # --- Initialize data handler (checkpoints, state, GP data) ---
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

        # --- Resume from checkpoint or start fresh ---
        if run_uuid is not None:
            if self.verbose:
                print(f"Resuming from saved run: {run_uuid}")
            activation, zoom, iteration, _ = self.data_handler.load_state()
            if self.verbose:
                print(f"Loaded state: activation={activation}, zoom={zoom}, iteration={iteration}")
        else:
            if self.verbose:
                print(f"Starting new run with UUID: {self.data_handler.run_uuid}")
                if checkpoint_dir:
                    print(f"Checkpoint directory: {self.data_handler.run_dir}")

            # --- Initialize with provided data (move to device, validate shapes) ---
            # X_init_actual: (n, d), X_init_expected: (n, d), Y_init: (n, 1)
            X_init_actual = X_init_actual.clone().to(device=self.device, dtype=self.dtype)
            X_init_expected = X_init_expected.clone().to(device=self.device, dtype=self.dtype)
            Y_init = Y_init.clone().to(device=self.device, dtype=self.dtype)

            assert X_init_actual.shape[1] == self.d, "X_init_actual must be (n, d)"
            assert X_init_expected.shape[1] == self.d, "X_init_expected must be (n, d)"
            assert Y_init.shape[1] == 1, "Y_init must be (n, 1)"
            assert X_init_actual.shape[0] == X_init_expected.shape[0] == Y_init.shape[0]

            self.data_handler.save_init(X_init_actual, X_init_expected, Y_init, bounds)

        # --- Bounds reference (from data_handler; shape (2, d)) ---
        self.bounds = self.data_handler.bounds

        # --- Initialize GP handler (acquisition, fitting, penalty radius) ---
        self.gp_handler = GPSimplex(
            data_handler=self.data_handler,
            proj_fn=self.proj_fn,
            random_sampler=self.random_sampler,
            num_restarts=self.n_restarts,
            raw_samples=self.raw,
            repulsion_lambda=self.repulsion_lambda,
            acquisition_type=self.acquisition_type,
            ucb_beta=self.ucb_beta,
            device=str(self.device),
            dtype=self.dtype,
        )

    # --- Properties: expose data handler state ---
    @property
    def run_uuid(self) -> str:
        """Run UUID (from data handler)."""
        return self.data_handler.run_uuid

    @property
    def current_activation(self) -> int:
        """Current activation index (from data handler)."""
        return self.data_handler.current_activation

    @property
    def current_zoom(self) -> int:
        """Current zoom level (from data handler)."""
        return self.data_handler.current_zoom

    @property
    def current_iteration(self) -> int:
        """Current iteration (from data handler)."""
        return self.data_handler.current_iteration

    def _log(self, message: str):
        """Print message if verbose is True."""
        if self.verbose:
            print(message)

    def _log_status(self, activation: int, zoom: int, iteration: int,
                    candidate: Optional[torch.Tensor], pi: Optional[float] = None):
        """Print current status. candidate: (d,) if not None. pi: Probability of Improvement for logging."""
        if self.verbose:
            candidate_str = f"{candidate.cpu().numpy()}" if candidate is not None else "None"
            extra = f" | PI={pi:.4f}" if pi is not None else ""
            print(f"[A{activation+1}/Z{zoom+1}/I{iteration+1}] Candidate: {candidate_str}{extra}")

    def _check_convergence_to_needle(
        self,
        candidate: torch.Tensor,
        unpenalized_X: torch.Tensor,
        unpenalized_Y: torch.Tensor,
        prev_best_X: Optional[torch.Tensor],
        prev_best_Y: Optional[torch.Tensor],
    ) -> Tuple[bool, float, float]:
        """
        Check if we have converged to a local optimum (needle).

        Uses the last unpenalized batch from the objective: latest best = argmax Y in that batch,
        and the corresponding X. Converge when:
        1. PI at candidate < convergence_pi_threshold
        2. Latest best Y improves by less than output_noise * output_noise_threshold_mult over prev best Y
        # 3. (commented out) Latest best X is within input_noise * input_noise_threshold_mult of prev best X

        Returns
        -------
        tuple
            (converged, pi, log_ei) for logging.
        """
        if unpenalized_X.shape[0] == 0:
            return False, 0.0, float('-inf')
        idx = unpenalized_Y.argmax().item()
        latest_best_X = unpenalized_X[idx : idx + 1].squeeze(0)
        latest_best_Y = unpenalized_Y[idx].item()

        X, Y = self.data_handler.get_gp_data()
        best_f = Y.max().item()
        pi = 0.0
        log_ei = float('-inf')
        try:
            pi = self.gp_handler.probability_of_improvement(candidate, best_f)
            log_ei = self.gp_handler.compute_log_ei_at_point(candidate, best_f)
        except Exception:
            pass
        self.log_ei_history.append(log_ei)
        pi_low = pi < self.convergence_pi_threshold

        if prev_best_X is None or prev_best_Y is None:
            converged = False
            improvement = 0.0
            input_distance = 0.0
        else:
            output_noise = self.gp_handler.get_output_noise()
            # input_noise = self.data_handler.get_normalized_input_noise()
            prev_y = prev_best_Y.item() if torch.is_tensor(prev_best_Y) else prev_best_Y
            improvement = latest_best_Y - prev_y
            input_distance = torch.norm(latest_best_X - prev_best_X).item()
            output_within_noise = improvement < (output_noise * self.output_noise_threshold_mult)
            # input_within_noise = input_distance < (input_noise * self.input_noise_threshold_mult)
            converged = pi_low and output_within_noise  # and input_within_noise  # input noise check commented out

        if converged and self.verbose:
            self._log(f"Converged: PI={pi:.4f}, improvement={improvement:.2e}, input_dist={input_distance:.2e}, logEI={log_ei:.2f}")
        return converged, pi, log_ei

    def _objective_wrapper(self, X: torch.Tensor, bounds: torch.Tensor, acquisition_function) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Call objective and update data handler.

        Parameters
        ----------
        X : torch.Tensor
            Candidate point, shape (d,).
        bounds : torch.Tensor
            Search bounds, shape (2, d).
        acquisition_function : callable
            Acquisition function used by objective.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            unpenalized_X: (n_unpen, d), unpenalized_Y: (n_unpen,) — points not in penalty regions.
        """
        # X: (d,) — single candidate
        assert X.shape == (self.d,)
        # Objective returns: X_expected (n, d), X_actual (n, d), Y (n,) — often n=1
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

        # Add to data handler; penalty_mask: (n,) bool — True where point is not in penalty region
        penalty_mask = self.data_handler.add_all_points(X_actual, X_expected, Y.unsqueeze(1))

        # unpenalized_Y: (n_unpen,), unpenalized_X: (n_unpen, d)
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
            needles_results: list of needle result dicts;
            needles: tensor (n_needles, d) — needle locations;
            needle_vals: tensor (n_needles,) or list — needle values;
            X_all_actual: (n_total, d) — all evaluated points;
            Y_all: (n_total, 1) — all observed values.
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self._log(f"Starting optimization. CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        start_time = time.time() if time_limit_hours is not None else None

        finished = False
        activation, zoom, iteration, _ = self.data_handler.get_iteration_state()
        start_activation = activation

        # --- Main loop: one activation = one “hopping” cycle (zoom → iterate → maybe find needle) ---
        while activation < max_activations and not finished:
            self._log(f"\n{'='*50}")
            self._log(f"ACTIVATION {activation+1}/{max_activations}")
            self._log(f"{'='*50}")

            # --- Time limit check (optional) ---
            if time_limit_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= time_limit_hours:
                    self._log(f"Time limit of {time_limit_hours} hours reached. Stopping.")
                    finished = True
                    self.data_handler.update_iteration_state(activation, zoom, iteration, self.data_handler.no_improvements)
                    self.data_handler.push_checkpoint(f"act{activation}_timeout", is_permanent=True)
                    break
                self._log(f"Elapsed time: {elapsed_hours:.2f} / {time_limit_hours:.2f} hours")

            if self.device.type == 'cuda' and activation > 0:
                torch.cuda.empty_cache()

            needle = None
            # bounds: (2, d) — current zoom bounds
            bounds = self.bounds.clone()
            activation_failed = False

            start_zoom = zoom if activation == start_activation else 0
            # --- Zoom loop: narrow search region each zoom level ---
            for zoom in range(start_zoom, self.max_zooms):
                self._log(f"\n--- Zoom {zoom+1}/{self.max_zooms} ---")
                self._log(f"Bounds: {bounds}")

                # X: (n_gp, d), Y: (n_gp, 1) — data used for GP fit
                X, Y = self.data_handler.get_gp_data()
                self._log(f"GP data points: {X.shape[0]}")
                self.gp_handler.fit(X, Y)

                start_iteration = iteration if (activation == start_activation and zoom == start_zoom) else 0
                consecutive_converged = 0  # Require N consecutive converged before declaring needle
                # --- Iteration loop: propose candidate, evaluate, check convergence ---
                for iteration in range(start_iteration, self.max_iterations):
                    # --- Optional time limit per iteration ---
                    if time_limit_hours is not None:
                        elapsed_hours = (time.time() - start_time) / 3600.0
                        if elapsed_hours >= time_limit_hours:
                            self._log(f"Time limit reached during iteration.")
                            finished = True
                            self.data_handler.update_iteration_state(activation, zoom, iteration, self.data_handler.no_improvements)
                            self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_iter{iteration}_timeout", is_permanent=True)
                            break

                    # candidate: (d,) or None
                    candidate = self.gp_handler.get_candidate(bounds, best_f=Y.max().item())
                    if self.verbose and candidate is not None:
                        self._log(f"  [ZoMBIHop] GP suggested candidate (acquisition argmax): {candidate.cpu().numpy()}")

                    if candidate is None:
                        self._log("No valid candidate found (all in penalized regions)")
                        activation_failed = True
                        self.data_handler.update_iteration_state(activation, zoom, iteration, self.data_handler.no_improvements)
                        self._log_status(activation, zoom, iteration, None)
                        break

                    # Previous best (before this batch) for convergence comparison
                    prev_best_X, prev_best_Y, _ = self.data_handler.get_best_unpenalized()

                    if self.verbose:
                        self._log(f"  [ZoMBIHop] Calling objective (LineBO samples lines through this candidate)...")

                    # Evaluate candidate; unpenalized_X: (n_unpen, d), unpenalized_Y: (n_unpen,)
                    unpenalized_X, unpenalized_Y = self._objective_wrapper(
                        candidate, bounds, self.gp_handler.acq_fn
                    )
                    if self.verbose and unpenalized_X.shape[0] > 0:
                        self._log(f"  [ZoMBIHop] Objective returned {unpenalized_X.shape[0]} points, Y in [{unpenalized_Y.min().item():.4f}, {unpenalized_Y.max().item():.4f}]")

                    # Refit GP; X: (n_gp, d), Y: (n_gp, 1)
                    X, Y = self.data_handler.get_gp_data()
                    self.gp_handler.fit(X, Y)

                    if unpenalized_Y.shape[0] == 0:
                        self._log(
                            "No unpenalized Y values, breaking — every point in this batch lies inside "
                            "at least one needle penalty ball (no usable points for convergence check)."
                        )
                        activation_failed = True
                        self.data_handler.update_iteration_state(activation, zoom, iteration, self.data_handler.no_improvements)
                        self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_iter{iteration}_failed", is_permanent=True)
                        break

                    # curr_best_X: (d,), curr_best_Y: (1,) scalar tensor
                    curr_best_X, curr_best_Y, _ = self.data_handler.get_best_unpenalized()

                    # Convergence: PI + stagnation (after refit); “no improvement” detection
                    converged, pi, log_ei = self._check_convergence_to_needle(
                        candidate, unpenalized_X, unpenalized_Y, prev_best_X, prev_best_Y
                    )
                    if converged:
                        consecutive_converged += 1
                    else:
                        consecutive_converged = 0
                    self._log_status(activation, zoom, iteration, candidate, pi=pi)
                    if consecutive_converged > 0:
                        self._log(f"Convergence count: {consecutive_converged}/{self.n_consecutive_converged}")
                    self.data_handler.update_iteration_state(activation, zoom, iteration, 0)
                    self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_iter{iteration}", is_permanent=False)

                    self._log(f"Current max Y: {curr_best_Y.item():.4f} | "
                              f"Overall max: {self.data_handler.Y_all[self.data_handler.get_penalty_mask()].max().item():.4f}")

                    # Declare needle only after N consecutive converged iterations
                    if consecutive_converged >= self.n_consecutive_converged:
                        needle_X, needle_Y, global_idx = self.data_handler.get_best_unpenalized()
                        needle = needle_X  # so we break out of zoom loop and go to next activation

                        self._log(f"\n*** Found needle at {needle_X.cpu().numpy()} with value {needle_Y.item():.4f} ***")

                        # Refit GP on full data for penalty radius; X: (n_gp, d), Y: (n_gp, 1)
                        X, Y = self.data_handler.get_gp_data()
                        self.gp_handler.fit(X, Y)
                        self.gp_handler.create_acquisition(best_f=Y.max().item(), penalty_value=-1e6)

                        # penalty_radius: scalar — radius around needle to penalize
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
                        if self._needle_plot_points_ref is not None:
                            center = self.data_handler.X_all_actual.mean(0)
                            distance = torch.norm(needle_X - center).item()
                            self._needle_plot_points_ref.append({
                                "sample_idx": global_idx + 1,
                                "y": needle_Y.item(),
                                "distance": distance,
                            })

                        self.data_handler.update_iteration_state(activation, zoom, iteration, self.data_handler.no_improvements)
                        self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_iter{iteration}_needle", is_permanent=True)
                        break

                if finished:
                    break

                if needle is not None or activation_failed:
                    # Check fraction of current bounds that is penalized
                    test_samples = self.random_sampler(
                        self.raw, self.bounds[0], self.bounds[1],
                        device=str(self.device), torch_dtype=self.dtype
                    )
                    unpenalized_mask = self.data_handler.get_penalty_mask(test_samples)
                    penalized_percentage = (1 - unpenalized_mask.float().mean().item()) * 100

                    if penalized_percentage > 90:
                        if max_activations == float("inf"):
                            # Infinite run: zoom out to full simplex and continue
                            full_bounds = torch.zeros((2, self.d), device=self.device, dtype=self.dtype)
                            full_bounds[0] = 0.0
                            full_bounds[1] = 1.0
                            self.data_handler.bounds = full_bounds.clone().to(device=self.data_handler.device, dtype=self.data_handler.dtype)
                            self.bounds = self.data_handler.bounds
                            self._log(f"Too much area penalized: {penalized_percentage:.2f}%. Zooming out to full simplex.")
                            self.data_handler.update_iteration_state(activation, zoom, iteration, self.data_handler.no_improvements)
                            self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_zoomed_out", is_permanent=True)
                        else:
                            # Finite run: stop
                            self._log(f"Too much area penalized: {penalized_percentage:.2f}%. Ending.")
                            finished = True
                            self.data_handler.update_iteration_state(activation, zoom, iteration, self.data_handler.no_improvements)
                            self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_finished", is_permanent=True)
                    break

                if finished:
                    break
                # Zoom in: new bounds (2, d) around top_m best points
                if zoom < self.max_zooms - 1:
                    bounds = self.data_handler.determine_new_bounds()
                    self.data_handler.bounds = bounds.clone().to(device=self.data_handler.device, dtype=self.data_handler.dtype)
                    self.bounds = self.data_handler.bounds
                    self.data_handler.update_iteration_state(activation, zoom, iteration, self.data_handler.no_improvements)
                    self.data_handler.push_checkpoint(f"act{activation}_zoom{zoom}_complete", is_permanent=True)

            activation += 1
            zoom = 0
            iteration = 0
            self.data_handler.update_iteration_state(activation, zoom, iteration, 0)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self._log(f"Optimization complete. Final CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        self.data_handler.push_checkpoint("final", is_permanent=True)
        self._log(f"\nOptimization complete. Run UUID: {self.data_handler.run_uuid}")
        self._log(f"Found {len(self.data_handler.needles_results)} needles")

        # X_all_actual: (n_total, d), Y_all: (n_total, 1); return needle results and all data
        X_all_actual, _, Y_all = self.data_handler.get_all_points()
        return (
            self.data_handler.get_needle_results(),
            self.data_handler.get_needle_locations(),
            self.data_handler.needle_vals,
            X_all_actual,
            Y_all
        )

    # --- Static methods: backward compatibility; delegate to simplex utils ---
    @staticmethod
    def proj_simplex(X):
        """Project points onto the simplex (differentiable). X: (n, d) -> (n, d)."""
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
        """Generate CFS samples from bounded simplex. a, b: (d,); returns (num_samples, d)."""
        return random_simplex(num_samples, a, b, S, max_batch, debug, device, torch_dtype, **ignored)

    @staticmethod
    def random_zero_sum_directions(n: int, d: int, device='cuda') -> torch.Tensor:
        """Sample n vectors of dimension d with zero sum and unit norm. Returns (n, d)."""
        return random_zero_sum_directions(n, d, device=device)
