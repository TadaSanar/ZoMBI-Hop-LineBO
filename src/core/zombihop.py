"""
ZoMBI-Hop: Zooming Multi-Basin Identification with Hopping
==========================================================

A novel Bayesian optimization algorithm for discovering multiple optima
in simplex-constrained spaces, designed for materials research applications.
"""

import torch
import time
from typing import Callable, Tuple, Optional, List, Any

from ..utils.simplex import (
    proj_simplex,
    random_simplex,
    random_zero_sum_directions,
)
from ..utils.datahandler import DataHandler
from ..utils.gp_simplex import GPSimplex


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

    Discovers multiple local optima in simplex-constrained spaces via:
    1. Zooming: iteratively narrowing the search space
    2. Basin Identification: detecting convergence to a local optimum
    3. Hopping: penalizing found optima to explore new basins

    All control variables are stored in self.data_handler and accessible
    as plain attributes (e.g. self.data_handler.max_zooms). At the end of
    every iteration, self.data_handler.take_snapshot() is called to save
    a complete checkpoint to disk.

    Parameters
    ----------
    objective : Callable
        Takes (X, bounds, acquisition_function), returns (X_expected, X_actual, Y).
    bounds : torch.Tensor
        (2, d) tensor: [lower_bounds, upper_bounds].
    X_init_actual : torch.Tensor
        Initial observed locations (n, d).
    X_init_expected : torch.Tensor
        Initial expected/requested locations (n, d).
    Y_init : torch.Tensor
        Initial observed values (n, 1).
    proj_fn, random_sampler, random_direction_sampler : Callable, optional
        Simplex utilities (defaults provided).
    max_zooms : int
        Max zoom levels per activation. Default: 3.
    max_iterations : int
        Max iterations per zoom level. Default: 10.
    top_m_points : int, optional
        Top points for zoom bounds. Auto-computed as max(d+1, 4) if None.
    n_restarts : int
        Acquisition optimization restarts. Default: 30.
    raw : int
        Raw samples for initial candidates. Default: 500.
    penalization_threshold : float
        Gradient threshold for penalty radius. Default: 1e-3.
    penalty_num_directions : int, optional
        Directions for penalty radius estimation. Auto-computed as 10*d if None.
    penalty_max_radius : float
        Max penalty radius. Default: 0.3.
    penalty_radius_step : float, optional
        Step size for radius search. Auto-computed from input noise if None.
    convergence_pi_threshold : float
        PI threshold for convergence. Default: 0.01.
    input_noise_threshold_mult : float
        Multiplier for input noise convergence check. Default: 2.0.
    output_noise_threshold_mult : float
        Multiplier for output noise convergence check. Default: 2.0.
    n_consecutive_converged : int
        Consecutive converged iterations before declaring needle. Default: 2.
    max_gp_points : int
        Max points for GP fitting. Default: 3000.
    repulsion_lambda : float, optional
        Repulsion lambda. Auto-computed dynamically if None.
    acquisition_type : str
        "ucb" or "ei". Default: "ucb".
    ucb_beta : float
        UCB exploration weight. Default: 0.1.
    nat_grad_step : float
        Natural-gradient ascent step on the probability simplex when maximizing
        acquisition. Default: 0.02.
    nat_grad_max_steps : int
        Max ascent steps per acquisition restart. Default: 50.
    device : str
        Torch device. Default: 'cuda'.
    dtype : torch.dtype
        Torch dtype. Default: torch.float64.
    run_uuid : str, optional
        4-character UUID to resume from a saved run.
    checkpoint_dir : str, optional
        Base directory for snapshots. Default: 'zombihop_checkpoints'.
    max_snapshots : int, optional
        Max snapshots to keep on disk. None = keep all. Default: None.
    verbose : bool
        Print progress. Default: True.
    needle_plot_points_ref : list, optional
        If provided, needle info is appended here on discovery (for live plots).
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
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float64,
                 run_uuid: Optional[str] = None,
                 checkpoint_dir: Optional[str] = 'zombihop_checkpoints',
                 num_iterations_saved: int = 50,
                 max_snapshots: Optional[int] = None,
                 verbose: bool = True,
                 needle_plot_points_ref: Optional[List[Any]] = None):
        """Initialize ZoMBIHop optimizer."""
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose
        self._needle_plot_points_ref = needle_plot_points_ref

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if self.verbose:
                print(f"Initialized ZoMBIHop on CUDA device: {torch.cuda.get_device_name()}")
                print(f"Initial CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Simplex utilities (functions; not checkpointable)
        self.proj_fn = proj_fn if proj_fn is not None else proj_simplex
        self.random_sampler = random_sampler if random_sampler is not None else random_simplex
        self.random_direction_sampler = (random_direction_sampler if random_direction_sampler is not None
                                         else random_zero_sum_directions)
        self.objective = objective

        # Auto-compute dimension-dependent params before passing to DataHandler
        d = bounds.shape[1]
        if top_m_points is None:
            top_m_points = max(d + 1, 4)
            if self.verbose:
                print(f"Auto-computed top_m_points = {top_m_points} (based on d={d})")
        if penalty_num_directions is None:
            penalty_num_directions = 10 * d
            if self.verbose:
                print(f"Auto-computed penalty_num_directions = {penalty_num_directions} (based on d={d})")

        effective_max_snapshots = max_snapshots if max_snapshots is not None else num_iterations_saved

        # --- DataHandler owns ALL control variables ---
        self.data_handler = DataHandler(
            max_zooms=max_zooms,
            max_iterations=max_iterations,
            top_m_points=top_m_points,
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
            max_gp_points=max_gp_points,
            repulsion_lambda=repulsion_lambda,
            acquisition_type=acquisition_type,
            ucb_beta=ucb_beta,
            nat_grad_step=nat_grad_step,
            nat_grad_max_steps=nat_grad_max_steps,
            directory=checkpoint_dir,
            run_uuid=run_uuid,
            max_snapshots=effective_max_snapshots,
            device=str(self.device),
            dtype=self.dtype,
            d=d,
        )

        # Resume from snapshot or start fresh
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
                    print(f"Snapshot directory: {self.data_handler.run_dir}")

            bounds = bounds.clone().to(device=self.device, dtype=self.dtype)
            X_init_actual = X_init_actual.clone().to(device=self.device, dtype=self.dtype)
            X_init_expected = X_init_expected.clone().to(device=self.device, dtype=self.dtype)
            Y_init = Y_init.clone().to(device=self.device, dtype=self.dtype)

            assert bounds.shape == (2, d), "bounds must be a (2, d) torch tensor"
            assert X_init_actual.shape[1] == d, "X_init_actual must be (n, d)"
            assert X_init_expected.shape[1] == d, "X_init_expected must be (n, d)"
            assert Y_init.shape[1] == 1, "Y_init must be (n, 1)"
            assert X_init_actual.shape[0] == X_init_expected.shape[0] == Y_init.shape[0]

            self.data_handler.save_init(X_init_actual, X_init_expected, Y_init, bounds)

        # self.bounds is a convenience alias; always kept in sync with data_handler.bounds
        self.bounds = self.data_handler.bounds

        # GP handler
        self.gp_handler = GPSimplex(
            data_handler=self.data_handler,
            proj_fn=self.proj_fn,
            random_sampler=self.random_sampler,
            num_restarts=self.data_handler.n_restarts,
            raw_samples=self.data_handler.raw,
            repulsion_lambda=self.data_handler.repulsion_lambda,
            acquisition_type=self.data_handler.acquisition_type,
            ucb_beta=self.data_handler.ucb_beta,
            nat_grad_step=self.data_handler.nat_grad_step,
            nat_grad_max_steps=self.data_handler.nat_grad_max_steps,
            device=str(self.device),
            dtype=self.dtype,
        )

    # --- Properties (expose data handler state) ---

    @property
    def run_uuid(self) -> str:
        return self.data_handler.run_uuid

    @property
    def current_activation(self) -> int:
        return self.data_handler.current_activation

    @property
    def current_zoom(self) -> int:
        return self.data_handler.current_zoom

    @property
    def current_iteration(self) -> int:
        return self.data_handler.current_iteration

    # Convenience shorthands so internal code stays readable
    @property
    def d(self) -> int:
        return self.data_handler.d

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _log_status(self, activation: int, zoom: int, iteration: int,
                    candidate: Optional[torch.Tensor], pi: Optional[float] = None):
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
        Check convergence to a local optimum. Returns (converged, pi, log_ei).

        Converge when:
        1. PI at candidate < convergence_pi_threshold
        2. Latest best Y improves by less than output_noise * output_noise_threshold_mult
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
        self.data_handler.log_ei_history.append(log_ei)
        pi_low = pi < self.data_handler.convergence_pi_threshold

        if prev_best_X is None or prev_best_Y is None:
            return False, pi, log_ei

        output_noise = self.gp_handler.get_output_noise()
        prev_y = prev_best_Y.item() if torch.is_tensor(prev_best_Y) else prev_best_Y
        improvement = latest_best_Y - prev_y
        input_distance = torch.norm(latest_best_X - prev_best_X).item()
        output_within_noise = improvement < (output_noise * self.data_handler.output_noise_threshold_mult)
        converged = pi_low and output_within_noise

        if converged and self.verbose:
            self._log(f"Converged: PI={pi:.4f}, improvement={improvement:.2e}, "
                      f"input_dist={input_distance:.2e}, logEI={log_ei:.2f}")
        return converged, pi, log_ei

    def _objective_wrapper(self, X: torch.Tensor, bounds: torch.Tensor, acquisition_function) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Call objective and update data handler.
        Returns (unpenalized_X, unpenalized_Y) — points not in any penalty region.
        """
        assert X.shape == (self.d,)
        X_expected, X_actual, Y = self.objective(X, bounds, acquisition_function)

        X_expected = X_expected.to(device=self.device, dtype=self.dtype)
        X_actual = X_actual.to(device=self.device, dtype=self.dtype)
        Y = Y.to(device=self.device, dtype=self.dtype)

        # Project actual measurements onto the simplex so off-simplex apparatus
        # noise doesn't corrupt stored data, distance calculations, or needle positions.
        X_actual = self.proj_fn(X_actual)

        assert X_expected.shape[1] == self.d
        assert X_actual.shape[1] == self.d
        assert Y.ndim == 1
        assert X_expected.shape[0] == X_actual.shape[0] == Y.shape[0]

        penalty_mask = self.data_handler.add_all_points(X_actual, X_expected, Y.unsqueeze(1))
        return X_actual[penalty_mask], Y[penalty_mask]

    def run(self, max_activations: int = 5, time_limit_hours: float = None):
        """
        Run ZoMBI-Hop optimization.

        Returns
        -------
        tuple
            (needles_results, needles, needle_vals, X_all_actual, Y_all)
        """
        dh = self.data_handler  # shorthand

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self._log(f"Starting optimization. CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        start_time = time.time() if time_limit_hours is not None else None

        finished = False
        activation, zoom, iteration, _ = dh.get_iteration_state()
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
                    dh.update_iteration_state(activation, zoom, iteration, dh.no_improvements)
                    dh.take_snapshot(f"act{activation}_timeout", permanent=True)
                    break
                self._log(f"Elapsed time: {elapsed_hours:.2f} / {time_limit_hours:.2f} hours")

            if self.device.type == 'cuda' and activation > 0:
                torch.cuda.empty_cache()

            needle = None
            bounds = self.bounds.clone()
            activation_failed = False

            start_zoom = zoom if activation == start_activation else 0

            for zoom in range(start_zoom, dh.max_zooms):
                self._log(f"\n--- Zoom {zoom+1}/{dh.max_zooms} ---")
                self._log(f"Bounds: {bounds}")

                X, Y = dh.get_gp_data()
                self._log(f"GP data points: {X.shape[0]}")
                self.gp_handler.fit(X, Y)

                start_iteration = iteration if (activation == start_activation and zoom == start_zoom) else 0
                consecutive_converged = 0

                for iteration in range(start_iteration, dh.max_iterations):
                    # Time limit check
                    if time_limit_hours is not None:
                        elapsed_hours = (time.time() - start_time) / 3600.0
                        if elapsed_hours >= time_limit_hours:
                            self._log(f"Time limit reached during iteration.")
                            finished = True
                            dh.update_iteration_state(activation, zoom, iteration, dh.no_improvements)
                            dh.take_snapshot(f"act{activation}_z{zoom}_i{iteration}_timeout", permanent=True)
                            break

                    candidate = self.gp_handler.get_candidate(bounds, best_f=Y.max().item())
                    if self.verbose and candidate is not None:
                        self._log(f"  [ZoMBIHop] GP suggested candidate: {candidate.cpu().numpy()}")

                    if candidate is None:
                        self._log("No valid candidate found (all in penalized regions)")
                        activation_failed = True
                        dh.update_iteration_state(activation, zoom, iteration, dh.no_improvements)
                        self._log_status(activation, zoom, iteration, None)
                        break

                    prev_best_X, prev_best_Y, _ = dh.get_best_unpenalized()

                    if self.verbose:
                        self._log(f"  [ZoMBIHop] Calling objective (LineBO samples lines through this candidate)...")

                    unpenalized_X, unpenalized_Y = self._objective_wrapper(
                        candidate, bounds, self.gp_handler.acq_fn
                    )
                    if self.verbose and unpenalized_X.shape[0] > 0:
                        self._log(f"  [ZoMBIHop] Objective returned {unpenalized_X.shape[0]} points, "
                                  f"Y in [{unpenalized_Y.min().item():.4f}, {unpenalized_Y.max().item():.4f}]")

                    X, Y = dh.get_gp_data()
                    self.gp_handler.fit(X, Y)

                    if unpenalized_Y.shape[0] == 0:
                        self._log("No unpenalized Y values, breaking — every point in this batch "
                                  "lies inside at least one needle penalty ball.")
                        activation_failed = True
                        dh.update_iteration_state(activation, zoom, iteration, dh.no_improvements)
                        dh.take_snapshot(f"act{activation}_z{zoom}_i{iteration}_failed", permanent=True)
                        break

                    curr_best_X, curr_best_Y, _ = dh.get_best_unpenalized()

                    converged, pi, log_ei = self._check_convergence_to_needle(
                        candidate, unpenalized_X, unpenalized_Y, prev_best_X, prev_best_Y
                    )
                    if converged:
                        consecutive_converged += 1
                    else:
                        consecutive_converged = 0

                    self._log_status(activation, zoom, iteration, candidate, pi=pi)
                    if consecutive_converged > 0:
                        self._log(f"Convergence count: {consecutive_converged}/{dh.n_consecutive_converged}")

                    self._log(f"Current max Y: {curr_best_Y.item():.4f} | "
                              f"Overall max: {dh.Y_all[dh.get_penalty_mask()].max().item():.4f}")

                    # --- Save snapshot at end of every iteration ---
                    dh.update_iteration_state(activation, zoom, iteration, 0)
                    dh.take_snapshot(f"act{activation}_z{zoom}_i{iteration}")

                    # --- Declare needle after N consecutive converged iterations ---
                    if consecutive_converged >= dh.n_consecutive_converged:
                        needle_X, needle_Y, global_idx = dh.get_best_unpenalized()
                        needle = needle_X

                        self._log(f"\n*** Found needle at {needle_X.cpu().numpy()} "
                                  f"with value {needle_Y.item():.4f} ***")

                        X, Y = dh.get_gp_data()
                        self.gp_handler.fit(X, Y)
                        self.gp_handler.create_acquisition(best_f=Y.max().item(), penalty_value=-1e6)

                        penalty_radius = self.gp_handler.determine_penalty_radius(
                            needle=needle_X,
                            penalization_threshold=dh.penalization_threshold,
                            num_directions=dh.penalty_num_directions,
                            max_radius=dh.penalty_max_radius,
                            radius_step=dh.penalty_radius_step,
                        )
                        self._log(f"Penalizing with radius {penalty_radius:.4f}")

                        dh.add_needle(
                            needle=needle_X,
                            needle_value=needle_Y.item(),
                            needle_penalty_radius=penalty_radius,
                            activation=activation,
                            zoom=zoom,
                            iteration=iteration,
                        )

                        if self._needle_plot_points_ref is not None:
                            center = dh.X_all_actual.mean(0)
                            distance = torch.norm(needle_X - center).item()
                            self._needle_plot_points_ref.append({
                                "sample_idx": global_idx + 1,
                                "y": needle_Y.item(),
                                "distance": distance,
                            })

                        dh.update_iteration_state(activation, zoom, iteration, dh.no_improvements)
                        dh.take_snapshot(f"act{activation}_z{zoom}_i{iteration}_needle", permanent=True)
                        break

                if finished:
                    break

                if needle is not None or activation_failed:
                    test_samples = self.random_sampler(
                        dh.raw, self.bounds[0], self.bounds[1],
                        device=str(self.device), torch_dtype=self.dtype
                    )
                    unpenalized_mask = dh.get_penalty_mask(test_samples)
                    penalized_pct = (1 - unpenalized_mask.float().mean().item()) * 100

                    if penalized_pct > 90:
                        if max_activations == float("inf"):
                            # Infinite run: zoom out to full simplex and continue
                            full_bounds = torch.zeros((2, self.d), device=self.device, dtype=self.dtype)
                            full_bounds[1] = 1.0
                            dh.bounds = full_bounds.clone().to(device=dh.device, dtype=dh.dtype)
                            self.bounds = dh.bounds
                            self._log(f"Too much area penalized: {penalized_pct:.2f}%. "
                                      f"Zooming out to full simplex.")
                            dh.update_iteration_state(activation, zoom, iteration, dh.no_improvements)
                            dh.take_snapshot(f"act{activation}_z{zoom}_zoomed_out", permanent=True)
                        else:
                            self._log(f"Too much area penalized: {penalized_pct:.2f}%. Ending.")
                            finished = True
                            dh.update_iteration_state(activation, zoom, iteration, dh.no_improvements)
                            dh.take_snapshot(f"act{activation}_z{zoom}_finished", permanent=True)
                    break

                if finished:
                    break

                if zoom < dh.max_zooms - 1:
                    bounds = dh.determine_new_bounds()
                    dh.update_iteration_state(activation, zoom, iteration, dh.no_improvements)
                    dh.take_snapshot(f"act{activation}_z{zoom}_complete")

            activation += 1
            zoom = 0
            iteration = 0
            dh.update_iteration_state(activation, zoom, iteration, 0)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self._log(f"Optimization complete. Final CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        dh.take_snapshot("final", permanent=True)
        self._log(f"\nOptimization complete. Run UUID: {dh.run_uuid}")
        self._log(f"Found {len(dh.needles_results)} needles")

        X_all_actual, _, Y_all = dh.get_all_points()
        return (
            dh.get_needle_results(),
            dh.get_needle_locations(),
            dh.needle_vals,
            X_all_actual,
            Y_all,
        )

    # --- Static methods: backward compatibility ---

    @staticmethod
    def proj_simplex(X):
        """Project points onto the simplex. X: (n, d) -> (n, d)."""
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
        """Generate CFS samples from bounded simplex. Returns (num_samples, d)."""
        return random_simplex(num_samples, a, b, S, max_batch, debug, device, torch_dtype, **ignored)

    @staticmethod
    def random_zero_sum_directions(n: int, d: int, device='cuda') -> torch.Tensor:
        """Sample n zero-sum unit vectors of dimension d. Returns (n, d)."""
        return random_zero_sum_directions(n, d, device=device)
