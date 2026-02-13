"""
GP Simplex Handler
==================

Handles Gaussian Process fitting and candidate selection for simplex-constrained
optimization. Uses repulsive acquisition function for smooth gradient-based exploration.
"""

import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.autograd import grad
from typing import Optional, Tuple, Callable

from .simplex import proj_simplex, random_simplex, random_zero_sum_directions
from .datahandler import DataHandler

class RepulsiveAcquisition(nn.Module):
    """
    Acquisition function with smooth distance-based repulsion.

    Instead of hard masking, applies a smooth penalty that grows quadratically
    as points get closer to needle centers. This provides gradients that
    push the optimizer away from penalized regions.

    Parameters
    ----------
    base : nn.Module
        Base acquisition function (e.g., LogExpectedImprovement).
    proj_fn : Callable
        Projection function to simplex.
    needles : torch.Tensor
        Center points of penalty regions (num_needles, d).
    penalty_radii : torch.Tensor
        Radius for each needle (num_needles, 1) or (num_needles,).
    repulsion_lambda : float
        Strength of repulsion penalty. Should be large (e.g., 1000)
        since LogEI values are around 100. Default: 1000.0.
    """

    def __init__(
        self,
        base: nn.Module,
        proj_fn: Callable,
        needles: torch.Tensor,
        penalty_radii: torch.Tensor,
        repulsion_lambda: float = 1000.0,
    ):
        super().__init__()
        self.base = base
        self.proj_fn = proj_fn
        self.needles = needles  # (M, d)
        self.penalty_radii = penalty_radii.view(-1)  # (M,)
        self.repulsion_lambda = repulsion_lambda

    def forward(self, Xq: torch.Tensor) -> torch.Tensor:
        """
        Evaluate acquisition with smooth repulsion.

        Parameters
        ----------
        Xq : torch.Tensor
            Query points (n, q, d) or (n, d).

        Returns
        -------
        torch.Tensor
            Acquisition values with repulsion penalty.
        """
        X_proj = self.proj_fn(Xq)

        # Get base acquisition value
        base_acq = self.base(X_proj)  # (n,) or (n, q)

        # If no needles, return base acquisition
        if self.needles.shape[0] == 0:
            return base_acq

        # Reshape for distance computation
        original_shape = X_proj.shape
        if X_proj.ndim == 3:
            n, q, d = X_proj.shape
            X_flat = X_proj.reshape(-1, d)  # (n*q, d)
        else:
            X_flat = X_proj.reshape(-1, X_proj.shape[-1])  # (B, d)

        # Compute distances to each needle
        # X_flat: (B, d), needles: (M, d)
        diffs = X_flat.unsqueeze(1) - self.needles.unsqueeze(0)  # (B, M, d)
        distances = torch.norm(diffs, dim=-1)  # (B, M)

        # Find minimum distance to any needle (per-needle radius)
        # violation = max(0, radius - distance) for each needle
        violations = torch.clamp(self.penalty_radii.unsqueeze(0) - distances, min=0.0)  # (B, M)

        # Sum of squared violations (smooth, differentiable)
        total_violation = (violations ** 2).sum(dim=1)  # (B,)

        # Penalty term (negative, reduces acquisition value inside radii)
        penalty = -self.repulsion_lambda * total_violation

        # Reshape penalty to match base_acq shape
        penalty = penalty.view(base_acq.shape)

        return base_acq + penalty


class GPSimplex:
    """
    Gaussian Process handler for simplex-constrained optimization.

    Manages GP fitting, acquisition function creation, and candidate selection.

    Parameters
    ----------
    data_handler : DataHandler
        Data handler for accessing points and penalty info.
    proj_fn : Callable, optional
        Projection function to simplex. Default: proj_simplex.
    random_sampler : Callable, optional
        Random sampler for simplex. Default: random_simplex.
    num_restarts : int
        Number of restarts for acquisition optimization.
    raw_samples : int
        Number of raw samples for initial candidates.
    repulsion_lambda : float, optional
        Lambda for repulsive acquisition. If None, auto-computed dynamically
        as 10 * median(|acquisition_values|) when creating acquisition function.
        Default: None (auto).
    device : str
        Torch device.
    dtype : torch.dtype
        Data type.
    """

    def __init__(
        self,
        data_handler: DataHandler,
        proj_fn: Optional[Callable] = None,
        random_sampler: Optional[Callable] = None,
        num_restarts: int = 30,
        raw_samples: int = 500,
        repulsion_lambda: Optional[float] = None,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64,
    ):
        self.data_handler = data_handler
        self.proj_fn = proj_fn if proj_fn is not None else proj_simplex
        self.random_sampler = random_sampler if random_sampler is not None else random_simplex
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.repulsion_lambda = repulsion_lambda  # None means auto-compute
        self.device = torch.device(device)
        self.dtype = dtype

        self.gp: Optional[SingleTaskGP] = None
        self.mll = None
        self.acq_fn = None
        self._last_computed_lambda = None  # Track auto-computed lambda for logging

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Fit GP to data.

        Parameters
        ----------
        X : torch.Tensor
            Training inputs (n, d).
        Y : torch.Tensor
            Training outputs (n, 1).
        """
        X = X.to(device=self.device, dtype=self.dtype)
        Y = Y.to(device=self.device, dtype=self.dtype)

        self.gp = SingleTaskGP(X, Y)
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(self.mll)

    def fit_from_data_handler(self):
        """Fit GP using data from the data handler."""
        X, Y = self.data_handler.get_gp_data()
        self.fit(X, Y)

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the GP.

        Parameters
        ----------
        X : torch.Tensor
            Query points (n, d).

        Returns
        -------
        tuple
            (mean, variance) tensors.
        """
        if self.gp is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        X = X.to(device=self.device, dtype=self.dtype)
        posterior = self.gp.posterior(X)
        return posterior.mean, posterior.variance

    def get_output_noise(self) -> float:
        """Get average output noise from the GP."""
        if self.gp is None:
            return 0.0
        return self.gp.likelihood.noise_covar.noise.mean().item()

    def probability_of_improvement(self, x: torch.Tensor, best_f: float) -> float:
        """
        P(f(x) > best_f) under the GP posterior at x.

        Parameters
        ----------
        x : torch.Tensor
            Query point (d,) or (1, d).
        best_f : float
            Current best observed value.

        Returns
        -------
        float
            Probability of improvement, in [0, 1].
        """
        if self.gp is None:
            return 0.0
        x_2d = x.unsqueeze(0) if x.dim() == 1 else x
        with torch.no_grad():
            posterior = self.gp.posterior(x_2d)
            mu = posterior.mean.squeeze().item()
            var = posterior.variance.squeeze().item()
        sigma = max(var ** 0.5, 1e-9)
        z = (mu - best_f) / sigma
        return torch.distributions.Normal(0.0, 1.0).cdf(torch.tensor(z, device=self.device)).item()

    def compute_log_ei_at_point(self, x: torch.Tensor, best_f: float) -> float:
        """
        Log Expected Improvement at x for maximization (best_f = current best).

        Parameters
        ----------
        x : torch.Tensor
            Query point (d,) or (1, d).
        best_f : float
            Current best observed value.

        Returns
        -------
        float
            log(EI(x)).
        """
        if self.gp is None:
            return float('-inf')
        x_3d = x.unsqueeze(0).unsqueeze(0) if x.dim() == 1 else x.reshape(1, 1, -1)
        base_acq = LogExpectedImprovement(self.gp, best_f=best_f)
        with torch.no_grad():
            val = base_acq(x_3d).squeeze().item()
        return val

    def create_acquisition(
        self,
        best_f: Optional[float] = None,
        penalty_value: Optional[float] = None,
    ) -> nn.Module:
        """
        Create acquisition function.

        Parameters
        ----------
        best_f : float, optional
            Best function value so far. If None, computed from data.
        penalty_value : float, optional
            Unused, kept for API compatibility.

        Returns
        -------
        nn.Module
            Acquisition function.
        """
        if self.gp is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        if best_f is None:
            _, Y = self.data_handler.get_gp_data()
            best_f = Y.max().item()

        base_acq = LogExpectedImprovement(self.gp, best_f=best_f)

        # Auto-compute repulsion_lambda if not provided
        if self.repulsion_lambda is None:
            computed_lambda = self._compute_repulsion_lambda(base_acq)
            self._last_computed_lambda = computed_lambda
        else:
            computed_lambda = self.repulsion_lambda

        needles, penalty_radii = self.data_handler.get_needles_and_penalty_radii()
        self.acq_fn = RepulsiveAcquisition(
            base=base_acq,
            proj_fn=self.proj_fn,
            needles=needles,
            penalty_radii=penalty_radii,
            repulsion_lambda=computed_lambda,
        )

        return self.acq_fn

    def _compute_repulsion_lambda(self, base_acq: nn.Module, n_samples: int = 100) -> float:
        """
        Auto-compute repulsion_lambda based on acquisition function scale.

        Uses 10 * median(|acquisition_values|) to ensure repulsion is strong
        enough relative to the acquisition function magnitude.

        Parameters
        ----------
        base_acq : nn.Module
            Base acquisition function to evaluate.
        n_samples : int
            Number of samples to estimate scale. Default: 100.

        Returns
        -------
        float
            Computed repulsion lambda value.
        """
        # Sample random points on the simplex
        bounds = self.data_handler.bounds
        samples = self.random_sampler(
            n_samples, bounds[0], bounds[1],
            device=str(self.device), torch_dtype=self.dtype
        )
        samples_3d = samples.unsqueeze(1)  # (n_samples, 1, d)

        # Evaluate base acquisition
        with torch.no_grad():
            acq_values = base_acq(samples_3d).squeeze()

        # Compute lambda as 10 * median(|acq_values|)
        # Use absolute value since LogEI can be negative
        median_abs_acq = torch.median(torch.abs(acq_values)).item()

        # Ensure minimum lambda to avoid numerical issues
        computed_lambda = max(10.0 * median_abs_acq, 100.0)

        return computed_lambda

    def get_last_computed_lambda(self) -> Optional[float]:
        """Get the last auto-computed repulsion_lambda value, if any."""
        return self._last_computed_lambda

    def _sample_random(self, n: int, bounds: torch.Tensor) -> torch.Tensor:
        """Sample random points on the simplex within bounds."""
        return self.random_sampler(
            n, bounds[0], bounds[1],
            device=str(self.device), torch_dtype=self.dtype
        )

    def get_candidate(
        self,
        bounds: torch.Tensor,
        best_f: Optional[float] = None,
        max_attempts: int = 5,
    ) -> Optional[torch.Tensor]:
        """
        Get next candidate point to evaluate.

        Uses projected gradient ascent to optimize the acquisition function
        while staying on the simplex.

        Parameters
        ----------
        bounds : torch.Tensor
            Search bounds (2, d).
        best_f : float, optional
            Best function value so far.
        max_attempts : int
            Maximum attempts to find unpenalized candidates.

        Returns
        -------
        torch.Tensor or None
            Candidate point (d,), or None if no valid candidate found.
        """
        if self.gp is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        bounds = bounds.to(device=self.device, dtype=self.dtype)

        # Create acquisition function
        acq = self.create_acquisition(best_f=best_f)

        # Sample initial candidates
        ic_candidates = self._sample_random(self.raw_samples, bounds)
        ic_candidates_3d = ic_candidates.unsqueeze(1)  # (raw, 1, d)

        # Evaluate acquisition
        with torch.no_grad():
            acq_values = acq(ic_candidates_3d).squeeze()

        # Find unpenalized candidates
        unpenalized_mask = self.data_handler.get_penalty_mask(ic_candidates)
        unpenalized_indices = torch.where(unpenalized_mask.squeeze())[0]

        # Try to get enough unpenalized candidates
        current_candidates = ic_candidates
        current_candidates_3d = ic_candidates_3d
        current_acq_values = acq_values
        current_unpenalized_indices = unpenalized_indices

        attempt = 0
        while len(current_unpenalized_indices) < self.num_restarts and attempt < max_attempts:
            attempt += 1

            additional_points = self._sample_random(self.raw_samples, bounds)
            additional_points_3d = additional_points.unsqueeze(1)

            with torch.no_grad():
                additional_acq_values = acq(additional_points_3d).squeeze()

            additional_unpenalized_mask = self.data_handler.get_penalty_mask(additional_points)
            additional_unpenalized_indices = torch.where(additional_unpenalized_mask.squeeze())[0]

            # Offset indices for concatenation
            offset = current_candidates.shape[0]
            additional_unpenalized_indices_offset = additional_unpenalized_indices + offset

            current_candidates = torch.cat([current_candidates, additional_points], dim=0)
            current_candidates_3d = torch.cat([current_candidates_3d, additional_points_3d], dim=0)
            current_acq_values = torch.cat([current_acq_values, additional_acq_values], dim=0)
            current_unpenalized_indices = torch.cat([current_unpenalized_indices, additional_unpenalized_indices_offset], dim=0)

        # Check if we have enough unpenalized candidates
        if len(current_unpenalized_indices) == 0:
            return None  # No valid candidates found

        if len(current_unpenalized_indices) < 0.1 * self.num_restarts:
            return None  # Not enough unpenalized area

        num_restarts_to_use = min(self.num_restarts, len(current_unpenalized_indices))

        # Select top candidates by acquisition value
        unpenalized_acq_values = current_acq_values[current_unpenalized_indices]
        top_unpenalized_indices = torch.argsort(unpenalized_acq_values, descending=True)[:num_restarts_to_use]
        selected_indices = current_unpenalized_indices[top_unpenalized_indices]
        initial_conditions = current_candidates_3d[selected_indices]  # (num_restarts, 1, d)

        # Optimize using projected gradient ascent
        best_candidate, best_value = self._optimize_acquisition(
            acq=acq,
            bounds=bounds,
            initial_conditions=initial_conditions,
        )
        if best_candidate is not None and best_value is not None:
            print(f"  [GP] get_candidate: best_candidate = {best_candidate.cpu().numpy()}, best_acq_value = {best_value.item():.6f}")

        # Verify the candidate is not penalized
        if best_candidate is not None:
            is_valid = self.data_handler.get_penalty_mask(best_candidate.unsqueeze(0))
            if not is_valid.any():
                return None

        return best_candidate

    def _optimize_acquisition(
        self,
        acq: nn.Module,
        bounds: torch.Tensor,
        initial_conditions: torch.Tensor,
        step_size: float = 0.05,
        max_steps: int = 50,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Optimize acquisition using projected gradient ascent.

        Parameters
        ----------
        acq : nn.Module
            Acquisition function.
        bounds : torch.Tensor
            Search bounds (2, d).
        initial_conditions : torch.Tensor
            Initial points (num_restarts, 1, d).
        step_size : float
            Gradient ascent step size.
        max_steps : int
            Maximum optimization steps.

        Returns
        -------
        tuple
            (best_candidate, best_value) or (None, None) if failed.
        """
        num_restarts = initial_conditions.shape[0]
        d = initial_conditions.shape[-1]

        best_x = None
        best_val = None

        for r in range(num_restarts):
            x = initial_conditions[r].clone()  # (1, d)

            # Ensure shape (1, 1, d) for acquisition
            if x.dim() == 2:
                x = x.unsqueeze(1)

            # Project initial point
            x = self.proj_fn(x)

            for step in range(max_steps):
                x = x.detach().requires_grad_(True)

                try:
                    val = acq(x)

                    # Gradient ascent
                    grad_x = torch.autograd.grad(val.sum(), x)[0]

                    with torch.no_grad():
                        x = x + step_size * grad_x

                        # Apply box constraints
                        lower, upper = bounds[0], bounds[1]
                        x = torch.max(torch.min(x, upper), lower)

                        # Project back to simplex
                        x = self.proj_fn(x)

                except RuntimeError:
                    break  # Skip this restart on error

            # Evaluate final value
            with torch.no_grad():
                x = x.detach()
                try:
                    final_val = acq(x)
                except RuntimeError:
                    continue  # Skip on error

            if best_val is None or final_val.item() > best_val.item():
                best_val = final_val
                best_x = x

        if best_x is None:
            return None, None

        # Return flattened candidate
        candidate = best_x.squeeze()  # (d,)
        return candidate, best_val

    def determine_penalty_radius(
        self,
        needle: torch.Tensor,
        penalization_threshold: float = 1e-3,
        num_directions: Optional[int] = None,
        max_radius: float = 0.3,
        radius_step: Optional[float] = None,
    ) -> float:
        """
        Determine penalty radius based on acquisition gradient magnitude.

        Parameters
        ----------
        needle : torch.Tensor
            Location of the needle (d,).
        penalization_threshold : float
            Gradient threshold for penalty radius.
        num_directions : int, optional
            Number of directions to sample. If None, uses 10 * d.
        max_radius : float
            Maximum penalty radius.
        radius_step : float, optional
            Step size for radius search. If None, auto-computed as
            max(3 * input_noise, 0.005) where input_noise is the median
            distance between expected and actual measurements.

        Returns
        -------
        float
            Determined penalty radius.
        """
        if self.acq_fn is None:
            # Create acquisition function if needed
            self.create_acquisition()

        d = needle.shape[0]
        
        # Auto-compute num_directions if not provided: 10 * d
        if num_directions is None:
            num_directions = 10 * d
        
        # Auto-compute radius_step if not provided: based on input noise
        # Rationale: no point having step size smaller than measurement noise
        if radius_step is None:
            input_noise = self.data_handler.get_input_noise()
            # Use 3x input noise as minimum meaningful step, with floor of 0.005
            radius_step = max(3.0 * input_noise, 0.005)
        
        dirs = random_zero_sum_directions(num_directions, d, device=str(self.device), dtype=self.dtype)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)

        r = radius_step
        while r <= max_radius:
            X_shell = needle.unsqueeze(0) + r * dirs  # (num_directions, d)
            max_grad = 0.0

            for x in X_shell:
                x = x.unsqueeze(0).unsqueeze(0).clone().detach().requires_grad_(True)  # (1, 1, d)
                try:
                    y = self.acq_fn(x)
                    grads = torch.autograd.grad(y.sum(), x)[0]
                    gm = grads.norm().item()
                    if gm > max_grad:
                        max_grad = gm
                        if max_grad > penalization_threshold:
                            break
                except RuntimeError:
                    continue

            if max_grad <= penalization_threshold:
                # Ensure minimum radius based on input noise
                input_noise = self.data_handler.get_input_noise()
                return max(r, float(3 * input_noise))

            r += radius_step

        return max_radius
