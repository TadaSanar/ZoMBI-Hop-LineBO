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
                 max_gp_points: int = 200,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float64):
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

    def run(self):
        # CUDA memory management
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"Starting optimization. CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        finished = False
        for activation in range(5):
            print(f"{"-"*10} Activation {activation+1} {"-"*10}")

            # CUDA memory cleanup between activations
            if self.device.type == 'cuda' and activation > 0:
                torch.cuda.empty_cache()
                print(f"Activation {activation} - CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

            self.no_improvements = 0
            needle = None
            bounds = self.bounds.clone()
            activation_failed = False
            for zoom in range(self.max_zooms):
                print(f"{"-"*10} Zoom {zoom+1}/{self.max_zooms}  {"-"*10}")
                print(f"Bounds: {bounds}")
                # 1) get GP data
                X, Y = self._get_gp_data(bounds)

                # 2) fit GP
                self.gp = SingleTaskGP(X, Y)
                mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
                fit_gpytorch_mll(mll)

                for iteration in range(self.max_iterations):
                    print(f"{"-"*10} Iteration {iteration+1}/{self.max_iterations}  {"-"*10}")

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
                    self.gp = SingleTaskGP(X, Y)
                    mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
                    fit_gpytorch_mll(mll)

                    if unpenalized_Y.shape[0] == 0:
                        print("No unpenalized Y values, breaking")
                        activation_failed = True
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

                    if self.no_improvements >= self.n_consecutive_no_improvements:
                        max_idx = self.Y_all[self.penalty_mask].argmax()
                        needle = self.X_all_actual[self.penalty_mask][max_idx]
                        assert isinstance(needle, torch.Tensor), "needle must be a torch tensor"
                        assert needle.shape == (self.d,), "needle must be a (d,) torch tensor"

                        print(f"No improvements! Found needle {needle} with value {self.Y_all[max_idx]}")

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
                            'value': self.Y_all[max_idx].item(),
                            'activation': activation,
                            'zoom': zoom,
                            'iteration': iteration
                        })
                        self.needles = torch.cat([self.needles, needle.unsqueeze(0)], dim=0)
                        self.needle_vals = torch.cat([self.needle_vals, self.Y_all[max_idx].reshape(1, 1)], dim=0)
                        self.needle_indices = torch.cat([self.needle_indices, max_idx.reshape(1, 1)], dim=0)
                        self.needle_penalty_radii = torch.cat([self.needle_penalty_radii, torch.tensor([[penalty_radius]], device=self.device, dtype=self.dtype)], dim=0)

                        # 4) update penalty mask
                        self._set_penalty_mask()
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
                    break

                if finished:
                    break
                if zoom < self.max_zooms - 1:
                    bounds = self._determine_new_bounds()

        # 5) return the needles

        # Final CUDA memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"Optimization complete. Final CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

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

        # Get acquisition values
        with torch.no_grad():
            acquisition_values_flat = acquisition_function(points_flat)

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

# Global minima locations in [0,1]^d simplex space
# Note: The ackley function internally scales these from [0,1] to [-100,100] for computation
# Global minima locations in [0,1]^d simplex space
MINIMA_ONE = torch.tensor([0.4, 0.0, 0.1, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
MINIMA_TWO = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.5], dtype=torch.float64)
MINIMA_THREE = torch.tensor([0.0, 0.1, 0.2, 0.1, 0.5, 0.1, 0.1, 0.0, 0.1, 0.0], dtype=torch.float64)

OUTPUT_SIGMA2 = 6.5e-2  # Output noise variance
INPUT_SIGMA2 = 4.3036e-3  # Input noise variance

def create_objective_functions(num_experiments=24, device='cuda'):
    """
    Creates a list of objective functions for the meta-optimization.
    Each function returns (x_actual, y) where x_actual are points on the simplex and y are function values.
    """
    from test_functions_torch import MultiMinimaAckley

    device = torch.device(device)

    # Create easy and hard parameter sets
    easy_params = {
        'global_scale': 20.0,
        'exp_scale': 0.2,
        'amplitudes': torch.ones(3),  # Will use first 1-3 values based on num_minima
        'sharpness': 5.0 * torch.ones(3)
    }

    hard_params = {
        'global_scale': 6.0,
        'exp_scale': 0.3,
        'amplitudes': torch.ones(3),
        'sharpness': 9.0 * torch.ones(3)
    }

    def create_line_objective(minima_locs, params):
        func = MultiMinimaAckley(
            minima_locs,
            amplitudes=params['amplitudes'][:len(minima_locs)],
            sharpness=params['sharpness'][:len(minima_locs)],
            global_scale=params['global_scale'],
            exp_scale=params['exp_scale']
        )

        def _line_objective(endpoints):
            if not torch.is_tensor(endpoints):
                endpoints = torch.tensor(endpoints, device=device, dtype=torch.float64)
            if endpoints.dim() == 2:
                endpoints = endpoints.unsqueeze(0)

            # Generate requested points along line
            t_values = torch.linspace(0, 1, num_experiments, device=device, dtype=torch.float64)
            x_requested = (1 - t_values).unsqueeze(1) * endpoints[0, 0].unsqueeze(0) + \
                         t_values.unsqueeze(1) * endpoints[0, 1].unsqueeze(0)

            # Add input noise
            input_noise = torch.randn_like(x_requested) * torch.sqrt(torch.tensor(INPUT_SIGMA2, dtype=torch.float64))
            x_actual = x_requested + input_noise

            # Project back to simplex
            x_actual = torch.clamp(x_actual, min=0.0)
            x_actual = x_actual / x_actual.sum(dim=1, keepdim=True)

            # Evaluate function and add output noise
            y = -func.evaluate(x_actual)
            output_noise = torch.randn_like(y) * torch.sqrt(torch.tensor(OUTPUT_SIGMA2, dtype=torch.float64))
            y = y + output_noise

            return x_actual, y

        def _evaluate_needles(needles):
            """
            Evaluates how well the found needles match the known minima locations.

            Args:
                needles: Tensor of shape (num_needles, d) containing needle locations

            Returns:
                List of distances from each minima to its closest needle, or None if insufficient needles
            """
            if needles.shape[0] < len(minima_locs):
                return None

            # Calculate pairwise distances between needles and minima
            needles = needles.to(device=device)
            minima = minima_locs.to(device=device)

            distances = torch.cdist(needles, minima)

            # For each minimum, find the closest needle
            min_distances = []
            remaining_needles = list(range(len(needles)))

            for i in range(len(minima)):
                if not remaining_needles:
                    return None

                # Find closest remaining needle to this minimum
                min_dist = float('inf')
                best_needle_idx = -1

                for needle_idx in remaining_needles:
                    dist = distances[needle_idx, i].item()
                    if dist < min_dist:
                        min_dist = dist
                        best_needle_idx = needle_idx

                if best_needle_idx >= 0:
                    min_distances.append(min_dist)
                    remaining_needles.remove(best_needle_idx)
                else:
                    return None

            # Print evaluation results
            print("\n" + "="*60)
            print("🚨🚨🚨 ALL NEEDLES FOUND 🚨🚨🚨")
            print("="*60)
            for i, dist in enumerate(min_distances):
                print(f"Distance to minimum {i+1}: {dist:.6f}")
            print("="*60 + "\n")

            return min_distances

        # Attach evaluator to objective function
        _line_objective.evaluate_needles = _evaluate_needles
        return _line_objective

    # Create objectives for 1, 2, and 3 minima cases
    minima_sets = [
        MINIMA_ONE.unsqueeze(0),  # One minimum
        torch.stack([MINIMA_ONE, MINIMA_TWO]),  # Two minima
        torch.stack([MINIMA_ONE, MINIMA_TWO, MINIMA_THREE])  # Three minima
    ]

    objectives = []
    for minima in minima_sets:
        # Add easy version
        objectives.append(create_line_objective(minima, easy_params))
        # Add hard version
        objectives.append(create_line_objective(minima, hard_params))

    return objectives  # Returns 6 functions: 1-easy, 1-hard, 2-easy, 2-hard, 3-easy, 3-hard

def analyze_objective_functions(resolution=100, device='cuda'):
    """
    Creates a mesh grid over the simplex and finds global maxima and minima
    for each objective function created by create_objective_functions.
    """
    import numpy as np
    import torch

    # Get the dimension from MINIMA_ONE
    d = len(MINIMA_ONE)

    bounds = torch.zeros((2, d), device=torch.device(device), dtype=torch.float64)
    bounds[0] = 0.0  # Lower bounds
    bounds[1] = 1.0  # Upper bounds

    points = ZoMBIHop.random_simplex(1000000, bounds[0], bounds[1])

    points = torch.tensor(points, device=device, dtype=torch.float64)

    # Get all objective functions
    objectives = create_objective_functions(device=device)

    # Evaluate each objective function
    for i, obj in enumerate(objectives):
        # Create endpoints tensor for each point
        values = []
        for p in points:
            # Call objective with single point duplicated as start/end
            endpoints = p.unsqueeze(0).repeat(2, 1)  # Shape: (2, d)
            x_actual, y = obj(endpoints)
            values.append(y[0].item())  # Take first value since points are duplicated

        values = torch.tensor(values, device=device)
        max_idx = torch.argmax(values)
        min_idx = torch.argmin(values)

        # Determine which case this is
        case_num = i // 2 + 1  # 1, 1, 2, 2, 3, 3
        difficulty = "Easy" if i % 2 == 0 else "Hard"

        print(f"\n{case_num}-minima {difficulty} function:")
        print(f"Maximum value {values[max_idx]:.4f} at point {points[max_idx]}")
        print(f"Minimum value {values[min_idx]:.4f} at point {points[min_idx]}")

class ZombihopPenaltyObjective:
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, x):
        """
        Objective function for optimizing ZoMBIHop penalty parameters.
        Returns a penalty value of 1.0 if optimization fails to converge.
        Otherwise returns average of distances to minima across runs.

        Args:
            x (list): List of penalty parameters:
                - penalization_threshold
                - improvement_threshold_noise_mult
                - input_noise_threshold_mult
                - n_consecutive_no_improvements
                - top_m_points
                - max_iterations

        Returns:
            float: Performance metric (lower is better)
        """
        param_names = [
            'penalization_threshold',
            'improvement_threshold_noise_mult',
            'input_noise_threshold_mult',
            'n_consecutive_no_improvements',
            'top_m_points',
            'max_iterations'
        ]
        params = dict(zip(param_names, x))

        # Round integer parameters
        params['n_consecutive_no_improvements'] = max(3, int(round(params['n_consecutive_no_improvements'])))
        params['top_m_points'] = max(2, int(round(params['top_m_points'])))
        params['max_iterations'] = max(3, int(round(params['max_iterations'])))

        # Set fixed values for other parameters
        fixed_params = {
            'max_zooms': 3,
            'n_restarts': 100,
            'raw': 10_000,
            'penalty_num_directions': 100,
            'penalty_max_radius': 0.3,
            'penalty_radius_step': 0.01,
            'max_gp_points': 500
        }
        params.update(fixed_params)

        objective_functions = create_objective_functions(device=torch.device(self.device))
        all_distances = []

        # Create bounds tensor with shape (2, d) for min/max in each dimension
        bounds = torch.zeros((2, 10), device=torch.device(self.device), dtype=torch.float64)
        bounds[0] = 0.0  # Lower bounds
        bounds[1] = 1.0  # Upper bounds

        penalization = 24.0

        # Run multiple repeats
        for _ in range(2):
            for objective_function in objective_functions:
                linebo = LineBO(objective_function, dimensions=10, num_points_per_line=100, num_lines=1000, device=self.device)

                num_random_points = 2
                random_points = ZoMBIHop.random_simplex(num_random_points, bounds[0], bounds[1])
                print(bounds)
                print(random_points)
                X_init_actual = torch.empty((0, 10), device=torch.device(self.device), dtype=torch.float64)
                X_init_expected = torch.empty((0, 10), device=torch.device(self.device), dtype=torch.float64)
                Y_init = torch.empty((0, 1), device=torch.device(self.device), dtype=torch.float64)

                for random_point in random_points:
                    x_requested, x_actual, y = linebo.sampler(random_point, bounds=bounds)
                    X_init_actual = torch.cat([X_init_actual, x_actual], dim=0)
                    X_init_expected = torch.cat([X_init_expected, x_requested], dim=0)
                    Y_init = torch.cat([Y_init, y.unsqueeze(1)], dim=0)

                zombihop = ZoMBIHop(
                    objective=linebo.sampler,
                    bounds=bounds,
                    X_init_actual=X_init_actual,
                    X_init_expected=X_init_expected,
                    Y_init=Y_init,
                    penalization_threshold=params['penalization_threshold'],
                    improvement_threshold_noise_mult=params['improvement_threshold_noise_mult'],
                    input_noise_threshold_mult=params['input_noise_threshold_mult'],
                    n_consecutive_no_improvements=params['n_consecutive_no_improvements'],
                    top_m_points=params['top_m_points'],
                    max_zooms=params['max_zooms'],
                    max_iterations=params['max_iterations'],
                    n_restarts=params['n_restarts'],
                    raw=params['raw'],
                    penalty_num_directions=params['penalty_num_directions'],
                    penalty_max_radius=params['penalty_max_radius'],
                    penalty_radius_step=params['penalty_radius_step'],
                    max_gp_points=params['max_gp_points'],
                    device=self.device
                )

                needles, needles_torch, needle_vals, X_all_actual, Y_all = zombihop.run()

                distances = objective_function.evaluate_needles(needles_torch)

                # Return penalty if optimization didn't converge OR found no needles
                if distances is None:
                    print("\n" + "="*60)
                    print("🚨🚨🚨 ZOMBIHOP OPTIMIZATION FAILED 🚨🚨🚨")
                    print(f"Optimization returned {len(needles) if needles else 0} needles")
                    print("="*60)
                    print("Parameters used:")
                    for k, v in params.items():
                        print(f"  {k}: {v}")
                    print("="*60 + "\n")
                    return penalization
                penalization -= 2.0

                all_distances.extend(distances)

        average_distance = sum(all_distances) / len(all_distances)

        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)

        # Generate UUID for this result
        result_uuid = str(uuid.uuid4())
        result_file = os.path.join('results', f'result_{result_uuid}.json')

        # Create result dictionary
        result_dict = {
            'params': params,
            'all_distances': all_distances,
            'average_distance': average_distance
        }

        # Save result to unique JSON file
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print("\n" + "="*60)
        print("🚨🚨🚨 ZOMBIHOP OPTIMIZATION COMPLETE 🚨🚨🚨")
        print("="*60)
        print("Parameters used:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print("-"*60)
        print(f"Number of successful runs: {len(all_distances) // 2}")
        print(f"Average distance to minima: {average_distance:.6f}")
        print("="*60 + "\n")

        return float(average_distance)

def optimize_penalty_parameters(device='cuda', seed=42):
    """
    Optimize ZoMBIHop penalty parameters using Differential Evolution
    """
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    bounds = [
        (1e-4, 1e-2),  # penalization_threshold
        (0.5, 6.0),    # improvement_threshold_noise_mult
        (2.5, 6.0),    # input_noise_threshold_mult
        (3.0, 9.0),   # n_consecutive_no_improvements
        (2.0, 8.0),   # top_m_points
        (8.0, 15.0)    # max_iterations
    ]

    objective = ZombihopPenaltyObjective(device=device)

    print("Starting penalty parameter optimization...")
    result = differential_evolution(
        objective,
        bounds,
        maxiter=80,
        popsize=30,
        seed=seed,
        polish=True,
        disp=True,
        updating='immediate'
    )

    best_params = {
        'penalization_threshold': result.x[0],
        'improvement_threshold_noise_mult': result.x[1],
        'input_noise_threshold_mult': result.x[2],
        'n_consecutive_no_improvements': max(3, int(round(result.x[3]))),
        'top_m_points': max(2, int(round(result.x[4]))),
        'max_iterations': max(3, int(round(result.x[5])))
    }

    print("\nOptimization complete!")
    print(f"Best performance: {result.fun:.6f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params, result.fun

if __name__ == "__main__":

    # analyze_objective_functions()
    import json

    # Ensure CUDA is available and optimize
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA-capable GPU.")

    # Set device and optimize CUDA settings
    device = 'cuda'
    torch.cuda.empty_cache()  # Clear GPU memory
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    best_params, best_perf = optimize_penalty_parameters(device=device)

    with open('penalty_optimization_results.json', 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_performance': best_perf
        }, f, indent=2)
    print("\nResults saved to penalty_optimization_results.json")

    # Test with the correct dimension (10D)
    d = len(MINIMA_ONE)  # Use the dimension from MINIMA_ONE
    bounds = torch.zeros((2, d), device='cuda', dtype=torch.float64)
    bounds[0] = 0.0  # Lower bounds
    bounds[1] = 1.0  # Upper bounds
    print(f"Testing with {d}D bounds:")
    print(bounds)

    # Create test points with correct dimension
    test_point_1 = torch.zeros(d, device='cuda', dtype=torch.float64)
    test_point_2 = torch.ones(d, device='cuda', dtype=torch.float64) / d  # Normalize to sum to 1

    zombi = ZoMBIHop(
        objective=lambda x: x,
        bounds=bounds,
        X_init_actual=torch.stack([test_point_1, test_point_2]),
        X_init_expected=torch.stack([test_point_1, test_point_2]),
        Y_init=torch.tensor([[0], [1]], device='cuda', dtype=torch.float64)
    )

    # Test penalty mask with correct dimension
    test_tensor = torch.stack([
        torch.stack([test_point_1, test_point_1]),
        torch.stack([test_point_2, test_point_2])
    ])
    print(f"Testing penalty mask with {d}D tensor:")
    print(zombi._get_penalty_mask(test_tensor))
