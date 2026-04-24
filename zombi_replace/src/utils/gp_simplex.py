"""
GP Simplex Handler
==================

Handles Gaussian Process fitting and candidate selection for simplex-constrained
optimization. Uses repulsive acquisition plus natural-gradient (Fisher-Rao) ascent
on the simplex for candidate optimization.
"""

import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.autograd import grad
from typing import Literal, Optional, Tuple, Callable, List

from .simplex import proj_simplex, random_simplex
from .datahandler import DataHandler

class RepulsiveAcquisition(nn.Module):
    """
    Acquisition function with smooth repulsion away from discovered needles.

    For needles that have an associated ellipsoid (M, B), the repulsion uses a
    Mahalanobis-distance violation: violation = max(0, 1 - u^T M u) where
    u = B^T (x - needle).  For needles with only a radius (no ellipsoid), the
    original Euclidean violation max(0, r - ||x - needle||) is used as a fallback.

    Parameters
    ----------
    base : nn.Module
        Base acquisition function (e.g., LogExpectedImprovement).
    proj_fn : Callable
        Projection function to simplex.
    needles : torch.Tensor
        Center points of penalty regions (num_needles, d).
    penalty_radii : torch.Tensor
        Radius for each needle (num_needles, 1) or (num_needles,). Used for
        needles without an ellipsoid.
    repulsion_lambda : float
        Strength of repulsion penalty. Default: 1000.0.
    needle_M_list : list of (torch.Tensor or None), optional
        Per-needle (d-1, d-1) Mahalanobis matrix. None entries fall back to
        Euclidean sphere repulsion.
    needle_B : torch.Tensor or None, optional
        Shared (d, d-1) tangent basis for all ellipsoid needles.
    """

    def __init__(
        self,
        base: nn.Module,
        proj_fn: Callable,
        needles: torch.Tensor,
        penalty_radii: torch.Tensor,
        repulsion_lambda: float = 1000.0,
        needle_M_list: Optional[List] = None,
        needle_B: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.base = base
        self.proj_fn = proj_fn
        self.needles = needles  # (M, d)
        self.penalty_radii = penalty_radii.view(-1)  # (M,)
        self.repulsion_lambda = repulsion_lambda
        self.needle_M_list = needle_M_list or []
        self.needle_B = needle_B  # (d, d-1) or None

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
            Acquisition values with repulsion penalty applied.
        """
        X_proj = self.proj_fn(Xq)
        base_acq = self.base(X_proj)

        if self.needles.shape[0] == 0:
            return base_acq

        if X_proj.ndim == 3:
            n, q, d = X_proj.shape
            X_flat = X_proj.reshape(-1, d)  # (B, d)
        else:
            X_flat = X_proj.reshape(-1, X_proj.shape[-1])

        total_violation = torch.zeros(X_flat.shape[0], device=X_flat.device, dtype=X_flat.dtype)

        for idx in range(self.needles.shape[0]):
            needle = self.needles[idx]  # (d,)
            diff = X_flat - needle.unsqueeze(0)  # (B, d)

            M = self.needle_M_list[idx] if idx < len(self.needle_M_list) else None
            if M is not None and self.needle_B is not None:
                # Ellipsoid repulsion: violation = max(0, 1 - u^T M u)
                u = diff @ self.needle_B          # (B, d-1)
                quad = (u @ M * u).sum(dim=-1)    # (B,)
                violation = torch.clamp(1.0 - quad, min=0.0)
            else:
                # Fallback: Euclidean sphere repulsion
                r = self.penalty_radii[idx]
                dist = torch.norm(diff, dim=-1)   # (B,)
                violation = torch.clamp(r - dist, min=0.0)

            total_violation = total_violation + violation ** 2

        penalty = (-self.repulsion_lambda * total_violation).view(base_acq.shape)
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
    acquisition_type : str
        Base acquisition: "ucb" (Upper Confidence Bound) or "ei" (Expected Improvement).
        Both are wrapped with needle repulsion. Default: "ucb".
    ucb_beta : float
        Exploration weight for UCB (mean + beta * std). Only used when acquisition_type=="ucb".
        Default: 0.1.
    nat_grad_step : float
        Step size α for natural-gradient ascent: x ∝ x ⊙ exp(α(g − ḡ_x)).
        Default: 0.02.
    nat_grad_max_steps : int
        Maximum ascent steps per restart. Default: 50.
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
        acquisition_type: Literal["ucb", "ei"] = "ucb",
        ucb_beta: float = 0.1,
        nat_grad_step: float = 0.02,
        nat_grad_max_steps: int = 50,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64,
    ):
        self.data_handler = data_handler
        self.proj_fn = proj_fn if proj_fn is not None else proj_simplex
        self.random_sampler = random_sampler if random_sampler is not None else random_simplex
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.repulsion_lambda = repulsion_lambda  # None means auto-compute
        self.acquisition_type = acquisition_type.lower()
        if self.acquisition_type not in ("ucb", "ei"):
            raise ValueError(f"acquisition_type must be 'ucb' or 'ei', got {acquisition_type!r}")
        self.ucb_beta = ucb_beta
        self.nat_grad_step = nat_grad_step
        self.nat_grad_max_steps = nat_grad_max_steps
        self.device = torch.device(device)
        self.dtype = dtype

        self.gp: Optional[SingleTaskGP] = None
        self.mll = None
        self.acq_fn = None
        self._last_computed_lambda = None  # Track auto-computed lambda for logging
        self._tangent_basis: Optional[torch.Tensor] = None  # cached (d, d-1)

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
        if self.acquisition_type == "ei":
            base_acq = LogExpectedImprovement(self.gp, best_f=best_f)
            with torch.no_grad():
                val = base_acq(x_3d).squeeze().item()
            return val
        else:
            # UCB: return acquisition value at point for logging (not log EI)
            base_acq = UpperConfidenceBound(self.gp, beta=self.ucb_beta)
            with torch.no_grad():
                val = base_acq(x_3d).squeeze().item()
            return val

    def create_acquisition(
        self,
        best_f: Optional[float] = None,
        penalty_value: Optional[float] = None,
    ) -> nn.Module:
        """
        Create acquisition function (base + repulsion).

        Parameters
        ----------
        best_f : float, optional
            Best function value so far. Used only for EI. If None, computed from data.
        penalty_value : float, optional
            Unused, kept for API compatibility.

        Returns
        -------
        nn.Module
            Acquisition function (UCB or EI, wrapped with needle repulsion).
        """
        if self.gp is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        if self.acquisition_type == "ucb":
            base_acq = UpperConfidenceBound(self.gp, beta=self.ucb_beta)
        else:
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
        needle_M_list, needle_B = self.data_handler.get_needle_ellipsoids()
        self.acq_fn = RepulsiveAcquisition(
            base=base_acq,
            proj_fn=self.proj_fn,
            needles=needles,
            penalty_radii=penalty_radii,
            repulsion_lambda=computed_lambda,
            needle_M_list=needle_M_list,
            needle_B=needle_B,
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
        exclude_near: Optional[torch.Tensor] = None,
        exclude_near_tol: float = 1e-8,
    ) -> Optional[torch.Tensor]:
        """
        Get next candidate point to evaluate.

        Uses natural-gradient ascent on the simplex to optimize the acquisition.
        If exclude_near is set, the best candidate
        that is not within exclude_near_tol of exclude_near is returned (no two
        same points in a row).

        Parameters
        ----------
        bounds : torch.Tensor
            Search bounds (2, d).
        best_f : float, optional
            Best function value so far.
        max_attempts : int
            Maximum attempts to find unpenalized candidates.
        exclude_near : torch.Tensor, optional
            Last suggested point(s) to avoid repeating. Shape (d,) or (K, d).
            If not None, skip candidates within exclude_near_tol of *any* of these
            points so we don't bounce between the same few local maxima.
        exclude_near_tol : float
            Minimum distance from each exclude_near point for a candidate to be allowed. Default 1e-8.

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

        candidates, values = self._optimize_acquisition(
            acq=acq,
            bounds=bounds,
            initial_conditions=initial_conditions,
        )

        if candidates.shape[0] == 0:
            return None

        # Sort by acquisition value descending (best first)
        order = torch.argsort(values, descending=True)
        candidates = candidates[order]
        values = values[order]

        # If exclude_near set, pick first candidate not within tol of *any* excluded point
        if exclude_near is not None:
            exclude_near = exclude_near.to(device=candidates.device, dtype=candidates.dtype)
            if exclude_near.dim() == 1:
                exclude_near = exclude_near.unsqueeze(0)  # (1, d)
            # candidates (R, d), exclude_near (K, d) -> distances (R, K)
            distances = torch.norm(
                candidates.unsqueeze(1) - exclude_near.unsqueeze(0), dim=2
            )
            allowed = (distances >= exclude_near_tol).all(dim=1)  # (R,)
            if allowed.any():
                # Take best allowed candidate
                best_idx = torch.where(allowed)[0][0]
                best_candidate = candidates[best_idx]
                best_value = values[best_idx]
            else:
                # All restarts converged to excluded points; fall back to best so we don't stall
                best_candidate = candidates[0]
                best_value = values[0]
        else:
            best_candidate = candidates[0]
            best_value = values[0]

        if best_candidate is not None:
            print(f"  [GP] get_candidate: best_candidate = {best_candidate.cpu().numpy()}, best_acq_value = {best_value.item():.6f}")

        # Verify the candidate is not penalized
        is_valid = self.data_handler.get_penalty_mask(best_candidate.unsqueeze(0))
        if not is_valid.any():
            return None

        return best_candidate

    def _optimize_acquisition(
        self,
        acq: nn.Module,
        bounds: torch.Tensor,
        initial_conditions: torch.Tensor,
        nat_grad_step: Optional[float] = None,
        nat_grad_max_steps: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Natural-gradient ascent on the simplex (all restarts).

        Update: g = ∇acq(x), ḡ = Σᵢ xᵢ gᵢ, then x ← normalize(x ⊙ exp(α(g − ḡ))),
        clamp to box bounds, renormalize. Matches Method D (Fisher-Rao) used
        previously in NaturalGradGPSimplex.

        For backward compatibility, ``step_size`` and ``max_steps`` kwargs are
        accepted as aliases for ``nat_grad_step`` and ``nat_grad_max_steps`` when
        those are omitted.
        """
        if "step_size" in kwargs:
            ss = kwargs.pop("step_size")
            if nat_grad_step is None:
                nat_grad_step = ss
        if "max_steps" in kwargs:
            ms = kwargs.pop("max_steps")
            if nat_grad_max_steps is None:
                nat_grad_max_steps = ms
        if kwargs:
            raise TypeError(f"_optimize_acquisition got unexpected keyword arguments: {set(kwargs)}")

        step = self.nat_grad_step if nat_grad_step is None else float(nat_grad_step)
        n_steps = self.nat_grad_max_steps if nat_grad_max_steps is None else int(nat_grad_max_steps)

        lo = bounds[0].to(device=self.device, dtype=self.dtype)
        hi = bounds[1].to(device=self.device, dtype=self.dtype)
        d = initial_conditions.shape[-1]
        candidates_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []

        for r in range(initial_conditions.shape[0]):
            x_raw = initial_conditions[r]
            x = x_raw.reshape(d).clone().to(device=self.device, dtype=self.dtype)
            x = self.proj_fn(x.unsqueeze(0)).squeeze(0)

            for _ in range(n_steps):
                x = x.detach().requires_grad_(True)
                x_in = x.unsqueeze(0).unsqueeze(0)

                try:
                    val = acq(x_in)
                    g = torch.autograd.grad(val.sum(), x)[0]
                except RuntimeError:
                    break

                with torch.no_grad():
                    x_det = x.detach()
                    g_bar = (x_det * g).sum()
                    shift = step * (g - g_bar)
                    shift = torch.clamp(shift, -10.0, 10.0)
                    x_new = x_det * torch.exp(shift)
                    s = x_new.sum()
                    if s < 1e-12:
                        break
                    x_new = x_new / s
                    x_new = torch.clamp(x_new, lo, hi)
                    s2 = x_new.sum()
                    if s2 < 1e-12:
                        break
                    x = x_new / s2

            x = x.detach()
            x_in = x.unsqueeze(0).unsqueeze(0)
            try:
                with torch.no_grad():
                    final_val = acq(x_in)
            except RuntimeError:
                continue

            candidates_list.append(x)
            values_list.append(final_val.squeeze())

        if not candidates_list:
            return (
                torch.empty(0, d, device=bounds.device, dtype=bounds.dtype),
                torch.empty(0, device=bounds.device, dtype=bounds.dtype),
            )

        return torch.stack(candidates_list, dim=0), torch.stack(values_list, dim=0)

    def _get_tangent_basis(self, d: int) -> torch.Tensor:
        """
        Return (d, d-1) orthonormal basis for the simplex tangent space
        {v : sum(v) = 0}.  Result is cached and reused while d is unchanged.
        """
        if self._tangent_basis is not None and self._tangent_basis.shape[0] == d:
            return self._tangent_basis
        P = torch.eye(d, device=self.device, dtype=self.dtype) - (1.0 / d)
        Q, _ = torch.linalg.qr(P)
        B = Q[:, :d - 1].contiguous()
        self._tangent_basis = B
        return B

    def determine_penalty_ellipsoid(
        self,
        needle: torch.Tensor,
        drop_fraction: float = 0.25,
        eigenvalue_floor: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the ellipsoidal penalty region for a needle via the tangent-space
        Hessian of the acquisition function.

        Returns (M, B) where a point x is considered inside the basin iff
            u = B^T (x - needle),  u^T M u <= 1.

        M is (d-1, d-1); B is (d, d-1) — the shared simplex tangent basis.

        Parameters
        ----------
        needle : torch.Tensor
            Converged point on the simplex, shape (d,).
        drop_fraction : float
            Delta = max(drop_fraction * |alpha(needle)|, noise_floor).
            Controls how far the ellipsoid extends from the peak.
        eigenvalue_floor : float
            Minimum eigenvalue of -H_tilde to cap absurdly elongated ellipsoids.
        """
        if self.acq_fn is None:
            self.create_acquisition()

        d = needle.shape[0]
        needle = needle.detach().to(device=self.device, dtype=self.dtype)
        B = self._get_tangent_basis(d)  # (d, d-1)

        def tilde_alpha(u: torch.Tensor) -> torch.Tensor:
            x = needle + B @ u
            return self.acq_fn(x.view(1, 1, d)).squeeze()

        u0 = torch.zeros(d - 1, device=self.device, dtype=self.dtype)
        H = torch.autograd.functional.hessian(tilde_alpha, u0)  # (d-1, d-1)
        neg_H = -0.5 * (H + H.T)  # symmetrize

        eigvals, eigvecs = torch.linalg.eigh(neg_H)
        eigvals = eigvals.clamp(min=eigenvalue_floor)

        alpha_peak = abs(self.acq_fn(needle.view(1, 1, d)).squeeze().item())
        sigma = self.data_handler.get_input_noise()
        lambda_max = eigvals.max().item()

        # Delta: acquisition drop that defines the basin boundary
        Delta_acq = drop_fraction * alpha_peak
        Delta_noise = 0.5 * lambda_max * (3.0 * sigma) ** 2  # noise floor
        Delta = max(Delta_acq, Delta_noise, 1e-12)

        neg_H_clean = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        M = neg_H_clean / (2.0 * Delta)
        return M, B

