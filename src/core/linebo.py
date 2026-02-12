"""
LineBO: Line-based Bayesian Optimization for Simplex Constraints
================================================================

LineBO optimizes over the probability simplex by:
1. Sampling zero-sum directions (so lines stay in the simplex-affine subspace).
2. Finding each line's intersection with the simplex (segment endpoints on the boundary).
3. Integrating an acquisition function along segments and picking the best line.
4. Evaluating the objective along that line and returning requested/actual points and values.

All tensors use float64 for numerical stability. Shapes use k = number of lines, d = dimensions,
n = number of evaluation points.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

# Resampling limits (worst-case caps)
ZERO_SUM_DIR_MAX_RESAMPLE = 100   # max resamples for zero-norm rows in zero_sum_dirs (default: 100)
SAMPLER_MAX_EXTRA_ATTEMPTS = 50  # max extra direction batches in LineBO.sampler when valid lines < num_lines (default: 50)


def zero_sum_dirs(k: int, d: int, device=None, dtype=torch.float64, seed=None) -> torch.Tensor:
    """
    Sample k vectors of dimension d with zero sum and unit norm.

    Parameters
    ----------
    k : int
        Number of vectors to sample.
    d : int
        Dimensionality of each vector.
    device : str, optional
        Torch device.
    dtype : torch.dtype
        Data type for the tensors.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        (k, d) tensor of zero-sum, unit-norm vectors.
    """
    if seed is not None:
        torch.manual_seed(seed)
    # Raw directions: (k, d)
    v = torch.randn(k, d, device=device, dtype=dtype)
    # Center each row so sum over d is 0 (zero-sum constraint)
    v -= v.mean(dim=1, keepdim=True)  # v: (k, d)
    # Row-wise L2 norm for normalization: (k, 1)
    n = v.norm(dim=1, keepdim=True)
    # Resample any rows that ended up with zero norm (degenerate); cap iterations to avoid worst-case infinite loop
    mask = (n.squeeze() == 0)  # (k,) bool
    for _ in range(ZERO_SUM_DIR_MAX_RESAMPLE):
        if not mask.any():
            break
        r = torch.randn(mask.sum().item(), d, device=device, dtype=dtype)  # (num_zero, d)
        r -= r.mean(dim=1, keepdim=True)
        v[mask] = r
        n[mask] = v[mask].norm(dim=1, keepdim=True)
        mask = (n.squeeze() == 0)
    # Return unit-norm directions: (k, d) [if any row still zero-norm after cap, divide may produce inf/nan]
    return v / n


def line_simplex_segment(x0: torch.Tensor, d: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Find the intersection of a line with the simplex.

    Given a starting point x0 on the simplex and a direction d,
    find the segment of the line x0 + t*d that lies within the simplex.

    Parameters
    ----------
    x0 : torch.Tensor
        Starting point (d,) on the simplex.
    d : torch.Tensor
        Direction vector (d,).

    Returns
    -------
    tuple or None
        (t_min, t_max, x_left, x_right) if line intersects simplex,
        None if no intersection.
    """
    # Which components of d are negative / positive (for boundary hit times)
    neg = d < 0   # (d,) bool
    pos = d > 0   # (d,) bool

    inf = torch.tensor(float('inf'), device=d.device, dtype=d.dtype)   # scalar
    ninf = torch.tensor(float('-inf'), device=d.device, dtype=d.dtype)  # scalar

    # For x0 + t*d to stay in simplex [0,1]^d: each coord must stay >= 0.
    # Coord j hits 0 when t = -x0[j]/d[j] (only if d[j]!=0). Take min over neg d, max over pos d.
    t_max = torch.min(torch.where(neg, -x0 / d, inf))   # scalar: largest t keeping all coords >= 0
    t_min = torch.max(torch.where(pos, -x0 / d, ninf))  # scalar: smallest t keeping all coords >= 0

    if t_min > t_max:
        return None  # Line does not intersect simplex

    # Endpoints of the segment on the simplex boundary
    x_left = x0 + t_min * d   # (d,)
    x_right = x0 + t_max * d  # (d,)
    return t_min, t_max, x_left, x_right


def batch_line_simplex_segments(x0: torch.Tensor, D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find intersections of multiple lines with the simplex (batched).

    Parameters
    ----------
    x0 : torch.Tensor
        Starting point (d,) on the simplex.
    D : torch.Tensor
        Direction vectors (k, d).

    Returns
    -------
    tuple
        (x_left, x_right, t_min, t_max, mask) where mask indicates valid lines.
    """
    neg = D < 0   # (k, d) bool: which direction components are negative
    pos = D > 0   # (k, d) bool: which direction components are positive

    inf = torch.tensor(float('inf'), device=D.device, dtype=D.dtype)   # scalar
    ninf = torch.tensor(float('-inf'), device=D.device, dtype=D.dtype)  # scalar

    # Per-line: -x0[j]/D[i,j] where D[i,j]!=0; broadcast x0 (d,) to (k, d) via unsqueeze(0)
    t_max = torch.min(torch.where(neg, -x0.unsqueeze(0)/D, inf), dim=1).values   # (k,)
    t_min = torch.max(torch.where(pos, -x0.unsqueeze(0)/D, ninf), dim=1).values  # (k,)

    # Only count segments with positive length (t_min < t_max); reject degenerate/vertex cases
    mask = t_min < t_max   # (k,) bool: valid lines
    t_min = t_min[mask]    # (num_valid,)
    t_max = t_max[mask]    # (num_valid,)
    Dm = D[mask]           # (num_valid, d)

    # Left/right endpoints on simplex boundary: x0 + t*d for each valid line
    x_left = x0.unsqueeze(0) + t_min.unsqueeze(1) * Dm   # (num_valid, d)
    x_right = x0.unsqueeze(0) + t_max.unsqueeze(1) * Dm  # (num_valid, d)

    return x_left, x_right, t_min, t_max, mask


class LineBO:
    """
    Line-based Bayesian Optimization for simplex-constrained problems.

    Given a current point x_tell on the simplex, samples many zero-sum directions,
    finds the simplex-boundary segment for each, integrates the acquisition along
    each segment, picks the best line, and evaluates the objective on that line.
    Uses float64 throughout for numerical stability.

    Parameters
    ----------
    objective_function : Callable
        Function that accepts line endpoints (2, d) and returns (x_actual, y).
    dimensions : int
        Number of dimensions (d).
    num_points_per_line : int
        Number of points to sample along each line for integration. Default: 100.
    num_lines : int or None
        Number of candidate lines to generate and evaluate. Default: None (auto: 10*d).
    device : str
        Device for computations. Default: 'cuda'.
    """

    def __init__(self, objective_function, dimensions: int, num_points_per_line: int = 100,
                 num_lines: Optional[int] = None, device: str = 'cuda'):
        """Initialize LineBO sampler: store objective, dimensions, line-discretization and device."""
        self.objective_function = objective_function
        self.d = dimensions
        self.num_points_per_line = num_points_per_line  # points per line for acquisition integration
        # Auto-compute num_lines as 10*d if not specified
        self.num_lines = num_lines if num_lines is not None else 10 * dimensions
        self.device = torch.device(device)
        self.dtype = torch.float64

    def _integrate_acquisition_along_lines(self, x_left: torch.Tensor, x_right: torch.Tensor,
                                         acquisition_function: nn.Module) -> torch.Tensor:
        """
        Integrate acquisition function along multiple line segments (average over t in [0,1]).

        Parameters
        ----------
        x_left : torch.Tensor
            (k, d) left endpoints of line segments.
        x_right : torch.Tensor
            (k, d) right endpoints of line segments.
        acquisition_function : nn.Module
            Acquisition function to evaluate.

        Returns
        -------
        torch.Tensor
            (k,) integrated acquisition values (mean over points per line).
        """
        k = x_left.shape[0]  # number of lines

        # Parameter along each segment: t in [0, 1], num_points_per_line values
        t_values = torch.linspace(0, 1, self.num_points_per_line,
                                device=self.device, dtype=torch.float64)  # (num_points_per_line,)

        # Interpolate: (1-t)*x_left + t*x_right for each line and each t
        # x_left.unsqueeze(1) -> (k, 1, d), (1-t).view(1,-1,1) -> (1, num_points_per_line, 1); broadcast -> (k, num_points_per_line, d)
        points = (x_left.unsqueeze(1) * (1 - t_values).view(1, -1, 1) +
                 x_right.unsqueeze(1) * t_values.view(1, -1, 1))  # (k, num_points_per_line, d)

        # Flatten to (k * num_points_per_line, 1, d) for acquisition (often expects batch with last dim d)
        points_flat = points.reshape(-1, 1, self.d)  # (k * num_points_per_line, 1, d)

        batch_size = 500
        num_points = points_flat.shape[0]
        acquisition_values_flat = torch.empty(num_points, device=self.device, dtype=torch.float64)  # (num_points,)

        with torch.no_grad():
            for i in range(0, num_points, batch_size):
                end_idx = min(i + batch_size, num_points)
                batch = points_flat[i:end_idx]  # (batch_len, 1, d)

                try:
                    batch_values = acquisition_function(batch)  # (batch_len, 1) or (batch_len,)
                    acquisition_values_flat[i:end_idx] = batch_values.squeeze()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        smaller_batch_size = batch_size // 4
                        for j in range(i, end_idx, smaller_batch_size):
                            small_end_idx = min(j + smaller_batch_size, end_idx)
                            small_batch = points_flat[j:small_end_idx]
                            small_batch_values = acquisition_function(small_batch)
                            acquisition_values_flat[j:small_end_idx] = small_batch_values.squeeze()
                    else:
                        raise e

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Reshape to (k, num_points_per_line) and average over the line
        acquisition_values = acquisition_values_flat.reshape(k, self.num_points_per_line)  # (k, num_points_per_line)
        integrated = acquisition_values.mean(dim=1)  # (k,)

        return integrated

    def sampler(self, x_tell: torch.Tensor, bounds: torch.Tensor = None,
                acquisition_function: nn.Module = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points using LineBO: pick best line (by acquisition or random), evaluate objective, return requested/actual/y.

        Parameters
        ----------
        x_tell : torch.Tensor
            Starting point (d,) on simplex.
        bounds : torch.Tensor, optional
            Bounds tensor (2, d) with [lower, upper] bounds.
        acquisition_function : nn.Module, optional
            Acquisition function to optimize. If None, random selection.

        Returns
        -------
        tuple
            (x_requested, x_actual, y) tensors.
        """
        x_tell = x_tell.to(device=self.device, dtype=self.dtype)  # (d,)
        if bounds is not None:
            bounds = bounds.to(device=self.device, dtype=self.dtype)  # (2, d)

        assert abs(x_tell.sum().item() - 1.0) < 1e-12, f"x_tell must sum to 1, got {x_tell.sum().item()}"

        # Sample more directions than needed so we have enough valid (non-degenerate) segments
        directions = zero_sum_dirs(self.num_lines * 2, self.d,
                                 device=self.device, dtype=self.dtype)  # (num_lines*2, d)

        x_left, x_right, t_min, t_max, valid_mask = batch_line_simplex_segments(x_tell, directions)
        # x_left, x_right: (num_valid, d); t_min, t_max: (num_valid,); valid_mask: (num_lines*2,) bool

        # If we got fewer valid lines than num_lines, resample and append until we have enough (cap attempts)
        if x_left.shape[0] < self.num_lines:
            print(f"Warning: Only found {x_left.shape[0]} valid lines, needed {self.num_lines}")
            for _ in range(SAMPLER_MAX_EXTRA_ATTEMPTS):
                if x_left.shape[0] >= self.num_lines:
                    break
                additional_dirs = zero_sum_dirs(self.num_lines, self.d,
                                              device=self.device, dtype=self.dtype)  # (num_lines, d)
                x_left_add, x_right_add, t_min_add, t_max_add, mask_add = batch_line_simplex_segments(x_tell, additional_dirs)

                if x_left_add.shape[0] > 0:
                    x_left = torch.cat([x_left, x_left_add], dim=0)   # (?, d)
                    x_right = torch.cat([x_right, x_right_add], dim=0)
                    t_min = torch.cat([t_min, t_min_add], dim=0)
                    t_max = torch.cat([t_max, t_max_add], dim=0)
                else:
                    break

        # Trim to exactly num_lines if we have more
        if x_left.shape[0] > self.num_lines:
            x_left = x_left[:self.num_lines]   # (num_lines, d)
            x_right = x_right[:self.num_lines]
            t_min = t_min[:self.num_lines]
            t_max = t_max[:self.num_lines]

        # Pick best line: by max integrated acquisition, or random if no acquisition
        if acquisition_function is None:
            best_idx = torch.randint(0, x_left.shape[0], (1,), device=self.device).item()
        else:
            integrated_values = self._integrate_acquisition_along_lines(
                x_left, x_right, acquisition_function)  # (num_lines,)
            best_idx = torch.argmax(integrated_values).item()

        selected_left = x_left[best_idx]   # (d,)
        selected_right = x_right[best_idx]  # (d,)

        endpoints = torch.stack([selected_left, selected_right], dim=0)  # (2, d)

        x_actual, y = self.objective_function(endpoints)
        # x_actual: (n, d), y: (n,) or (n, 1)

        assert torch.is_tensor(x_actual) and x_actual.shape[1] == self.d
        assert torch.is_tensor(y) and y.shape[0] == x_actual.shape[0]

        x_actual = x_actual.to(device=self.device, dtype=self.dtype)  # (n, d)
        y = y.to(device=self.device, dtype=self.dtype)  # (n,) or (n, 1)

        # Build x_requested: points along the first principal direction of x_actual (or clone if single point)
        if x_actual.shape[0] > 1:
            x_centered = x_actual - x_actual.mean(dim=0, keepdim=True)  # (n, d)
            U, S, V = torch.svd(x_centered)  # U: (n, d), S: (d,), V: (d, d)
            direction = V[:, 0]  # (d,) first principal direction

            projections = torch.matmul(x_centered, direction.unsqueeze(1)).squeeze(1)  # (n,)

            t_vals = torch.linspace(projections.min(), projections.max(),
                                  x_actual.shape[0], device=self.device, dtype=torch.float64)  # (n,)

            x_requested = x_actual.mean(dim=0).unsqueeze(0) + t_vals.unsqueeze(1) * direction.unsqueeze(0)  # (n, d)
        else:
            x_requested = x_actual.clone()  # (1, d)

        return x_requested, x_actual, y
