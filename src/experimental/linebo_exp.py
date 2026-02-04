"""
LineBO: Line-based Bayesian Optimization for Simplex Constraints
================================================================

A simplified line-based Bayesian optimization algorithm for
simplex-constrained problems in materials research.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


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
    v = torch.randn(k, d, device=device, dtype=dtype)
    v -= v.mean(dim=1, keepdim=True)  # sum=0
    n = v.norm(dim=1, keepdim=True)
    # Resample zero-norm rows
    mask = (n.squeeze() == 0)
    while mask.any():
        r = torch.randn(mask.sum(), d, device=device, dtype=dtype)
        r -= r.mean(dim=1, keepdim=True)
        v[mask] = r
        n[mask] = v[mask].norm(dim=1, keepdim=True)
        mask = (n.squeeze() == 0)
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
    neg = d < 0
    pos = d > 0

    inf = torch.tensor(float('inf'), device=d.device, dtype=d.dtype)
    ninf = torch.tensor(float('-inf'), device=d.device, dtype=d.dtype)

    t_max = torch.min(torch.where(neg, -x0 / d, inf))
    t_min = torch.max(torch.where(pos, -x0 / d, ninf))

    if t_min > t_max:
        return None

    x_left = x0 + t_min * d
    x_right = x0 + t_max * d
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
    neg = D < 0
    pos = D > 0

    inf = torch.tensor(float('inf'), device=D.device, dtype=D.dtype)
    ninf = torch.tensor(float('-inf'), device=D.device, dtype=D.dtype)

    t_max = torch.min(torch.where(neg, -x0.unsqueeze(0)/D, inf), dim=1).values
    t_min = torch.max(torch.where(pos, -x0.unsqueeze(0)/D, ninf), dim=1).values

    mask = t_min <= t_max
    t_min = t_min[mask]
    t_max = t_max[mask]
    Dm = D[mask]

    x_left = x0.unsqueeze(0) + t_min.unsqueeze(1) * Dm
    x_right = x0.unsqueeze(0) + t_max.unsqueeze(1) * Dm

    return x_left, x_right, t_min, t_max, mask


class LineBO:
    """
    Line-based Bayesian Optimization for simplex-constrained problems.

    Uses float64 throughout for numerical stability. Optimizes along
    1D line segments within the simplex, integrating acquisition functions
    to select the most promising direction.

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
        """Initialize LineBO sampler."""
        self.objective_function = objective_function
        self.d = dimensions
        self.num_points_per_line = num_points_per_line
        # Auto-compute num_lines as 10*d if not specified
        self.num_lines = num_lines if num_lines is not None else 10 * dimensions
        self.device = torch.device(device)
        self.dtype = torch.float64

    def _integrate_acquisition_along_lines(self, x_left: torch.Tensor, x_right: torch.Tensor,
                                         acquisition_function: nn.Module) -> torch.Tensor:
        """
        Integrate acquisition function along multiple line segments.

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
            (k,) integrated acquisition values.
        """
        k = x_left.shape[0]

        t_values = torch.linspace(0, 1, self.num_points_per_line,
                                device=self.device, dtype=torch.float64)

        points = (x_left.unsqueeze(1) * (1 - t_values).view(1, -1, 1) +
                 x_right.unsqueeze(1) * t_values.view(1, -1, 1))

        points_flat = points.reshape(-1, 1, self.d)

        batch_size = 500
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

        acquisition_values = acquisition_values_flat.reshape(k, self.num_points_per_line)
        integrated = acquisition_values.mean(dim=1)

        return integrated

    def sampler(self, x_tell: torch.Tensor, bounds: torch.Tensor = None,
                acquisition_function: nn.Module = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points using LineBO algorithm.

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
        x_tell = x_tell.to(device=self.device, dtype=self.dtype)
        if bounds is not None:
            bounds = bounds.to(device=self.device, dtype=self.dtype)

        assert abs(x_tell.sum().item() - 1.0) < 1e-12, f"x_tell must sum to 1, got {x_tell.sum().item()}"

        directions = zero_sum_dirs(self.num_lines * 2, self.d,
                                 device=self.device, dtype=self.dtype)

        x_left, x_right, t_min, t_max, valid_mask = batch_line_simplex_segments(x_tell, directions)

        if x_left.shape[0] < self.num_lines:
            print(f"Warning: Only found {x_left.shape[0]} valid lines, needed {self.num_lines}")
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

        if x_left.shape[0] > self.num_lines:
            x_left = x_left[:self.num_lines]
            x_right = x_right[:self.num_lines]
            t_min = t_min[:self.num_lines]
            t_max = t_max[:self.num_lines]

        if acquisition_function is None:
            best_idx = torch.randint(0, x_left.shape[0], (1,), device=self.device).item()
        else:
            integrated_values = self._integrate_acquisition_along_lines(
                x_left, x_right, acquisition_function)
            best_idx = torch.argmax(integrated_values).item()

        selected_left = x_left[best_idx]
        selected_right = x_right[best_idx]

        endpoints = torch.stack([selected_left, selected_right], dim=0)

        x_actual, y = self.objective_function(endpoints, bounds)

        assert torch.is_tensor(x_actual) and x_actual.shape[1] == self.d
        assert torch.is_tensor(y) and y.shape[0] == x_actual.shape[0]

        x_actual = x_actual.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        if x_actual.shape[0] > 1:
            x_centered = x_actual - x_actual.mean(dim=0, keepdim=True)
            U, S, V = torch.svd(x_centered)
            direction = V[:, 0]

            projections = torch.matmul(x_centered, direction.unsqueeze(1)).squeeze(1)

            t_vals = torch.linspace(projections.min(), projections.max(),
                                  x_actual.shape[0], device=self.device, dtype=torch.float64)

            x_requested = x_actual.mean(dim=0).unsqueeze(0) + t_vals.unsqueeze(1) * direction.unsqueeze(0)
        else:
            x_requested = x_actual.clone()

        return x_requested, x_actual, y
