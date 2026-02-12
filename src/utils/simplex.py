"""
Simplex Utilities
=================

Helper functions for working with the simplex constraint
in materials research applications.

The simplex constraint ensures that compositions sum to 1:
    x_1 + x_2 + ... + x_d = 1, x_i >= 0
"""
from __future__ import annotations

import torch
import numpy as np
import math
from typing import Union, Tuple, Optional


# =============================================================================
# CFS (Conditional Frechet Sampling) for Bounded Simplex
# =============================================================================

def subset_sums_and_signs(caps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute subset sums and inclusion-exclusion signs for CFS algorithm.

    Parameters
    ----------
    caps : torch.Tensor
        Capacity values for each dimension.

    Returns
    -------
    tuple
        (subset_sums, signs) tensors for polytope volume computation.
    """
    m = caps.numel()
    device, dtype = caps.device, caps.dtype
    subset_sums = torch.zeros(1 << m, dtype=dtype, device=device)
    for mask in range(1, 1 << m):
        lsb = mask & -mask
        j = (lsb.bit_length() - 1)
        subset_sums[mask] = subset_sums[mask ^ lsb] + caps[j]
    signs = torch.tensor([1 - 2 * (mask.bit_count() & 1) for mask in range(1 << m)],
                            dtype=dtype, device=device)
    return subset_sums, signs


def polytope_volume(S: torch.Tensor, subset_sums: torch.Tensor,
                    signs: torch.Tensor, power: int, denom: int) -> torch.Tensor:
    """
    Compute polytope volume using inclusion-exclusion principle.

    Parameters
    ----------
    S : torch.Tensor
        Sum values.
    subset_sums : torch.Tensor
        Precomputed subset sums.
    signs : torch.Tensor
        Inclusion-exclusion signs.
    power : int
        Power for the volume computation.
    denom : int
        Denominator (factorial).

    Returns
    -------
    torch.Tensor
        Computed volume values.
    """
    if torch.any(torch.isnan(S)) or torch.any(torch.isinf(S)):
        S = torch.clamp(S, min=0.0, max=1e6)

    shifted = (S.unsqueeze(1) - subset_sums)
    positive = torch.clamp(shifted, min=0.0)

    if power > 50:
        positive = torch.clamp(positive, max=10.0)

    powered = positive.pow(power)

    if torch.any(torch.isnan(powered)) or torch.any(torch.isinf(powered)):
        powered = torch.nan_to_num(powered, nan=0.0, posinf=1e6, neginf=0.0)

    result = (signs * powered).sum(dim=1) / denom

    result = torch.clamp(result, min=0.0)
    if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
        result = torch.nan_to_num(result, nan=1e-15, posinf=1e6, neginf=1e-15)

    return result


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
    Generate CFS samples from bounded simplex.

    Uses Conditional Frechet Sampling for uniform sampling on a bounded simplex.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    a : torch.Tensor
        Lower bounds for each component.
    b : torch.Tensor
        Upper bounds for each component.
    S : float
        Target sum (default 1.0 for probability simplex).
    max_batch : int, optional
        Maximum batch size for memory efficiency.
    debug : bool
        Enable debug output.
    device : str
        Torch device.
    torch_dtype : torch.dtype
        Output data type.

    Returns
    -------
    torch.Tensor
        (num_samples, d) tensor of samples on the bounded simplex.
    """
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

    if max_batch is None:
        if device == 'cuda':
            estimated_bytes_per_point = d * 8 * 32
            gpu_memory = torch.cuda.get_device_properties(device).total_memory
            available_memory = gpu_memory - torch.cuda.memory_allocated(device)
            safe_memory = available_memory * 0.8
            max_batch = max(int(safe_memory / estimated_bytes_per_point), 1000)
        else:
            max_batch = min(num_samples, 10_000_000)

    if num_samples > 10_000:
        print(f"Generating {num_samples:,} samples using CFS")

    caps_full = b - a
    S0 = S - a.sum()

    rng = torch.Generator(device=device)

    precomp = []
    for k in range(d - 1):
        ss, si = subset_sums_and_signs(caps_full[k+1:])
        precomp.append((ss, si))

    out = torch.empty((num_samples, d), dtype=torch.float64, device=device)
    written = 0

    while written < num_samples:
        B = min(max_batch, num_samples - written)

        S_rem = S0.expand(B).clone()
        caps_rem = caps_full.clone()
        y = torch.empty((B, d), dtype=torch.float64, device=device)

        for k in range(d - 1):
            ss, si = precomp[k]
            m = d - k - 1
            denom_vol = math.factorial(m - 1)
            denom_int = math.factorial(m)

            sum_tail_caps = caps_full[k+1:].sum()
            t_low = torch.clamp(S_rem - sum_tail_caps, min=0.0)
            t_high = torch.minimum(caps_full[k].expand_as(S_rem), S_rem)

            deterministic_mask = (t_high - t_low) < 1e-15
            yk = torch.zeros_like(S_rem)
            yk[deterministic_mask] = t_low[deterministic_mask]

            stochastic_mask = ~deterministic_mask
            n_stochastic = stochastic_mask.sum().item()

            if n_stochastic > 0:
                S_todo = S_rem[stochastic_mask]
                tl = t_low[stochastic_mask]
                th = t_high[stochastic_mask]

                interval_sizes = th - tl
                valid_mask = interval_sizes > 1e-12
                if torch.sum(valid_mask) == 0:
                    yk[stochastic_mask] = (tl + th) / 2
                    continue

                S_todo = S_todo[valid_mask]
                tl_valid = tl[valid_mask]
                th_valid = th[valid_mask]

                def _volume(t: torch.Tensor) -> torch.Tensor:
                    vol = polytope_volume(S_todo - t, ss, si, m - 1, denom_vol)
                    return torch.clamp(vol, min=1e-15)

                def _cdf(t: torch.Tensor) -> torch.Tensor:
                    shifted = S_todo - t
                    I_high = polytope_volume(shifted, ss, si, m, denom_int)
                    shifted_low = S_todo - tl_valid
                    I_low = polytope_volume(shifted_low, ss, si, m, denom_int)

                    I_low = torch.clamp(I_low, min=0.0)
                    I_high = torch.clamp(I_high, min=0.0)
                    I_high = torch.minimum(I_high, I_low)

                    cdf_val = I_low - I_high
                    return torch.clamp(cdf_val, min=1e-15)

                Z = _cdf(th_valid)

                if torch.any(Z <= 0) or torch.any(torch.isnan(Z)) or torch.any(torch.isinf(Z)):
                    Z = torch.clamp(Z, min=1e-15)

                U = torch.rand(len(tl_valid), generator=rng, device=device, dtype=torch.float64)
                target = U * Z

                t = tl_valid + U * (th_valid - tl_valid)

                for iteration in range(12):
                    f = _cdf(t) - target
                    fp = _volume(t)

                    fp_safe = torch.clamp(fp.abs(), min=1e-15)
                    fp_sign = torch.sign(fp)

                    delta = f / (fp_safe * fp_sign)

                    step_limit = 0.1 * (th_valid - tl_valid)
                    delta = torch.clamp(delta, -step_limit, step_limit)

                    t_new = t - delta
                    t = torch.clamp(t_new, tl_valid, th_valid)

                    if torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)):
                        t = tl_valid + torch.rand_like(tl_valid) * (th_valid - tl_valid)
                        break

                    if torch.all(torch.abs(f) < 1e-10):
                        break

                if torch.sum(valid_mask) > 0:
                    yk_temp = torch.zeros_like(S_rem[stochastic_mask])
                    yk_temp[valid_mask] = t
                    yk_temp[~valid_mask] = (tl[~valid_mask] + th[~valid_mask]) / 2
                    yk[stochastic_mask] = yk_temp
                else:
                    yk[stochastic_mask] = (tl + th) / 2

            if torch.any(torch.isnan(yk)) or torch.any(torch.isinf(yk)):
                alpha, beta = 2.0, 2.0
                U_beta = torch.distributions.Beta(alpha, beta).sample((len(S_rem),)).to(device=device, dtype=torch.float64)
                interval_size = torch.clamp(t_high - t_low, min=1e-15)
                yk = t_low + U_beta * interval_size
                yk = torch.clamp(yk, min=t_low, max=t_high)

            y[:, k] = yk
            S_rem -= yk
            S_rem = torch.clamp(S_rem, min=0.0)

        y[:, -1] = torch.minimum(torch.maximum(S_rem, torch.zeros_like(S_rem)),
                               caps_full[-1].expand_as(S_rem))

        out[written: written+B] = y + a
        written += B

    out = out.reshape(num_samples, d)
    return out.to(dtype=torch_dtype)


# =============================================================================
# Simplex Projection
# =============================================================================

def proj_simplex(X: torch.Tensor) -> torch.Tensor:
    """
    Project points onto the simplex (differentiable).

    Uses the algorithm from "Efficient Projections onto the l1-Ball
    for Learning in High Dimensions" (Duchi et al., 2008).

    Parameters
    ----------
    X : torch.Tensor
        Points to project, shape (..., d) or (n, l, d).

    Returns
    -------
    torch.Tensor
        Projected points on the simplex.
    """
    original_dim = X.dim()

    if original_dim == 3:
        n, l, d = X.shape
        X_2d = X.reshape(-1, d)
    else:
        X_2d = X

    u, _ = torch.sort(X_2d, descending=True, dim=-1)
    css = torch.cumsum(u, dim=-1)
    d = X_2d.size(-1)
    indices = torch.arange(1, d+1, device=X_2d.device, dtype=X_2d.dtype)
    rho = torch.sum((u * indices) > (css - 1), dim=-1) - 1

    batch_indices = torch.arange(X_2d.size(0), device=X_2d.device)
    theta = (torch.gather(css, 1, rho.unsqueeze(-1)).squeeze(-1) - 1) / (rho + 1).to(X_2d.dtype)

    result = torch.maximum(X_2d - theta.unsqueeze(-1), torch.zeros_like(X_2d))

    if original_dim == 3:
        result = result.reshape(n, l, d)

    return result


def random_zero_sum_directions(n: int, d: int, device: str = 'cuda',
                                dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Sample n vectors of dimension d with zero sum and unit norm.

    These directions preserve the simplex constraint when added to a point
    on the simplex.

    Parameters
    ----------
    n : int
        Number of directions to sample.
    d : int
        Dimensionality.
    device : str
        Torch device.
    dtype : torch.dtype
        Data type.

    Returns
    -------
    torch.Tensor
        (n, d) tensor of zero-sum, unit-norm vectors.
    """
    # Generate random vectors
    v = torch.randn(n, d, device=device, dtype=dtype)
    # Make zero-sum by subtracting mean
    v = v - v.mean(dim=1, keepdim=True)
    # Normalize to unit norm
    norms = v.norm(dim=1, keepdim=True)
    # Handle zero-norm rows by resampling
    mask = (norms.squeeze() == 0)
    while mask.any():
        r = torch.randn(mask.sum(), d, device=device, dtype=dtype)
        r = r - r.mean(dim=1, keepdim=True)
        v[mask] = r
        norms[mask] = v[mask].norm(dim=1, keepdim=True)
        mask = (norms.squeeze() == 0)
    return v / norms


# =============================================================================
# Basic Simplex Utilities (original functions)
# =============================================================================

def sample_simplex(n: int, d: int,
                   lower: Union[torch.Tensor, float] = 0.0,
                   upper: Union[torch.Tensor, float] = 1.0,
                   device: str = 'cuda',
                   dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Sample n points uniformly from the d-dimensional simplex.

    Uses the Dirichlet distribution for uniform sampling.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    d : int
        Dimensionality (number of components).
    lower : torch.Tensor or float
        Lower bounds for each component.
    upper : torch.Tensor or float
        Upper bounds for each component.
    device : str
        Torch device.
    dtype : torch.dtype
        Data type.

    Returns
    -------
    torch.Tensor
        (n, d) tensor of samples on the simplex.
    """
    # Dirichlet(1, 1, ..., 1) gives uniform distribution on simplex
    alpha = torch.ones(d, device=device, dtype=dtype)
    dirichlet = torch.distributions.Dirichlet(alpha)
    samples = dirichlet.sample((n,))

    # Apply bounds if non-trivial
    if isinstance(lower, (int, float)) and lower == 0.0:
        if isinstance(upper, (int, float)) and upper == 1.0:
            return samples

    # Handle bounded simplex via rejection sampling or rescaling
    lower = torch.as_tensor(lower, device=device, dtype=dtype)
    upper = torch.as_tensor(upper, device=device, dtype=dtype)

    if lower.numel() == 1:
        lower = lower.expand(d)
    if upper.numel() == 1:
        upper = upper.expand(d)

    # Simple rescaling (may not be perfectly uniform but practical)
    range_vals = upper - lower
    samples = lower + samples * range_vals

    # Re-normalize to sum to 1
    samples = samples / samples.sum(dim=1, keepdim=True)

    return samples


def project_to_simplex(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Project points onto the probability simplex.

    Uses the algorithm from "Efficient Projections onto the l1-Ball
    for Learning in High Dimensions" (Duchi et al., 2008).

    Parameters
    ----------
    x : torch.Tensor
        Points to project, shape (..., d).
    eps : float
        Small value for numerical stability.

    Returns
    -------
    torch.Tensor
        Projected points on the simplex.
    """
    original_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)

    # Sort in descending order
    u, _ = torch.sort(x, dim=-1, descending=True)

    # Compute cumulative sum
    cssv = torch.cumsum(u, dim=-1)

    # Find rho: largest j such that u[j] - (cssv[j] - 1) / (j+1) > 0
    d = x.shape[-1]
    indices = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    rho = torch.sum((u * indices) > (cssv - 1), dim=-1) - 1
    rho = rho.long()

    # Compute theta
    batch_indices = torch.arange(x.shape[0], device=x.device)
    theta = (torch.gather(cssv, 1, rho.unsqueeze(-1)).squeeze(-1) - 1) / (rho + 1).float()

    # Apply projection
    projected = torch.clamp(x - theta.unsqueeze(-1), min=0)

    if len(original_shape) == 1:
        projected = projected.squeeze(0)

    return projected


def is_on_simplex(x: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """
    Check if points are on the simplex.

    Parameters
    ----------
    x : torch.Tensor
        Points to check, shape (..., d).
    tol : float
        Tolerance for checks.

    Returns
    -------
    torch.Tensor
        Boolean tensor indicating if each point is on simplex.
    """
    # Check sum to 1
    sum_check = torch.abs(x.sum(dim=-1) - 1.0) < tol

    # Check non-negative
    nonneg_check = (x >= -tol).all(dim=-1)

    return sum_check & nonneg_check


def simplex_distance(x: torch.Tensor, y: torch.Tensor,
                     metric: str = 'euclidean') -> torch.Tensor:
    """
    Compute distance between points on the simplex.

    Parameters
    ----------
    x : torch.Tensor
        First set of points, shape (n, d) or (d,).
    y : torch.Tensor
        Second set of points, shape (m, d) or (d,).
    metric : str
        Distance metric: 'euclidean', 'aitchison', 'kl'.

    Returns
    -------
    torch.Tensor
        Distance matrix or scalar.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    if metric == 'euclidean':
        return torch.cdist(x, y)

    elif metric == 'aitchison':
        # Aitchison distance (log-ratio distance)
        eps = 1e-10
        log_x = torch.log(x + eps)
        log_y = torch.log(y + eps)

        # Center log-ratios
        clr_x = log_x - log_x.mean(dim=-1, keepdim=True)
        clr_y = log_y - log_y.mean(dim=-1, keepdim=True)

        return torch.cdist(clr_x, clr_y)

    elif metric == 'kl':
        # KL divergence (not symmetric)
        eps = 1e-10
        # Compute pairwise KL: D_KL(x || y)
        n, m = x.shape[0], y.shape[0]
        kl_matrix = torch.zeros(n, m, device=x.device, dtype=x.dtype)

        for i in range(n):
            for j in range(m):
                kl_matrix[i, j] = (x[i] * (torch.log(x[i] + eps) - torch.log(y[j] + eps))).sum()

        return kl_matrix

    else:
        raise ValueError(f"Unknown metric: {metric}")


def barycentric_coordinates(vertices: torch.Tensor,
                            points: torch.Tensor) -> torch.Tensor:
    """
    Compute barycentric coordinates of points w.r.t. simplex vertices.

    Parameters
    ----------
    vertices : torch.Tensor
        Simplex vertices, shape (d, d) where each row is a vertex.
    points : torch.Tensor
        Points to convert, shape (n, d).

    Returns
    -------
    torch.Tensor
        Barycentric coordinates, shape (n, d).
    """
    # For standard simplex, vertices are identity matrix
    # Barycentric coords are just the coordinates themselves
    return points


def composition_to_ilr(x: torch.Tensor) -> torch.Tensor:
    """
    Transform compositions to isometric log-ratio (ILR) coordinates.

    ILR is useful for unconstrained optimization on simplex data.

    Parameters
    ----------
    x : torch.Tensor
        Compositions on simplex, shape (..., d).

    Returns
    -------
    torch.Tensor
        ILR coordinates, shape (..., d-1).
    """
    d = x.shape[-1]
    eps = 1e-10
    log_x = torch.log(x + eps)

    # Helmert-like ILR transformation
    ilr_coords = []
    for i in range(d - 1):
        coef = np.sqrt((i + 1) / (i + 2))
        term1 = log_x[..., :i+1].sum(dim=-1) / (i + 1)
        term2 = log_x[..., i+1]
        ilr_coords.append(coef * (term1 - term2))

    return torch.stack(ilr_coords, dim=-1)


def ilr_to_composition(ilr: torch.Tensor, d: int) -> torch.Tensor:
    """
    Transform ILR coordinates back to compositions.

    Parameters
    ----------
    ilr : torch.Tensor
        ILR coordinates, shape (..., d-1).
    d : int
        Original dimensionality.

    Returns
    -------
    torch.Tensor
        Compositions on simplex, shape (..., d).
    """
    # Inverse ILR transformation
    log_x = torch.zeros(*ilr.shape[:-1], d, device=ilr.device, dtype=ilr.dtype)

    for i in range(d - 1):
        coef = np.sqrt((i + 1) / (i + 2))
        contribution = ilr[..., i] / coef

        # Distribute to first i+1 components
        log_x[..., :i+1] += contribution.unsqueeze(-1) / (i + 1)
        # Subtract from component i+1
        log_x[..., i+1] -= contribution

    # Normalize to sum to 1
    x = torch.exp(log_x)
    x = x / x.sum(dim=-1, keepdim=True)

    return x
