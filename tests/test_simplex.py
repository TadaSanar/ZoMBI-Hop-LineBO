"""Tests for simplex utilities."""

import torch
import sys
from pathlib import Path

# Add src/utils to path for direct import (avoids loading botorch through __init__.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "utils"))

from simplex import (
    random_simplex,
    is_on_simplex,
    simplex_distance,
)


class TestRandomSimplex:
    """Tests for bounded simplex CFS sampler."""

    def test_samples_on_full_simplex(self):
        d = 5
        lo = torch.zeros(d, dtype=torch.float64)
        hi = torch.ones(d, dtype=torch.float64)
        samples = random_simplex(200, lo, hi, device='cpu', torch_dtype=torch.float64)
        assert samples.shape == (200, d)
        assert torch.allclose(samples.sum(dim=1), torch.ones(200, dtype=samples.dtype), atol=1e-9)
        assert (samples >= -1e-10).all()

    def test_samples_respect_bounds(self):
        lo = torch.tensor([0.05, 0.10, 0.05, 0.10], dtype=torch.float64)
        hi = torch.tensor([0.60, 0.70, 0.55, 0.70], dtype=torch.float64)
        samples = random_simplex(300, lo, hi, device='cpu', torch_dtype=torch.float64)
        assert (samples >= lo.unsqueeze(0) - 1e-10).all()
        assert (samples <= hi.unsqueeze(0) + 1e-10).all()
        assert torch.allclose(samples.sum(dim=1), torch.ones(300, dtype=samples.dtype), atol=1e-9)


class TestIsOnSimplex:
    """Tests for simplex membership check."""

    def test_valid_simplex_point(self):
        """Valid simplex points should return True."""
        x = torch.tensor([0.2, 0.3, 0.5])
        assert is_on_simplex(x).item()

    def test_invalid_sum(self):
        """Points not summing to 1 should return False."""
        x = torch.tensor([0.2, 0.3, 0.4])  # sums to 0.9
        assert not is_on_simplex(x).item()

    def test_negative_component(self):
        """Points with negative components should return False."""
        x = torch.tensor([-0.1, 0.6, 0.5])
        assert not is_on_simplex(x).item()

    def test_batch_check(self):
        """Should work on batches."""
        x = torch.tensor([
            [0.2, 0.3, 0.5],  # Valid
            [0.5, 0.5, 0.5],  # Invalid (sum != 1)
            [0.0, 1.0, 0.0],  # Valid (corner)
        ])
        result = is_on_simplex(x)
        expected = torch.tensor([True, False, True])
        assert torch.equal(result, expected)


class TestSimplexDistance:
    """Tests for distance computations."""

    def test_zero_distance(self):
        """Distance to self should be zero."""
        x = torch.tensor([[0.2, 0.3, 0.5]])
        dist = simplex_distance(x, x, metric='euclidean')
        assert torch.allclose(dist, torch.zeros(1, 1), atol=1e-10)

    def test_euclidean_symmetric(self):
        """Euclidean distance should be symmetric."""
        x = torch.tensor([[0.2, 0.3, 0.5]])
        y = torch.tensor([[0.1, 0.4, 0.5]])

        d_xy = simplex_distance(x, y, metric='euclidean')
        d_yx = simplex_distance(y, x, metric='euclidean')

        assert torch.allclose(d_xy, d_yx.T)

    def test_aitchison_distance(self):
        """Test Aitchison (log-ratio) distance."""
        x = torch.tensor([[0.2, 0.3, 0.5]])
        y = torch.tensor([[0.1, 0.4, 0.5]])

        dist = simplex_distance(x, y, metric='aitchison')
        assert dist.shape == (1, 1)
        assert dist[0, 0] > 0
