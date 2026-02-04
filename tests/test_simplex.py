"""Tests for simplex utilities."""

import torch
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src/utils to path for direct import (avoids loading botorch through __init__.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "utils"))

from simplex import (
    sample_simplex,
    project_to_simplex,
    is_on_simplex,
    simplex_distance,
)


class TestSampleSimplex:
    """Tests for simplex sampling."""

    def test_samples_on_simplex(self):
        """Samples should lie on the simplex."""
        samples = sample_simplex(100, 5, device='cpu')

        # Check sum to 1
        sums = samples.sum(dim=1)
        assert torch.allclose(sums, torch.ones(100, dtype=samples.dtype), atol=1e-6)

        # Check non-negative
        assert (samples >= -1e-10).all()

    def test_correct_shape(self):
        """Should return correct shape."""
        samples = sample_simplex(50, 7, device='cpu')
        assert samples.shape == (50, 7)

    def test_different_dimensions(self):
        """Should work for various dimensions."""
        for d in [2, 3, 5, 10]:
            samples = sample_simplex(10, d, device='cpu')
            assert samples.shape == (10, d)
            assert torch.allclose(samples.sum(dim=1), torch.ones(10, dtype=samples.dtype), atol=1e-6)


class TestProjectToSimplex:
    """Tests for simplex projection."""

    def test_already_on_simplex(self):
        """Points on simplex should not change much."""
        x = torch.tensor([0.2, 0.3, 0.5])
        projected = project_to_simplex(x)
        assert torch.allclose(x, projected, atol=1e-6)

    def test_projection_on_simplex(self):
        """Projected points should be on simplex."""
        x = torch.randn(10, 5)
        projected = project_to_simplex(x)

        assert torch.allclose(projected.sum(dim=1), torch.ones(10), atol=1e-6)
        assert (projected >= -1e-10).all()

    def test_negative_input(self):
        """Should handle negative inputs."""
        x = torch.tensor([-1.0, 2.0, 0.5])
        projected = project_to_simplex(x)

        assert torch.allclose(projected.sum(), torch.tensor(1.0), atol=1e-6)
        assert (projected >= -1e-10).all()


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
