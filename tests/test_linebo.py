"""Tests for LineBO utilities."""

import torch
import pytest
import sys
from pathlib import Path

# Add src/core to path for direct import (avoids loading botorch through __init__.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "core"))

from linebo import (
    zero_sum_dirs,
    line_simplex_segment,
    batch_line_simplex_segments,
)


class TestZeroSumDirs:
    """Tests for zero-sum direction generation."""

    def test_zero_sum(self):
        """Directions should sum to zero."""
        dirs = zero_sum_dirs(100, 5, device='cpu')
        sums = dirs.sum(dim=1)
        assert torch.allclose(sums, torch.zeros(100, dtype=dirs.dtype), atol=1e-10)

    def test_unit_norm(self):
        """Directions should have unit norm."""
        dirs = zero_sum_dirs(100, 5, device='cpu')
        norms = dirs.norm(dim=1)
        assert torch.allclose(norms, torch.ones(100, dtype=dirs.dtype), atol=1e-10)

    def test_correct_shape(self):
        """Should return correct shape."""
        dirs = zero_sum_dirs(50, 7, device='cpu')
        assert dirs.shape == (50, 7)

    def test_reproducibility(self):
        """Same seed should give same directions."""
        dirs1 = zero_sum_dirs(10, 5, device='cpu', seed=42)
        dirs2 = zero_sum_dirs(10, 5, device='cpu', seed=42)
        assert torch.allclose(dirs1, dirs2)


class TestLineSimplexSegment:
    """Tests for line-simplex intersection."""

    def test_interior_point(self):
        """Line from interior should intersect simplex."""
        x0 = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        d = torch.tensor([0.1, -0.1, 0.0], dtype=torch.float64)  # zero-sum direction

        result = line_simplex_segment(x0, d)
        assert result is not None

        t_min, t_max, x_left, x_right = result
        assert t_min <= 0 <= t_max

        # Check endpoints on simplex
        assert torch.allclose(x_left.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(x_right.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)

    def test_endpoints_nonnegative(self):
        """Endpoints should be non-negative."""
        x0 = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
        d = torch.tensor([0.1, -0.1, 0.05, -0.05], dtype=torch.float64)

        result = line_simplex_segment(x0, d)
        if result is not None:
            _, _, x_left, x_right = result
            assert (x_left >= -1e-10).all()
            assert (x_right >= -1e-10).all()


class TestBatchLineSimplexSegments:
    """Tests for batched line-simplex intersection."""

    def test_batch_processing(self):
        """Should process multiple directions."""
        x0 = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
        D = zero_sum_dirs(20, 4, device='cpu', dtype=torch.float64)

        x_left, x_right, t_min, t_max, mask = batch_line_simplex_segments(x0, D)

        # Should find some valid lines
        assert mask.sum() > 0
        assert x_left.shape[0] == mask.sum()

    def test_endpoints_valid(self):
        """All returned endpoints should be valid."""
        x0 = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        D = zero_sum_dirs(50, 3, device='cpu', dtype=torch.float64)

        x_left, x_right, t_min, t_max, mask = batch_line_simplex_segments(x0, D)

        # Check sum to 1
        assert torch.allclose(x_left.sum(dim=1), torch.ones(x_left.shape[0], dtype=torch.float64), atol=1e-10)
        assert torch.allclose(x_right.sum(dim=1), torch.ones(x_right.shape[0], dtype=torch.float64), atol=1e-10)

        # Check non-negative
        assert (x_left >= -1e-10).all()
        assert (x_right >= -1e-10).all()
