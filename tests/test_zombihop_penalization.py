"""
Tests for ZoMBIHop penalization strategy.

Evaluates:
- Penalty mask boundary behavior (inside / on / outside radius).
- Multiple needles and overlapping penalty balls.
- Penalty radius determination (GPSimplex.determine_penalty_radius).
- Consistency of get_best_unpenalized with penalty mask after adding needles.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.datahandler import DataHandler
from utils.simplex import proj_simplex


class TestPenaltyMaskBoundary:
    """Penalty mask should penalize points inside/on radius and not outside."""

    @pytest.fixture
    def handler(self):
        h = DataHandler(
            directory=None,
            max_saved_recent_checkpoints=0,
            device="cpu",
            dtype=torch.float64,
            d=3,
        )
        X = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.3, 0.4, 0.3],
                [0.2, 0.5, 0.3],
            ],
            dtype=torch.float64,
        )
        Y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        h.save_init(X, X, Y, bounds)
        return h

    def test_center_penalized(self, handler):
        """Point at needle center must be penalized."""
        needle = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        handler.add_needle(needle, 1.0, 0.1, 0, 0, 0)
        mask = handler.get_penalty_mask(needle.unsqueeze(0))
        assert not mask[0].item()

    def test_inside_radius_penalized(self, handler):
        """Point strictly inside penalty radius must be penalized."""
        needle = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        radius = 0.1
        handler.add_needle(needle, 1.0, radius, 0, 0, 0)
        direction = torch.tensor([-0.05, 0.02, 0.03], dtype=torch.float64)
        direction = direction / direction.norm() * (radius * 0.5)
        point = (needle + direction).unsqueeze(0)
        point = proj_simplex(point)
        mask = handler.get_penalty_mask(point)
        assert not mask[0].item()

    def test_on_radius_penalized(self, handler):
        """Point exactly at distance == radius is penalized (implementation uses <=)."""
        needle = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        radius = 0.1
        handler.add_needle(needle, 1.0, radius, 0, 0, 0)
        direction = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
        direction = direction / direction.norm() * radius
        point = (needle + direction).unsqueeze(0)
        point = proj_simplex(point)
        mask = handler.get_penalty_mask(point)
        assert not mask[0].item()

    def test_outside_radius_not_penalized(self, handler):
        """Point strictly outside penalty radius must not be penalized."""
        needle = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        radius = 0.05
        handler.add_needle(needle, 1.0, radius, 0, 0, 0)
        far = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float64)
        mask = handler.get_penalty_mask(far)
        assert mask[0].item()


class TestPenaltyMaskMultipleNeedles:
    """Multiple needles: point can be in zero, one, or multiple balls."""

    @pytest.fixture
    def handler_two_needles(self):
        h = DataHandler(
            directory=None,
            device="cpu",
            dtype=torch.float64,
            d=3,
        )
        X = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.3, 0.4, 0.3],
                [0.2, 0.5, 0.3],
                [0.4, 0.4, 0.2],
            ],
            dtype=torch.float64,
        )
        Y = torch.tensor([[1.0], [2.0], [3.0], [1.5]], dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        h.save_init(X, X, Y, bounds)
        h.add_needle(X[0], 1.0, 0.15, 0, 0, 0)
        h.add_needle(X[1], 2.0, 0.15, 0, 0, 0)
        return h

    def test_point_in_first_only_penalized(self, handler_two_needles):
        """Point in first needle ball only is penalized."""
        h = handler_two_needles
        needle0 = h.needles[0]
        point = (needle0 + torch.tensor([0.02, 0.0, -0.02], dtype=torch.float64)).unsqueeze(0)
        point = proj_simplex(point)
        mask = h.get_penalty_mask(point)
        assert not mask[0].item()

    def test_point_in_second_only_penalized(self, handler_two_needles):
        """Point in second needle ball only is penalized."""
        h = handler_two_needles
        needle1 = h.needles[1]
        point = (needle1 + torch.tensor([0.02, -0.01, -0.01], dtype=torch.float64)).unsqueeze(0)
        point = proj_simplex(point)
        mask = h.get_penalty_mask(point)
        assert not mask[0].item()

    def test_point_in_neither_not_penalized(self, handler_two_needles):
        """Point far from both needles is not penalized."""
        h = handler_two_needles
        point = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float64)
        mask = h.get_penalty_mask(point)
        assert mask[0].item()

    def test_cached_mask_after_add_needle(self, handler_two_needles):
        """Stored points: cached penalty mask should mark needle centers as penalized."""
        h = handler_two_needles
        mask = h.get_penalty_mask()
        assert not mask[0].item()
        assert not mask[1].item()
        assert mask.shape[0] == 4


class TestBestUnpenalizedConsistency:
    """get_best_unpenalized must ignore penalized points and return best among unpenalized."""

    def test_best_unpenalized_excludes_needle_region(self):
        """After adding a needle at the global best, best unpenalized is the next best."""
        h = DataHandler(directory=None, device="cpu", dtype=torch.float64, d=3)
        X = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.4, 0.4, 0.2],
                [0.3, 0.5, 0.2],
            ],
            dtype=torch.float64,
        )
        Y = torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        h.save_init(X, X, Y, bounds)
        best_X, best_Y, idx = h.get_best_unpenalized()
        assert idx == 1
        assert best_Y.item() == pytest.approx(5.0)

        h.add_needle(X[1], 5.0, 0.1, 0, 0, 0)
        best_X2, best_Y2, idx2 = h.get_best_unpenalized()
        assert idx2 == 2
        assert best_Y2.item() == pytest.approx(3.0)

    def test_best_unpenalized_all_penalized_returns_none(self):
        """When all points are penalized, get_best_unpenalized returns (None, None, None)."""
        h = DataHandler(directory=None, device="cpu", dtype=torch.float64, d=3)
        X = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]], dtype=torch.float64)
        Y = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        h.save_init(X, X, Y, bounds)
        h.add_needle(X[0], 1.0, 1.0, 0, 0, 0)
        h.add_needle(X[1], 2.0, 1.0, 0, 0, 0)
        mask = h.get_penalty_mask()
        assert not mask.any()
        best_X, best_Y, idx = h.get_best_unpenalized()
        assert best_X is None and best_Y is None and idx is None


class TestDeterminePenaltyRadius:
    """GPSimplex.determine_penalty_radius returns a radius in valid range and mask behaves."""

    @pytest.fixture
    def handler_with_gp_data(self):
        h = DataHandler(
            directory=None,
            device="cpu",
            dtype=torch.float64,
            d=3,
        )
        X = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.4, 0.4, 0.2],
                [0.3, 0.5, 0.2],
                [0.35, 0.45, 0.2],
                [0.45, 0.35, 0.2],
            ],
            dtype=torch.float64,
        )
        Y = torch.tensor([[1.0], [2.0], [1.5], [1.8], [1.9]], dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        h.save_init(X, X, Y, bounds)
        return h

    def test_determine_penalty_radius_in_range(self, handler_with_gp_data):
        """determine_penalty_radius returns a float in [min_radius, max_radius]."""
        pytest.importorskip("botorch")
        from utils.gp_simplex import GPSimplex

        h = handler_with_gp_data
        gp = GPSimplex(
            data_handler=h,
            device="cpu",
            dtype=torch.float64,
        )
        gp.fit(h.X_all_actual, h.Y_all)
        gp.create_acquisition(best_f=h.Y_all.max().item(), penalty_value=-1e6)

        needle = h.X_all_actual[0]
        max_radius = 0.3
        radius_step = 0.01
        r = gp.determine_penalty_radius(
            needle,
            penalization_threshold=1e-3,
            num_directions=20,
            max_radius=max_radius,
            radius_step=radius_step,
        )
        assert isinstance(r, float)
        assert r > 0
        assert r <= max_radius

    def test_penalty_radius_used_in_mask(self, handler_with_gp_data):
        """After determine_penalty_radius, adding needle with that radius penalizes center and nearby."""
        pytest.importorskip("botorch")
        from utils.gp_simplex import GPSimplex

        h = handler_with_gp_data
        gp = GPSimplex(
            data_handler=h,
            device="cpu",
            dtype=torch.float64,
        )
        gp.fit(h.X_all_actual, h.Y_all)
        gp.create_acquisition(best_f=h.Y_all.max().item(), penalty_value=-1e6)

        needle = h.X_all_actual[0].clone()
        r = gp.determine_penalty_radius(
            needle,
            penalization_threshold=1e-3,
            num_directions=10,
            max_radius=0.2,
            radius_step=0.02,
        )
        h.add_needle(needle, 2.0, r, 0, 0, 0)
        mask_at_center = h.get_penalty_mask(needle.unsqueeze(0))
        assert not mask_at_center[0].item()
        far = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float64)
        mask_far = h.get_penalty_mask(far)
        assert mask_far[0].item()
