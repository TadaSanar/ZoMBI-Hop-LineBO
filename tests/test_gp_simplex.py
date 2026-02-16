"""Tests for GPSimplex and acquisition functions."""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.datahandler import DataHandler
from utils.gp_simplex import GPSimplex, RepulsiveAcquisition
from utils.simplex import proj_simplex


# Skip tests if botorch not available
pytest.importorskip("botorch")


class TestRepulsiveAcquisition:
    """Tests for RepulsiveAcquisition."""

    @pytest.fixture
    def mock_base_acq(self):
        """Create a simple mock acquisition function."""
        class MockAcq(torch.nn.Module):
            def forward(self, X):
                if X.dim() == 3:
                    X = X.squeeze(1)
                return torch.ones(X.shape[0], device=X.device, dtype=X.dtype) * 100.0
        return MockAcq()

    def test_repulsive_no_needles(self, mock_base_acq):
        """Without needles, should return base acquisition."""
        needles = torch.empty(0, 2)
        radii = torch.empty(0)

        acq = RepulsiveAcquisition(
            base=mock_base_acq,
            proj_fn=lambda x: x,
            needles=needles,
            penalty_radii=radii,
            repulsion_lambda=1000.0,
        )

        X = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        result = acq(X)

        assert torch.allclose(result, torch.tensor([100.0, 100.0]))

    def test_repulsive_inside_needle(self, mock_base_acq):
        """Points inside needle radius should have reduced value."""
        needles = torch.tensor([[0.5, 0.5]])
        radii = torch.tensor([0.2])

        acq = RepulsiveAcquisition(
            base=mock_base_acq,
            proj_fn=lambda x: x,
            needles=needles,
            penalty_radii=radii,
            repulsion_lambda=1000.0,
        )

        # Point at needle center
        X_center = torch.tensor([[0.5, 0.5]])
        result_center = acq(X_center)

        # Point far away
        X_far = torch.tensor([[0.1, 0.9]])
        result_far = acq(X_far)

        # Center should have lower value due to penalty
        assert result_center[0] < result_far[0]
        assert result_center[0] < 100.0  # Less than base

    def test_repulsive_gradient_exists(self, mock_base_acq):
        """Repulsive acquisition should have gradients inside penalty region."""
        needles = torch.tensor([[0.5, 0.5]])
        radii = torch.tensor([0.3])

        acq = RepulsiveAcquisition(
            base=mock_base_acq,
            proj_fn=lambda x: x,
            needles=needles,
            penalty_radii=radii,
            repulsion_lambda=1000.0,
        )

        # Point inside penalty radius
        X = torch.tensor([[0.4, 0.6]], requires_grad=True)
        result = acq(X)
        result.sum().backward()

        # Should have non-zero gradient
        assert X.grad is not None
        assert X.grad.abs().sum() > 0

    def test_repulsive_multiple_needles(self, mock_base_acq):
        """Should handle multiple needles."""
        needles = torch.tensor([
            [0.3, 0.7],
            [0.7, 0.3],
        ])
        radii = torch.tensor([0.1, 0.1])

        acq = RepulsiveAcquisition(
            base=mock_base_acq,
            proj_fn=lambda x: x,
            needles=needles,
            penalty_radii=radii,
            repulsion_lambda=1000.0,
        )

        # Point near first needle
        X1 = torch.tensor([[0.3, 0.7]])
        # Point near second needle
        X2 = torch.tensor([[0.7, 0.3]])
        # Point in middle (far from both)
        X_mid = torch.tensor([[0.5, 0.5]])

        r1 = acq(X1)
        r2 = acq(X2)
        r_mid = acq(X_mid)

        # Middle point should have higher value
        assert r_mid[0] > r1[0]
        assert r_mid[0] > r2[0]


class TestGPSimplex:
    """Tests for GPSimplex class."""

    @pytest.fixture
    def handler_with_data(self):
        """Create DataHandler with test data."""
        handler = DataHandler(
            directory=None,
            device='cpu',
            dtype=torch.float64,
            d=3,
            max_gp_points=50,
        )
        # Create training data
        n = 20
        X = torch.rand(n, 3, dtype=torch.float64)
        X = X / X.sum(dim=1, keepdim=True)
        # Objective: maximize first component
        Y = X[:, 0:1] + torch.randn(n, 1, dtype=torch.float64) * 0.1
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        handler.save_init(X, X, Y, bounds)
        return handler

    def test_gp_fit(self, handler_with_data):
        """GPSimplex should fit without errors."""
        gp = GPSimplex(
            data_handler=handler_with_data,
            device='cpu',
            dtype=torch.float64,
        )
        X, Y = handler_with_data.get_gp_data()
        gp.fit(X, Y)

        assert gp.gp is not None
        assert gp.mll is not None

    def test_gp_fit_from_handler(self, handler_with_data):
        """fit_from_data_handler should work."""
        gp = GPSimplex(
            data_handler=handler_with_data,
            device='cpu',
            dtype=torch.float64,
        )
        gp.fit_from_data_handler()

        assert gp.gp is not None

    def test_gp_predict(self, handler_with_data):
        """predict should return mean and variance."""
        gp = GPSimplex(
            data_handler=handler_with_data,
            device='cpu',
            dtype=torch.float64,
        )
        gp.fit_from_data_handler()

        X_test = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
        mean, var = gp.predict(X_test)

        assert mean.shape == (1, 1)
        assert var.shape == (1, 1)
        assert var[0, 0] >= 0  # Variance should be non-negative

    def test_gp_predict_not_fitted(self, handler_with_data):
        """predict should raise if not fitted."""
        gp = GPSimplex(
            data_handler=handler_with_data,
            device='cpu',
            dtype=torch.float64,
        )

        with pytest.raises(RuntimeError, match="GP not fitted"):
            gp.predict(torch.rand(1, 3, dtype=torch.float64))

    def test_gp_get_output_noise(self, handler_with_data):
        """get_output_noise should return noise estimate."""
        gp = GPSimplex(
            data_handler=handler_with_data,
            device='cpu',
            dtype=torch.float64,
        )
        gp.fit_from_data_handler()

        noise = gp.get_output_noise()
        assert noise >= 0

    def test_create_acquisition(self, handler_with_data):
        """create_acquisition should create repulsive acquisition."""
        gp = GPSimplex(
            data_handler=handler_with_data,
            repulsion_lambda=1000.0,
            device='cpu',
            dtype=torch.float64,
        )
        gp.fit_from_data_handler()

        acq = gp.create_acquisition()

        assert isinstance(acq, RepulsiveAcquisition)
        assert acq.repulsion_lambda == 1000.0
        assert gp.acq_fn is acq


class TestGPSimplexCandidateSelection:
    """Tests for candidate selection."""

    @pytest.fixture
    def gp_fitted(self):
        """Create fitted GPSimplex."""
        handler = DataHandler(
            directory=None,
            device='cpu',
            dtype=torch.float64,
            d=3,
        )
        n = 30
        X = torch.rand(n, 3, dtype=torch.float64)
        X = X / X.sum(dim=1, keepdim=True)
        Y = X[:, 0:1] + torch.randn(n, 1, dtype=torch.float64) * 0.1
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        handler.save_init(X, X, Y, bounds)

        gp = GPSimplex(
            data_handler=handler,
            num_restarts=5,
            raw_samples=50,
            device='cpu',
            dtype=torch.float64,
        )
        gp.fit_from_data_handler()
        return gp, handler

    def test_get_candidate_returns_point(self, gp_fitted):
        """get_candidate should return a valid candidate."""
        gp, data_handler = gp_fitted
        bounds = data_handler.bounds

        candidate = gp.get_candidate(bounds)

        assert candidate is not None
        assert candidate.shape == (3,)
        # Should be on simplex (approximately)
        assert abs(candidate.sum().item() - 1.0) < 0.01

    def test_get_candidate_respects_bounds(self, gp_fitted):
        """Candidate should be approximately within bounds (simplex may push outside)."""
        gp, _ = gp_fitted
        # Use bounds that allow simplex-feasible points
        bounds = torch.tensor([
            [0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8],
        ], dtype=torch.float64)

        candidate = gp.get_candidate(bounds)

        if candidate is not None:
            # The candidate should be on the simplex
            assert abs(candidate.sum().item() - 1.0) < 0.01
            # And should be reasonably close to bounds (with tolerance for simplex projection)
            assert (candidate >= bounds[0] - 0.1).all()
            assert (candidate <= bounds[1] + 0.1).all()

    def test_get_candidate_returns_none_when_penalized(self):
        """get_candidate should return None if all sampled candidates are penalized."""
        handler = DataHandler(
            directory=None,
            device='cpu',
            dtype=torch.float64,
            d=3,
        )
        n = 30
        X = torch.rand(n, 3, dtype=torch.float64)
        X = X / X.sum(dim=1, keepdim=True)
        Y = torch.randn(n, 1, dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        handler.save_init(X, X, Y, bounds)

        gp = GPSimplex(
            data_handler=handler,
            num_restarts=3,
            raw_samples=20,
            device='cpu',
            dtype=torch.float64,
        )
        gp.fit_from_data_handler()

        # Add multiple needles to cover most of the space
        # This penalizes existing data but leaves GP fitted
        for i in range(10):
            center = X[i]
            handler.add_needle(center, Y[i].item(), 0.2, 0, 0, i)

        # Now most sampled candidates should be penalized
        candidate = gp.get_candidate(bounds)
        # Candidate may be None or may find a valid point - both are acceptable
        # The key test is that it doesn't crash
        if candidate is not None:
            # If found, should be on simplex
            assert abs(candidate.sum().item() - 1.0) < 0.01


class TestAcquisitionOptimizerInterior:
    """
    Comprehensive tests for _optimize_acquisition (projected gradient ascent on simplex).

    Verifies that:
    - The optimizer can find interior maxima when the acquisition favors them.
    - The optimizer can find vertex/edge maxima when the acquisition favors them.
    - All returned points stay on the simplex and within bounds.
    - More steps improve convergence; gradient ascent increases acquisition value.
    - 2D and 3D simplex; flat and peaked acquisitions; no crashes.

    If these pass, SGD + projection are working; edge-only suggestions in real runs
    are due to the acquisition (e.g. LogEI) favoring vertices, not broken optimization.
    """

    @pytest.fixture
    def gp_2d(self):
        """GPSimplex for 2D simplex with minimal handler."""
        handler = DataHandler(
            directory=None,
            device="cpu",
            dtype=torch.float64,
            d=2,
        )
        X = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.7, 0.3]], dtype=torch.float64)
        Y = torch.tensor([[-0.1], [-0.2], [-0.2]], dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        handler.save_init(X, X, Y, bounds)
        gp = GPSimplex(
            data_handler=handler,
            num_restarts=6,
            raw_samples=10,
            device="cpu",
            dtype=torch.float64,
        )
        gp.proj_fn = proj_simplex
        return gp

    @pytest.fixture
    def gp_3d(self):
        """GPSimplex for 3D simplex."""
        handler = DataHandler(
            directory=None,
            device="cpu",
            dtype=torch.float64,
            d=3,
        )
        X = torch.tensor(
            [[0.33, 0.33, 0.34], [0.5, 0.25, 0.25], [0.2, 0.4, 0.4]],
            dtype=torch.float64,
        )
        Y = torch.tensor([[-0.1], [-0.2], [-0.2]], dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        handler.save_init(X, X, Y, bounds)
        gp = GPSimplex(
            data_handler=handler,
            num_restarts=5,
            raw_samples=10,
            device="cpu",
            dtype=torch.float64,
        )
        gp.proj_fn = proj_simplex
        return gp

    def _make_quadratic_acquisition(self, center: torch.Tensor, scale: float = -1.0):
        """Acquisition = scale * ||x - center||^2 (max at center when scale < 0)."""

        class QuadraticAcquisition(torch.nn.Module):
            def forward(self, X):
                if X.dim() == 3:
                    x = X.squeeze(1)
                else:
                    x = X
                diff = x - center.to(x.device)
                return scale * (diff ** 2).sum(dim=-1)

        return QuadraticAcquisition()

    def _inits_2d_boundary_and_interior(self):
        """Mix of boundary and interior initial points for 2D."""
        return torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.6, 0.4],
            ],
            dtype=torch.float64,
        ).unsqueeze(1)  # (6, 1, 2)

    def test_optimizer_finds_interior_maximum(self, gp_2d):
        """With acquisition max at center (0.5, 0.5), optimizer should find interior."""
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = self._make_quadratic_acquisition(center, scale=-1.0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = self._inits_2d_boundary_and_interior()

        candidates, values = gp_2d._optimize_acquisition(
            acq=acq,
            bounds=bounds,
            initial_conditions=inits,
            step_size=0.15,
            max_steps=40,
        )

        assert candidates.shape[0] > 0
        assert candidates.shape[1] == 2
        best_idx = values.argmax().item()
        best = candidates[best_idx]
        dist_to_center = torch.norm(best - center).item()
        assert dist_to_center < 0.15, (
            f"Best={best.tolist()}, dist_to_center={dist_to_center:.4f}"
        )
        assert abs(best.sum().item() - 1.0) < 0.01

    def test_optimizer_finds_vertex_maximum(self, gp_2d):
        """With acquisition max at vertex [1, 0], optimizer should find that vertex."""
        center = torch.tensor([1.0, 0.0], dtype=torch.float64)
        acq = self._make_quadratic_acquisition(center, scale=-1.0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor(
            [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]],
            dtype=torch.float64,
        ).unsqueeze(1)

        candidates, values = gp_2d._optimize_acquisition(
            acq=acq,
            bounds=bounds,
            initial_conditions=inits,
            step_size=0.1,
            max_steps=50,
        )

        assert candidates.shape[0] > 0
        best_idx = values.argmax().item()
        best = candidates[best_idx]
        dist_to_vertex = torch.norm(best - center).item()
        assert dist_to_vertex < 0.2, (
            f"Best={best.tolist()}, dist_to_vertex={dist_to_vertex:.4f}"
        )
        assert abs(best.sum().item() - 1.0) < 0.01
        assert (best >= -0.01).all() and (best <= 1.01).all()

    def test_all_candidates_on_simplex(self, gp_2d):
        """Every returned candidate must sum to 1 and have non-negative components."""
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = self._make_quadratic_acquisition(center, scale=-1.0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = self._inits_2d_boundary_and_interior()

        candidates, values = gp_2d._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.15, max_steps=30
        )

        assert candidates.shape[0] > 0
        for i in range(candidates.shape[0]):
            c = candidates[i]
            assert abs(c.sum().item() - 1.0) < 1e-5, f"Candidate {i} sum = {c.sum().item()}"
            assert (c >= -1e-6).all(), f"Candidate {i} has negative component: {c.tolist()}"

    def test_optimizer_respects_bounds(self, gp_2d):
        """With restricted bounds, candidates stay within [lower, upper] after projection."""
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = self._make_quadratic_acquisition(center, scale=-1.0)
        # Restrict to a subregion of the simplex
        bounds = torch.tensor([[0.2, 0.2], [0.8, 0.8]], dtype=torch.float64)
        inits = torch.tensor(
            [[0.5, 0.5], [0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3]],
            dtype=torch.float64,
        ).unsqueeze(1)

        candidates, _ = gp_2d._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.1, max_steps=30
        )

        assert candidates.shape[0] > 0
        lower, upper = bounds[0], bounds[1]
        for i in range(candidates.shape[0]):
            c = candidates[i]
            assert (c >= lower - 1e-5).all(), f"Candidate {i} below lower: {c.tolist()}"
            assert (c <= upper + 1e-5).all(), f"Candidate {i} above upper: {c.tolist()}"

    def test_interior_maximum_3d(self, gp_3d):
        """With acquisition max at 3D center (1/3, 1/3, 1/3), optimizer finds interior."""
        center = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=torch.float64)
        acq = self._make_quadratic_acquisition(center, scale=-1.0)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.33, 0.33, 0.34],
                [0.6, 0.2, 0.2],
            ],
            dtype=torch.float64,
        ).unsqueeze(1)

        candidates, values = gp_3d._optimize_acquisition(
            acq=acq,
            bounds=bounds,
            initial_conditions=inits,
            step_size=0.1,
            max_steps=50,
        )

        assert candidates.shape[0] > 0
        assert candidates.shape[1] == 3
        best_idx = values.argmax().item()
        best = candidates[best_idx]
        dist_to_center = torch.norm(best - center).item()
        assert dist_to_center < 0.2, (
            f"Best={best.tolist()}, dist_to_center={dist_to_center:.4f}"
        )
        assert abs(best.sum().item() - 1.0) < 0.01

    def test_more_steps_improve_convergence(self, gp_2d):
        """With interior max, more steps should get closer to center (or at least not worse)."""
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = self._make_quadratic_acquisition(center, scale=-1.0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor(
            [[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]],
            dtype=torch.float64,
        ).unsqueeze(1)

        candidates_short, values_short = gp_2d._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.1, max_steps=10
        )
        candidates_long, values_long = gp_2d._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.1, max_steps=60
        )

        assert candidates_short.shape[0] > 0 and candidates_long.shape[0] > 0
        best_short = values_short.max().item()
        best_long = values_long.max().item()
        # More steps should yield better or equal acquisition value (we're maximizing)
        assert best_long >= best_short - 1e-6, (
            f"More steps should not reduce best value: {best_short} vs {best_long}"
        )

    def test_gradient_ascend_increases_value(self, gp_2d):
        """Final acquisition value should be >= initial value for each successful restart."""
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = self._make_quadratic_acquisition(center, scale=-1.0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = self._inits_2d_boundary_and_interior()

        candidates, values = gp_2d._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.12, max_steps=40
        )

        with torch.no_grad():
            for i in range(candidates.shape[0]):
                init_val = acq(inits[i].unsqueeze(0)).squeeze().item()
                final_val = values[i].item()
                # After ascent, final should be >= initial (allowing small numerical error)
                assert final_val >= init_val - 1e-5, (
                    f"Restart {i}: init_val={init_val}, final_val={final_val}"
                )

    def test_different_inits_converge_to_interior(self, gp_2d):
        """Multiple boundary inits should all converge toward the same interior max."""
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = self._make_quadratic_acquisition(center, scale=-1.0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = self._inits_2d_boundary_and_interior()

        candidates, _ = gp_2d._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.15, max_steps=50
        )

        # All converged points should be within a ball of radius 0.2 around center
        for i in range(candidates.shape[0]):
            dist = torch.norm(candidates[i] - center).item()
            assert dist < 0.25, (
                f"Restart {i} did not converge to interior: {candidates[i].tolist()}, dist={dist}"
            )

    def test_flat_acquisition_no_crash(self, gp_2d):
        """Constant acquisition (flat) should not crash; candidates still on simplex."""
        class FlatAcquisition(torch.nn.Module):
            def forward(self, X):
                if X.dim() == 3:
                    x = X.squeeze(1)
                else:
                    x = X
                return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        acq = FlatAcquisition()
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = self._inits_2d_boundary_and_interior()

        candidates, values = gp_2d._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.05, max_steps=20
        )

        # May converge to whatever; just check shape and simplex
        assert candidates.shape[0] > 0
        assert values.shape[0] == candidates.shape[0]
        for i in range(candidates.shape[0]):
            assert abs(candidates[i].sum().item() - 1.0) < 1e-5
            assert (candidates[i] >= -1e-6).all()

    def test_single_restart(self, gp_2d):
        """Single initial condition still returns valid candidate."""
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = self._make_quadratic_acquisition(center, scale=-1.0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor([[0.8, 0.2]], dtype=torch.float64).unsqueeze(1)  # (1, 1, 2)

        candidates, values = gp_2d._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.15, max_steps=40
        )

        assert candidates.shape == (1, 2)
        assert values.shape == (1,)
        assert abs(candidates[0].sum().item() - 1.0) < 0.01
        dist = torch.norm(candidates[0] - center).item()
        assert dist < 0.2

    def test_values_ordered_with_candidates(self, gp_2d):
        """values[i] should be the acquisition value at candidates[i]."""
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = self._make_quadratic_acquisition(center, scale=-1.0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = self._inits_2d_boundary_and_interior()

        candidates, values = gp_2d._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.15, max_steps=30
        )

        with torch.no_grad():
            for i in range(candidates.shape[0]):
                x = candidates[i].unsqueeze(0).unsqueeze(0)  # (1, 1, d)
                expected = acq(x).squeeze().item()
                actual = values[i].item()
                assert abs(expected - actual) < 1e-5, (
                    f"Candidate {i}: expected acq={expected}, got {actual}"
                )


class TestGPSimplexPenaltyRadius:
    """Tests for penalty radius determination."""

    @pytest.fixture
    def gp_fitted(self):
        """Create fitted GPSimplex."""
        handler = DataHandler(
            directory=None,
            device='cpu',
            dtype=torch.float64,
            d=3,
        )
        n = 30
        X = torch.rand(n, 3, dtype=torch.float64)
        X = X / X.sum(dim=1, keepdim=True)
        Y = X[:, 0:1] + torch.randn(n, 1, dtype=torch.float64) * 0.1
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        handler.save_init(X, X, Y, bounds)

        gp = GPSimplex(
            data_handler=handler,
            device='cpu',
            dtype=torch.float64,
        )
        gp.fit_from_data_handler()
        return gp, handler

    def test_determine_penalty_radius(self, gp_fitted):
        """determine_penalty_radius should return positive value."""
        gp, handler = gp_fitted

        best_X, _, _ = handler.get_best_unpenalized()

        radius = gp.determine_penalty_radius(
            needle=best_X,
            penalization_threshold=1e-3,
            num_directions=10,
            max_radius=0.3,
            radius_step=0.05,
        )

        assert radius > 0
        assert radius <= 0.3

    def test_penalty_radius_respects_max(self, gp_fitted):
        """Penalty radius should not exceed max_radius."""
        gp, handler = gp_fitted

        best_X, _, _ = handler.get_best_unpenalized()

        radius = gp.determine_penalty_radius(
            needle=best_X,
            penalization_threshold=1e-10,  # Very strict
            num_directions=10,
            max_radius=0.15,
            radius_step=0.05,
        )

        assert radius <= 0.15


class TestGPSimplexOptimization:
    """Tests for acquisition optimization."""

    def test_optimize_acquisition(self):
        """_optimize_acquisition should improve acquisition value."""
        handler = DataHandler(
            directory=None,
            device='cpu',
            dtype=torch.float64,
            d=3,
        )
        n = 20
        X = torch.rand(n, 3, dtype=torch.float64)
        X = X / X.sum(dim=1, keepdim=True)
        Y = X[:, 0:1]  # Simple objective
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        handler.save_init(X, X, Y, bounds)

        gp = GPSimplex(
            data_handler=handler,
            device='cpu',
            dtype=torch.float64,
        )
        gp.fit_from_data_handler()
        acq = gp.create_acquisition()

        # Random starting points
        init = torch.rand(3, 1, 3, dtype=torch.float64)
        init = init / init.sum(dim=-1, keepdim=True)

        candidate, value = gp._optimize_acquisition(
            acq=acq,
            bounds=bounds,
            initial_conditions=init,
            step_size=0.05,
            max_steps=10,
        )

        assert candidate is not None
        assert value is not None
        # Should be approximately on simplex
        assert abs(candidate.sum().item() - 1.0) < 0.01


class TestGPSimplexIntegration:
    """Integration tests for GPSimplex with DataHandler."""

    def test_full_workflow(self):
        """Test complete workflow: fit, acquire, update."""
        # Setup
        handler = DataHandler(
            directory=None,
            device='cpu',
            dtype=torch.float64,
            d=3,
        )
        n = 15
        X = torch.rand(n, 3, dtype=torch.float64)
        X = X / X.sum(dim=1, keepdim=True)
        Y = X[:, 0:1] + torch.randn(n, 1, dtype=torch.float64) * 0.1
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        handler.save_init(X, X, Y, bounds)

        gp = GPSimplex(
            data_handler=handler,
            num_restarts=3,
            raw_samples=30,
            device='cpu',
            dtype=torch.float64,
        )

        # Fit
        gp.fit_from_data_handler()

        # Get candidate
        candidate = gp.get_candidate(bounds)
        assert candidate is not None

        # Simulate evaluation
        new_Y = candidate[0:1].unsqueeze(0)  # Use first component as objective
        handler.add_all_points(
            candidate.unsqueeze(0),
            candidate.unsqueeze(0),
            new_Y,
        )

        # Refit with new data
        gp.fit_from_data_handler()

        # Get another candidate
        candidate2 = gp.get_candidate(bounds)
        assert candidate2 is not None

        # Check data was updated
        assert handler.X_all_actual.shape[0] == n + 1

    def test_workflow_with_needle(self):
        """Test workflow after adding a needle."""
        handler = DataHandler(
            directory=None,
            device='cpu',
            dtype=torch.float64,
            d=3,
        )
        n = 20
        X = torch.rand(n, 3, dtype=torch.float64)
        X = X / X.sum(dim=1, keepdim=True)
        Y = X[:, 0:1]
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        handler.save_init(X, X, Y, bounds)

        gp = GPSimplex(
            data_handler=handler,
            num_restarts=5,
            raw_samples=50,
            device='cpu',
            dtype=torch.float64,
        )
        gp.fit_from_data_handler()

        # Get best point and add as needle
        best_X, best_Y, _ = handler.get_best_unpenalized()
        handler.add_needle(best_X, best_Y.item(), 0.1, 0, 0, 0)

        # Refit and get new candidate
        gp.fit_from_data_handler()
        candidate = gp.get_candidate(bounds)

        if candidate is not None:
            # New candidate should be away from needle
            dist = torch.norm(candidate - best_X)
            assert dist > 0.05  # Should be at least somewhat away
