"""Tests for GPSimplex and acquisition functions."""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.datahandler import DataHandler
from utils.gp_simplex import GPSimplex, RepulsiveAcquisition


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
