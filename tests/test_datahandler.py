"""Tests for DataHandler."""

import torch
import pytest
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.datahandler import DataHandler


class TestDataHandlerInit:
    """Tests for DataHandler initialization."""

    def test_init_no_save_mode(self):
        """DataHandler should work without saving."""
        handler = DataHandler(
            directory=None,
            max_saved_recent_checkpoints=0,
            device='cpu',
            dtype=torch.float64,
            d=5,
        )
        assert handler.save_enabled is False
        assert handler.run_uuid is not None

    def test_init_with_save_mode(self):
        """DataHandler should create directory when saving enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = DataHandler(
                directory=tmpdir,
                max_saved_recent_checkpoints=10,
                device='cpu',
                dtype=torch.float64,
                d=5,
            )
            assert handler.save_enabled is True
            assert handler.run_dir.exists()

    def test_init_generates_uuid(self):
        """DataHandler should generate unique UUIDs."""
        handler1 = DataHandler(directory=None, device='cpu', d=5)
        handler2 = DataHandler(directory=None, device='cpu', d=5)
        assert handler1.run_uuid != handler2.run_uuid


class TestDataHandlerStorage:
    """Tests for DataHandler data storage."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        return DataHandler(
            directory=None,
            max_saved_recent_checkpoints=0,
            device='cpu',
            dtype=torch.float64,
            d=5,
            top_m_points=4,
            max_gp_points=100,
        )

    @pytest.fixture
    def init_data(self):
        """Create initial test data."""
        n = 10
        d = 5
        X_actual = torch.rand(n, d, dtype=torch.float64)
        X_actual = X_actual / X_actual.sum(dim=1, keepdim=True)  # On simplex
        X_expected = X_actual + torch.randn(n, d, dtype=torch.float64) * 0.01
        X_expected = X_expected / X_expected.sum(dim=1, keepdim=True)
        Y = torch.randn(n, 1, dtype=torch.float64)
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.float64)
        return X_actual, X_expected, Y, bounds

    def test_save_init(self, handler, init_data):
        """save_init should store initial data."""
        X_actual, X_expected, Y, bounds = init_data
        handler.save_init(X_actual, X_expected, Y, bounds)

        assert handler.X_all_actual is not None
        assert handler.X_all_actual.shape == X_actual.shape
        assert torch.allclose(handler.X_all_actual, X_actual)
        assert torch.allclose(handler.Y_all, Y)

    def test_add_all_points(self, handler, init_data):
        """add_all_points should append new data."""
        X_actual, X_expected, Y, bounds = init_data
        handler.save_init(X_actual, X_expected, Y, bounds)

        # Add more points
        new_X = torch.rand(5, 5)
        new_X = new_X / new_X.sum(dim=1, keepdim=True)
        new_Y = torch.randn(5, 1)

        handler.add_all_points(new_X, new_X, new_Y)

        assert handler.X_all_actual.shape[0] == 15
        assert handler.Y_all.shape[0] == 15

    def test_get_all_points(self, handler, init_data):
        """get_all_points should return all stored data."""
        X_actual, X_expected, Y, bounds = init_data
        handler.save_init(X_actual, X_expected, Y, bounds)

        X_ret, _, Y_ret = handler.get_all_points()

        assert torch.allclose(X_ret, X_actual)
        assert torch.allclose(Y_ret, Y)


class TestDataHandlerNeedles:
    """Tests for needle management."""

    @pytest.fixture
    def handler_with_data(self):
        """Create handler with initial data."""
        handler = DataHandler(
            directory=None,
            device='cpu',
            dtype=torch.float64,
            d=5,
        )
        n, d = 20, 5
        X = torch.rand(n, d)
        X = X / X.sum(dim=1, keepdim=True)
        Y = torch.randn(n, 1)
        bounds = torch.tensor([[0.0] * d, [1.0] * d])
        handler.save_init(X, X, Y, bounds)
        return handler

    def test_add_needle(self, handler_with_data):
        """add_needle should store needle info."""
        handler = handler_with_data
        needle = handler.X_all_actual[0]
        
        handler.add_needle(
            needle=needle,
            needle_value=1.5,
            needle_penalty_radius=0.1,
            activation=0,
            zoom=0,
            iteration=5,
        )

        assert handler.needles.shape[0] == 1
        assert handler.needle_penalty_radii.shape[0] == 1
        assert len(handler.needles_results) == 1

    def test_get_needles_and_radii(self, handler_with_data):
        """get_needles_and_penalty_radii should return correct data."""
        handler = handler_with_data
        
        # Add two needles
        handler.add_needle(handler.X_all_actual[0], 1.0, 0.1, 0, 0, 0)
        handler.add_needle(handler.X_all_actual[1], 2.0, 0.15, 0, 1, 0)

        needles, radii = handler.get_needles_and_penalty_radii()

        assert needles.shape[0] == 2
        assert radii.shape[0] == 2
        assert radii[0].item() == pytest.approx(0.1)
        assert radii[1].item() == pytest.approx(0.15)

    def test_get_needle_results(self, handler_with_data):
        """get_needle_results should return detailed info."""
        handler = handler_with_data
        handler.add_needle(handler.X_all_actual[0], 1.0, 0.1, 2, 1, 5)

        results = handler.get_needle_results()

        assert len(results) == 1
        assert results[0]['activation'] == 2
        assert results[0]['zoom'] == 1
        assert results[0]['iteration'] == 5
        assert results[0]['value'] == 1.0


class TestDataHandlerPenaltyMask:
    """Tests for penalty mask computation."""

    @pytest.fixture
    def handler_with_needle(self):
        """Create handler with a needle."""
        handler = DataHandler(
            directory=None,
            device='cpu',
            dtype=torch.float64,
            d=3,
        )
        # Create points on simplex
        X = torch.tensor([
            [0.5, 0.3, 0.2],
            [0.3, 0.4, 0.3],
            [0.2, 0.5, 0.3],
        ])
        Y = torch.tensor([[1.0], [2.0], [3.0]])
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        handler.save_init(X, X, Y, bounds)
        
        # Add needle at first point with radius 0.1
        handler.add_needle(X[0], 1.0, 0.1, 0, 0, 0)
        return handler

    def test_penalty_mask_no_needles(self):
        """Without needles, all points should be unpenalized."""
        handler = DataHandler(directory=None, device='cpu', d=3)
        X = torch.rand(10, 3)
        X = X / X.sum(dim=1, keepdim=True)
        Y = torch.randn(10, 1)
        bounds = torch.zeros(2, 3)
        bounds[1] = 1.0
        handler.save_init(X, X, Y, bounds)

        mask = handler.get_penalty_mask()
        assert mask.all()  # All True (unpenalized)

    def test_penalty_mask_with_needle(self, handler_with_needle):
        """Points near needle should be penalized."""
        handler = handler_with_needle
        needle = handler.needles[0]
        
        # Point at needle center should be penalized
        mask = handler.get_penalty_mask(needle.unsqueeze(0))
        assert not mask[0].item()  # Penalized (False)

        # Point far away should not be penalized
        far_point = torch.tensor([[0.1, 0.1, 0.8]])
        mask = handler.get_penalty_mask(far_point)
        assert mask[0].item()  # Not penalized (True)

    def test_penalty_mask_3d_input(self, handler_with_needle):
        """Penalty mask should work with 3D input."""
        handler = handler_with_needle
        X = torch.rand(5, 3, 3)
        X = X / X.sum(dim=-1, keepdim=True)

        mask = handler.get_penalty_mask(X)
        assert mask.shape == (5, 3)


class TestDataHandlerGPData:
    """Tests for GP data preparation."""

    @pytest.fixture
    def handler_with_varied_data(self):
        """Create handler with varied Y values."""
        handler = DataHandler(
            directory=None,
            device='cpu',
            dtype=torch.float64,
            d=3,
            max_gp_points=5,
        )
        X = torch.tensor([
            [0.5, 0.3, 0.2],
            [0.3, 0.4, 0.3],
            [0.2, 0.5, 0.3],
            [0.4, 0.4, 0.2],
            [0.1, 0.6, 0.3],
            [0.6, 0.2, 0.2],
            [0.2, 0.2, 0.6],
        ])
        Y = torch.tensor([[1.0], [5.0], [3.0], [2.0], [4.0], [6.0], [0.5]])
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        handler.save_init(X, X, Y, bounds)
        return handler

    def test_get_gp_data_returns_top_points(self, handler_with_varied_data):
        """get_gp_data should return top points by Y value."""
        handler = handler_with_varied_data
        X_gp, Y_gp = handler.get_gp_data()

        # Should return top 5 points (max_gp_points=5)
        assert X_gp.shape[0] == 5
        assert Y_gp.shape[0] == 5

        # Should be sorted descending by Y
        assert Y_gp[0] >= Y_gp[-1]

    def test_get_gp_data_excludes_penalized(self, handler_with_varied_data):
        """get_gp_data should exclude penalized points."""
        handler = handler_with_varied_data
        
        # Add needle at the best point
        best_idx = handler.Y_all.argmax()
        handler.add_needle(
            handler.X_all_actual[best_idx],
            handler.Y_all[best_idx].item(),
            0.05,
            0, 0, 0
        )

        _, Y_gp = handler.get_gp_data()

        # Best point should now be excluded
        assert Y_gp.max() < 6.0


class TestDataHandlerBounds:
    """Tests for bounds determination."""

    def test_determine_new_bounds(self):
        """determine_new_bounds should narrow based on top points."""
        handler = DataHandler(
            directory=None,
            device='cpu',
            d=3,
            top_m_points=2,
        )
        X = torch.tensor([
            [0.5, 0.3, 0.2],  # Y=1
            [0.4, 0.4, 0.2],  # Y=5 (top)
            [0.3, 0.5, 0.2],  # Y=4 (top)
            [0.1, 0.6, 0.3],  # Y=2
        ])
        Y = torch.tensor([[1.0], [5.0], [4.0], [2.0]])
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        handler.save_init(X, X, Y, bounds)

        new_bounds = handler.determine_new_bounds()

        # Should be based on top 2 points (indices 1 and 2)
        assert new_bounds.shape == (2, 3)
        # Min bounds
        assert new_bounds[0, 0].item() == pytest.approx(0.3, abs=0.01)
        # Max bounds  
        assert new_bounds[1, 0].item() == pytest.approx(0.4, abs=0.01)


class TestDataHandlerInputNoise:
    """Tests for input noise calculation."""

    def test_normalized_input_noise(self):
        """get_normalized_input_noise should compute median normalized distance."""
        handler = DataHandler(directory=None, device='cpu', d=3)
        
        X_actual = torch.tensor([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
        ])
        X_expected = torch.tensor([
            [0.51, 0.29, 0.20],  # Small offset
            [0.42, 0.38, 0.20],  # Larger offset
        ])
        Y = torch.tensor([[1.0], [2.0]])
        bounds = torch.zeros(2, 3)
        bounds[1] = 1.0
        handler.save_init(X_actual, X_expected, Y, bounds)

        noise = handler.get_normalized_input_noise()
        assert noise > 0
        assert noise < 1.0  # Should be small for these offsets

    def test_input_noise_empty(self):
        """Input noise should be 0 for empty data."""
        handler = DataHandler(directory=None, device='cpu', d=3)
        assert handler.get_normalized_input_noise() == 0.0
        assert handler.get_input_noise() == 0.0


class TestDataHandlerCheckpointing:
    """Tests for checkpointing functionality."""

    def test_checkpoint_save_and_load(self):
        """Should save and load state correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate handler
            handler1 = DataHandler(
                directory=tmpdir,
                max_saved_recent_checkpoints=10,
                device='cpu',
                dtype=torch.float64,
                d=3,
            )
            X = torch.rand(5, 3, dtype=torch.float64)
            X = X / X.sum(dim=1, keepdim=True)
            Y = torch.randn(5, 1, dtype=torch.float64)
            bounds = torch.zeros(2, 3, dtype=torch.float64)
            bounds[1] = 1.0
            handler1.save_init(X, X, Y, bounds)
            handler1.update_iteration_state(2, 1, 5, 3)
            handler1.push_checkpoint("test_state", is_permanent=True)

            # Create new handler and load
            handler2 = DataHandler(
                directory=tmpdir,
                run_uuid=handler1.run_uuid,
                max_saved_recent_checkpoints=10,
                device='cpu',
                dtype=torch.float64,
                d=3,
            )
            act, zoom, iter_, no_imp = handler2.load_state()

            assert act == 2
            assert zoom == 1
            assert iter_ == 5
            assert no_imp == 3
            assert torch.allclose(handler2.X_all_actual, X)

    def test_no_save_mode_no_files(self):
        """No-save mode should not create files."""
        handler = DataHandler(
            directory=None,  # No saving
            device='cpu',
            d=3,
        )
        X = torch.rand(5, 3)
        X = X / X.sum(dim=1, keepdim=True)
        Y = torch.randn(5, 1)
        bounds = torch.zeros(2, 3)
        bounds[1] = 1.0
        handler.save_init(X, X, Y, bounds)
        handler.push_checkpoint("test")

        # No run directory should exist
        assert handler.run_dir is None


class TestDataHandlerBestPoint:
    """Tests for best point retrieval."""

    def test_get_best_unpenalized(self):
        """get_best_unpenalized should return the best point."""
        handler = DataHandler(directory=None, device='cpu', dtype=torch.float64, d=3)
        X = torch.tensor([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.3, 0.5, 0.2],
        ], dtype=torch.float64)
        Y = torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float64)
        bounds = torch.zeros(2, 3, dtype=torch.float64)
        bounds[1] = 1.0
        handler.save_init(X, X, Y, bounds)

        best_X, best_Y, idx = handler.get_best_unpenalized()

        assert torch.allclose(best_X, X[1])
        assert best_Y.item() == pytest.approx(5.0)
        assert idx == 1

    def test_get_best_unpenalized_with_needle(self):
        """Best point should exclude penalized areas."""
        handler = DataHandler(directory=None, device='cpu', d=3)
        X = torch.tensor([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.3, 0.5, 0.2],
        ])
        Y = torch.tensor([[1.0], [5.0], [3.0]])
        bounds = torch.zeros(2, 3)
        bounds[1] = 1.0
        handler.save_init(X, X, Y, bounds)

        # Penalize the best point
        handler.add_needle(X[1], 5.0, 0.05, 0, 0, 0)

        _, best_Y, idx = handler.get_best_unpenalized()

        # Should now return the second best
        assert best_Y.item() == pytest.approx(3.0)
        assert idx == 2
