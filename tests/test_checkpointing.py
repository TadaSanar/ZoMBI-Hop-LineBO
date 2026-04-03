"""
Tests for ZoMBI-Hop state reload / checkpointing feature.

Covers:
- Bounds are saved and reloaded correctly (including zoomed/updated bounds).
- Iteration state (activation, zoom, iteration) is at the right point after reload.
- No duplicate points: points from before the checkpoint are NOT re-added on reload.
- Data is not manipulated on restart: X_all_actual, X_all_expected, Y_all, needles
  have identical values/shapes/order after reload with no new data collected.
- Adding new data after reload correctly appends (does not overwrite) prior points.
- Needle state (locations, radii, results) is fully restored.
- Penalty mask is consistent with the reloaded needle state.
- Hyperparameter config values are restored from disk.
- Snapshot counter continues from where it left off (no re-numbering).
- Latest snapshot is loaded when multiple snapshots exist.
"""

import sys
import json
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.datahandler import DataHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simplex_points(n: int, d: int, dtype=torch.float64, seed: int = 0) -> torch.Tensor:
    """Generate n points on the d-simplex (rows sum to 1)."""
    torch.manual_seed(seed)
    X = torch.rand(n, d, dtype=dtype)
    return X / X.sum(dim=1, keepdim=True)


def _make_handler(tmpdir, d=3, **kwargs) -> DataHandler:
    return DataHandler(
        directory=tmpdir,
        device="cpu",
        dtype=torch.float64,
        d=d,
        **kwargs,
    )


def _init_handler(handler: DataHandler, n=10, d=3, seed=0):
    """Populate handler with n simplex points and return (X, Y, bounds)."""
    X = _simplex_points(n, d, seed=seed)
    Y = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.float64)
    handler.save_init(X, X.clone(), Y, bounds)
    return X, Y, bounds


# ---------------------------------------------------------------------------
# 1. Bounds persistence
# ---------------------------------------------------------------------------

class TestBoundsPersistence:

    def test_initial_bounds_saved_and_reloaded(self):
        """Original bounds are preserved exactly after snapshot + reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, bounds = _init_handler(h1)
            h1.take_snapshot("after_init")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert h2.bounds is not None
            assert torch.allclose(h2.bounds, bounds)

    def test_updated_bounds_are_reloaded(self):
        """When bounds are updated (e.g. after zoom), the new bounds are reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, bounds = _init_handler(h1)

            # Simulate zooming: narrow the bounds
            zoomed_bounds = torch.tensor(
                [[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]], dtype=torch.float64
            )
            h1.bounds = zoomed_bounds.clone()
            h1.update_iteration_state(0, 1, 0, 0)
            h1.take_snapshot("after_zoom")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert torch.allclose(h2.bounds, zoomed_bounds)

    def test_bounds_shape_preserved(self):
        """Reloaded bounds tensor has shape (2, d)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 5
            h1 = _make_handler(tmpdir, d=d)
            _init_handler(h1, d=d)
            h1.take_snapshot("bounds_shape")

            h2 = _make_handler(tmpdir, d=d, run_uuid=h1.run_uuid)
            h2.load_state()

            assert h2.bounds.shape == (2, d)


# ---------------------------------------------------------------------------
# 1b. Bounds persistence through actual zooming (determine_new_bounds)
# ---------------------------------------------------------------------------

class TestZoomedBoundsPersistence:

    def test_determine_new_bounds_then_reload(self):
        """Bounds computed by determine_new_bounds() are reloaded exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            # Spread points so zooming narrows bounds noticeably
            X = torch.tensor([
                [0.5, 0.3, 0.2],  # Y=1
                [0.4, 0.4, 0.2],  # Y=5  (top)
                [0.3, 0.5, 0.2],  # Y=4  (top)
                [0.1, 0.6, 0.3],  # Y=2
            ], dtype=torch.float64)
            Y = torch.tensor([[1.0], [5.0], [4.0], [2.0]], dtype=torch.float64)
            bounds = torch.zeros(2, d, dtype=torch.float64)
            bounds[1] = 1.0
            h1.save_init(X, X, Y, bounds)

            zoomed = h1.determine_new_bounds()
            h1.bounds = zoomed.clone()
            h1.update_iteration_state(0, 1, 0, 0)
            h1.take_snapshot("zoom1")

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            assert torch.allclose(h2.bounds, zoomed)

    def test_zoomed_bounds_are_narrower_than_original(self):
        """After zooming, reloaded bounds span less than original [0,1]^d."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            X = torch.tensor([
                [0.5, 0.3, 0.2],
                [0.4, 0.4, 0.2],
                [0.3, 0.5, 0.2],
                [0.1, 0.6, 0.3],
            ], dtype=torch.float64)
            Y = torch.tensor([[1.0], [5.0], [4.0], [2.0]], dtype=torch.float64)
            bounds = torch.zeros(2, d, dtype=torch.float64)
            bounds[1] = 1.0
            h1.save_init(X, X, Y, bounds)

            zoomed = h1.determine_new_bounds()
            h1.bounds = zoomed.clone()
            h1.take_snapshot("narrow")

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            # Width of reloaded bounds must be strictly less than 1 in at least one dim
            widths = h2.bounds[1] - h2.bounds[0]
            assert (widths < 1.0).any()

    def test_sequential_zoom_levels_reload_last(self):
        """Three successive zoom snapshots: reload gives the third (narrowest) bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            X = torch.tensor([
                [0.5, 0.3, 0.2],
                [0.4, 0.4, 0.2],
                [0.3, 0.5, 0.2],
                [0.1, 0.6, 0.3],
                [0.45, 0.35, 0.2],
                [0.35, 0.45, 0.2],
            ], dtype=torch.float64)
            Y = torch.tensor([[1.0], [5.0], [4.0], [2.0], [4.5], [3.5]], dtype=torch.float64)
            bounds = torch.zeros(2, d, dtype=torch.float64)
            bounds[1] = 1.0
            h1.save_init(X, X, Y, bounds)

            # Zoom 1
            z1 = h1.determine_new_bounds()
            h1.bounds = z1.clone()
            h1.update_iteration_state(0, 1, 0, 0)
            h1.take_snapshot("zoom1")

            # Zoom 2: narrow further around the now-top points
            z2 = h1.determine_new_bounds()
            h1.bounds = z2.clone()
            h1.update_iteration_state(0, 2, 0, 0)
            h1.take_snapshot("zoom2")

            # Zoom 3
            z3 = h1.determine_new_bounds()
            h1.bounds = z3.clone()
            h1.update_iteration_state(0, 3, 0, 0)
            h1.take_snapshot("zoom3")

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            assert torch.allclose(h2.bounds, z3)
            assert h2.current_zoom == 3

    def test_bounds_lower_leq_upper_after_zoom_reload(self):
        """lower <= upper for every dimension of reloaded zoomed bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 4
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=3,
            )
            X = _simplex_points(12, d, seed=42)
            Y = torch.arange(12, dtype=torch.float64).unsqueeze(1)
            bounds = torch.zeros(2, d, dtype=torch.float64)
            bounds[1] = 1.0
            h1.save_init(X, X, Y, bounds)

            zoomed = h1.determine_new_bounds()
            h1.bounds = zoomed.clone()
            h1.take_snapshot("zoomed_4d")

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            assert (h2.bounds[0] <= h2.bounds[1]).all()

    def test_bounds_dtype_and_device_preserved_after_zoom(self):
        """Reloaded zoomed bounds have the correct dtype and device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            X, Y, bounds = _init_handler(h1, n=8, d=d)
            zoomed = h1.determine_new_bounds()
            h1.bounds = zoomed.clone()
            h1.take_snapshot("dtype_check")

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            assert h2.bounds.dtype == torch.float64
            assert h2.bounds.device.type == "cpu"

    def test_zoomed_bounds_independent_of_original_bounds_object(self):
        """Mutating the original bounds tensor after snapshot does not affect reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            X, Y, _ = _init_handler(h1, n=8, d=d)
            zoomed = h1.determine_new_bounds()
            saved_zoomed = zoomed.clone()
            h1.bounds = zoomed.clone()
            h1.take_snapshot("independence")

            # Mutate h1.bounds after snapshot
            h1.bounds[:] = 999.0

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            assert torch.allclose(h2.bounds, saved_zoomed)


# ---------------------------------------------------------------------------
# 1c. Zoom-out: bounds reset to full simplex after zoom-in
# ---------------------------------------------------------------------------

class TestZoomOutBoundsPersistence:
    """
    When the optimizer detects > 90 % of the space is penalized (infinite run),
    it resets bounds to the full [0,1]^d simplex.  Verify that the reset bounds
    are what is saved to the snapshot and correctly reloaded.
    """

    def _full_bounds(self, d: int) -> torch.Tensor:
        b = torch.zeros(2, d, dtype=torch.float64)
        b[1] = 1.0
        return b

    def test_zoom_out_to_full_simplex_is_reloaded(self):
        """After zooming in then out, reloaded bounds equal full [0,1]^d."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            X, Y, _ = _init_handler(h1, n=10, d=d)

            # Zoom in
            zoomed = h1.determine_new_bounds()
            h1.bounds = zoomed.clone()
            h1.update_iteration_state(0, 1, 0, 0)
            h1.take_snapshot("zoomed_in")

            # Zoom out: reset to full simplex (mirrors zombihop.py zoom-out logic)
            full = self._full_bounds(d)
            h1.bounds = full.clone()
            h1.update_iteration_state(0, 2, 0, 0)
            h1.take_snapshot("zoomed_out")

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            assert torch.allclose(h2.bounds, full)

    def test_zoom_out_bounds_wider_than_zoomed_in(self):
        """Reloaded zoom-out bounds span strictly more than the preceding zoomed-in bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            X, Y, _ = _init_handler(h1, n=10, d=d)

            zoomed = h1.determine_new_bounds()
            zoomed_width = (zoomed[1] - zoomed[0]).sum().item()

            h1.bounds = self._full_bounds(d).clone()
            h1.take_snapshot("after_zoom_out")

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            reloaded_width = (h2.bounds[1] - h2.bounds[0]).sum().item()
            assert reloaded_width > zoomed_width

    def test_zoom_in_snapshot_not_clobbered_by_zoom_out(self):
        """The zoom-in snapshot still exists on disk after the zoom-out snapshot is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            _init_handler(h1, n=10, d=d)

            zoomed = h1.determine_new_bounds()
            h1.bounds = zoomed.clone()
            h1.take_snapshot("zoomed_in")

            h1.bounds = self._full_bounds(d).clone()
            h1.take_snapshot("zoomed_out")

            snapshots_dir = h1.run_dir / "snapshots"
            names = [p.name for p in snapshots_dir.iterdir()]
            assert any("zoomed_in" in n for n in names)
            assert any("zoomed_out" in n for n in names)

    def test_multiple_zoom_in_out_cycles_reload_last(self):
        """After two zoom-in/zoom-out cycles, reload gives the last zoom-out (full) bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            _init_handler(h1, n=10, d=d)
            full = self._full_bounds(d)

            for cycle in range(2):
                zoomed = h1.determine_new_bounds()
                h1.bounds = zoomed.clone()
                h1.update_iteration_state(cycle, 1, 0, 0)
                h1.take_snapshot(f"cycle{cycle}_in")

                h1.bounds = full.clone()
                h1.update_iteration_state(cycle + 1, 0, 0, 0)
                h1.take_snapshot(f"cycle{cycle}_out")

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            assert torch.allclose(h2.bounds, full)

    def test_zoom_out_iteration_state_correct(self):
        """After zoom-out snapshot, reloaded activation/zoom/iteration match the reset state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            _init_handler(h1, n=10, d=d)

            # Zoom in at activation 0, zoom 1
            h1.bounds = h1.determine_new_bounds().clone()
            h1.update_iteration_state(0, 1, 5, 0)
            h1.take_snapshot("in")

            # Zoom out: zombihop resets zoom to 0 and increments activation
            h1.bounds = self._full_bounds(d).clone()
            h1.update_iteration_state(1, 0, 0, 0)
            h1.take_snapshot("out")

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            act, zoom, it, _ = h2.load_state()

            assert act == 1
            assert zoom == 0
            assert it == 0
            assert torch.allclose(h2.bounds, self._full_bounds(d))

    def test_data_unchanged_through_zoom_in_out(self):
        """All collected data points are identical after a zoom-in → zoom-out cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                top_m_points=2,
            )
            X, Y, _ = _init_handler(h1, n=10, d=d)

            # Add a batch during zoom-in phase
            X_mid = _simplex_points(4, d, seed=77)
            Y_mid = torch.ones(4, 1, dtype=torch.float64) * 7
            h1.bounds = h1.determine_new_bounds().clone()
            h1.add_all_points(X_mid, X_mid.clone(), Y_mid)
            h1.take_snapshot("zoom_in_with_data")

            # Zoom out
            h1.bounds = self._full_bounds(d).clone()
            h1.take_snapshot("zoom_out")

            X_all_before = h1.X_all_actual.clone()
            Y_all_before = h1.Y_all.clone()

            h2 = DataHandler(
                directory=tmpdir, device="cpu", dtype=torch.float64, d=d,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            assert torch.allclose(h2.X_all_actual, X_all_before)
            assert torch.allclose(h2.Y_all, Y_all_before)
            # Bounds are full simplex, data count is init + mid batch
            assert h2.X_all_actual.shape[0] == 14
            assert torch.allclose(h2.bounds, self._full_bounds(d))


# ---------------------------------------------------------------------------
# 2. Iteration state
# ---------------------------------------------------------------------------

class TestIterationState:

    @pytest.mark.parametrize("activation,zoom,iteration", [
        (0, 0, 0),
        (0, 1, 3),
        (2, 0, 7),
        (3, 2, 9),
    ])
    def test_iteration_state_exact_match(self, activation, zoom, iteration):
        """Activation / zoom / iteration are restored exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            _init_handler(h1)
            h1.update_iteration_state(activation, zoom, iteration, 0)
            h1.take_snapshot(f"act{activation}_z{zoom}_i{iteration}")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            act, z, it, _ = h2.load_state()

            assert act == activation
            assert z == zoom
            assert it == iteration

    def test_latest_snapshot_is_loaded(self):
        """When multiple snapshots exist, the latest one is loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            _init_handler(h1)

            # Save multiple snapshots with different iteration states
            h1.update_iteration_state(0, 0, 2, 0)
            h1.take_snapshot("early")
            h1.update_iteration_state(0, 1, 5, 0)
            h1.take_snapshot("middle")
            h1.update_iteration_state(1, 0, 3, 0)
            h1.take_snapshot("latest")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            act, z, it, _ = h2.load_state()

            # Should load the last snapshot
            assert act == 1
            assert z == 0
            assert it == 3


# ---------------------------------------------------------------------------
# 3. No duplicate points
# ---------------------------------------------------------------------------

class TestNoDuplicatePoints:

    def test_point_count_unchanged_on_reload(self):
        """Number of data points is the same before and after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, _ = _init_handler(h1, n=12)
            h1.take_snapshot("full_init")

            n_before = h1.X_all_actual.shape[0]

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert h2.X_all_actual.shape[0] == n_before

    def test_reload_then_add_new_data_count(self):
        """After reload, adding new points grows count by exactly the new batch size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            _init_handler(h1, n=10)
            h1.take_snapshot("saved")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            n_before = h2.X_all_actual.shape[0]
            new_X = _simplex_points(5, 3, seed=99)
            new_Y = torch.ones(5, 1, dtype=torch.float64)
            h2.add_all_points(new_X, new_X.clone(), new_Y)

            assert h2.X_all_actual.shape[0] == n_before + 5

    def test_no_init_data_repeated_on_reload(self):
        """Initial X values appear exactly once in X_all_actual after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, _, _ = _init_handler(h1, n=8)
            h1.take_snapshot("check_no_dup")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            # Each row of X should appear exactly once
            for i in range(X.shape[0]):
                matches = (h2.X_all_actual - X[i].unsqueeze(0)).norm(dim=1) < 1e-9
                assert matches.sum().item() == 1, (
                    f"Row {i} appears {matches.sum().item()} times instead of 1"
                )


# ---------------------------------------------------------------------------
# 4. Data integrity on restart (no manipulation)
# ---------------------------------------------------------------------------

class TestDataIntegrityOnRestart:

    def test_X_all_actual_identical(self):
        """X_all_actual is identical (same values, same order) after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=4)
            X, _, _ = _init_handler(h1, n=15, d=4)
            h1.take_snapshot("integrity")

            h2 = _make_handler(tmpdir, d=4, run_uuid=h1.run_uuid)
            h2.load_state()

            assert torch.allclose(h2.X_all_actual, h1.X_all_actual)

    def test_X_all_expected_identical(self):
        """X_all_expected is identical after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X = _simplex_points(10, 3, seed=7)
            X_exp = _simplex_points(10, 3, seed=8)  # Different from actual
            Y = torch.randn(10, 1, dtype=torch.float64)
            bounds = torch.zeros(2, 3, dtype=torch.float64)
            bounds[1] = 1.0
            h1.save_init(X, X_exp, Y, bounds)
            h1.take_snapshot("expected_check")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert torch.allclose(h2.X_all_expected, h1.X_all_expected)

    def test_Y_all_identical(self):
        """Y_all values and order are unchanged after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            _, Y, _ = _init_handler(h1, n=10)
            h1.take_snapshot("y_check")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert torch.allclose(h2.Y_all, h1.Y_all)

    def test_Y_all_order_preserved(self):
        """Y_all is not sorted or reordered on reload — original insertion order is kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            # Create deliberately non-monotone Y
            X = _simplex_points(6, 3)
            Y = torch.tensor([[3.0], [1.0], [5.0], [2.0], [4.0], [0.5]], dtype=torch.float64)
            bounds = torch.zeros(2, 3, dtype=torch.float64)
            bounds[1] = 1.0
            h1.save_init(X, X, Y, bounds)
            h1.take_snapshot("order_check")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            for i in range(Y.shape[0]):
                assert h2.Y_all[i].item() == pytest.approx(Y[i].item()), (
                    f"Y_all[{i}] changed from {Y[i].item()} to {h2.Y_all[i].item()}"
                )

    def test_X_init_actual_preserved(self):
        """X_init_actual (the original init slice) is unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, _, _ = _init_handler(h1, n=8)
            h1.take_snapshot("init_preserved")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert torch.allclose(h2.X_init_actual, h1.X_init_actual)

    def test_Y_init_preserved(self):
        """Y_init is unchanged after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            _, Y, _ = _init_handler(h1, n=8)
            h1.take_snapshot("y_init")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert torch.allclose(h2.Y_init, h1.Y_init)

    def test_no_data_mutation_without_new_objective_calls(self):
        """Reload followed by NO new data: all tensors are byte-for-byte equal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, _ = _init_handler(h1, n=10)
            # Simulate some progress: add needle, update state
            h1.add_needle(X[5], Y[5].item(), 0.05, 0, 0, 4)
            h1.update_iteration_state(0, 0, 4, 0)
            h1.take_snapshot("mid_run")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert torch.allclose(h2.X_all_actual, h1.X_all_actual)
            assert torch.allclose(h2.Y_all, h1.Y_all)
            assert torch.allclose(h2.needles, h1.needles)
            assert torch.allclose(h2.needle_vals, h1.needle_vals)
            assert torch.allclose(h2.needle_penalty_radii, h1.needle_penalty_radii)


# ---------------------------------------------------------------------------
# 5. Needle state
# ---------------------------------------------------------------------------

class TestNeedleReload:

    def test_needle_count_preserved(self):
        """Number of needles is the same after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, _ = _init_handler(h1, n=10)
            h1.add_needle(X[3], Y[3].item(), 0.08, 0, 0, 2)
            h1.add_needle(X[7], Y[7].item(), 0.06, 0, 1, 3)
            h1.take_snapshot("needles")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert h2.needles.shape[0] == 2

    def test_needle_locations_identical(self):
        """Needle locations (tensor) are exactly preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, _ = _init_handler(h1, n=10)
            h1.add_needle(X[4], float(Y[4].item()), 0.07, 0, 0, 3)
            h1.take_snapshot("needle_loc")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert torch.allclose(h2.needles, h1.needles)

    def test_needle_radii_identical(self):
        """Needle penalty radii are exactly preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, _ = _init_handler(h1, n=10)
            h1.add_needle(X[2], float(Y[2].item()), 0.12, 0, 0, 1)
            h1.add_needle(X[6], float(Y[6].item()), 0.07, 0, 1, 2)
            h1.take_snapshot("radii")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert torch.allclose(h2.needle_penalty_radii, h1.needle_penalty_radii)

    def test_needles_results_metadata(self):
        """Needle metadata (activation/zoom/iteration/value) matches after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, _ = _init_handler(h1, n=10)
            h1.add_needle(X[1], 1.23, 0.09, activation=2, zoom=1, iteration=7)
            h1.take_snapshot("needle_meta")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert len(h2.needles_results) == 1
            r = h2.needles_results[0]
            assert r["activation"] == 2
            assert r["zoom"] == 1
            assert r["iteration"] == 7
            assert r["value"] == pytest.approx(1.23)

    def test_zero_needles_reloaded(self):
        """With no needles, reload produces empty needle tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            _init_handler(h1)
            h1.take_snapshot("no_needles")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert h2.needles.shape[0] == 0
            assert len(h2.needles_results) == 0


# ---------------------------------------------------------------------------
# 6. Penalty mask consistency after reload
# ---------------------------------------------------------------------------

class TestPenaltyMaskAfterReload:

    def test_penalty_mask_matches_original_after_reload(self):
        """Reloaded penalty mask equals the original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, _ = _init_handler(h1, n=10)
            h1.add_needle(X[0], float(Y[0].item()), 0.1, 0, 0, 0)
            h1.take_snapshot("mask_check")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert torch.equal(h2.get_penalty_mask(), h1.get_penalty_mask())

    def test_penalized_points_still_penalized_after_reload(self):
        """Points that were penalized before snapshot remain penalized after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, _ = _init_handler(h1, n=10)
            needle = X[0].clone()
            h1.add_needle(needle, float(Y[0].item()), 0.15, 0, 0, 0)

            # Verify penalized before saving
            mask_before = h1.get_penalty_mask(needle.unsqueeze(0))
            assert not mask_before[0].item()

            h1.take_snapshot("pen_check")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            mask_after = h2.get_penalty_mask(needle.unsqueeze(0))
            assert not mask_after[0].item()

    def test_unpenalized_points_still_unpenalized_after_reload(self):
        """Points outside all penalty balls remain unpenalized after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, _ = _init_handler(h1, n=10)
            # Small radius needle far from the test point
            h1.add_needle(X[0], float(Y[0].item()), 0.01, 0, 0, 0)
            far = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float64)

            # Verify not penalized before
            assert h1.get_penalty_mask(far)[0].item()

            h1.take_snapshot("unpen_check")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            assert h2.get_penalty_mask(far)[0].item()

    def test_new_data_after_reload_gets_correct_penalty_mask(self):
        """New points added after reload are correctly masked against existing needles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            X, Y, _ = _init_handler(h1, n=10)
            needle = X[0].clone()
            radius = 0.15
            h1.add_needle(needle, float(Y[0].item()), radius, 0, 0, 0)
            h1.take_snapshot("needle_for_new_data")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            # Point inside the needle radius — should be penalized
            inside = (needle + torch.tensor([0.02, -0.01, -0.01], dtype=torch.float64)).unsqueeze(0)
            inside = inside / inside.sum()
            new_Y = torch.tensor([[99.0]], dtype=torch.float64)
            mask_returned = h2.add_all_points(inside, inside.clone(), new_Y)
            assert not mask_returned[0].item()  # Penalized = False in mask

            # Point far away — should be unpenalized
            far = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float64)
            new_Y2 = torch.tensor([[1.0]], dtype=torch.float64)
            mask_returned2 = h2.add_all_points(far, far.clone(), new_Y2)
            assert mask_returned2[0].item()


# ---------------------------------------------------------------------------
# 7. Hyperparameter config persistence
# ---------------------------------------------------------------------------

class TestConfigPersistence:

    def test_hyperparams_restored_from_config_json(self):
        """All hyperparameters written to config.json are reloaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = DataHandler(
                directory=tmpdir,
                device="cpu",
                dtype=torch.float64,
                d=4,
                max_zooms=5,
                max_iterations=15,
                top_m_points=6,
                n_restarts=20,
                raw=300,
                penalization_threshold=5e-4,
                penalty_num_directions=40,
                penalty_max_radius=0.25,
                convergence_pi_threshold=0.005,
                n_consecutive_converged=3,
                max_gp_points=500,
                acquisition_type="ei",
                ucb_beta=0.2,
            )
            X = _simplex_points(8, 4)
            Y = torch.randn(8, 1, dtype=torch.float64)
            bounds = torch.zeros(2, 4, dtype=torch.float64)
            bounds[1] = 1.0
            h1.save_init(X, X, Y, bounds)
            # Config is written in save_init via _save_config

            h2 = DataHandler(
                directory=tmpdir,
                device="cpu",
                dtype=torch.float64,
                d=4,
                run_uuid=h1.run_uuid,
            )
            h2.load_state()

            assert h2.max_zooms == 5
            assert h2.max_iterations == 15
            assert h2.top_m_points == 6
            assert h2.n_restarts == 20
            assert h2.raw == 300
            assert h2.penalization_threshold == pytest.approx(5e-4)
            assert h2.penalty_num_directions == 40
            assert h2.penalty_max_radius == pytest.approx(0.25)
            assert h2.convergence_pi_threshold == pytest.approx(0.005)
            assert h2.n_consecutive_converged == 3
            assert h2.max_gp_points == 500
            assert h2.acquisition_type == "ei"
            assert h2.ucb_beta == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 8. Snapshot counter continuity
# ---------------------------------------------------------------------------

class TestSnapshotCounterContinuity:

    def test_snapshot_count_continues_after_reload(self):
        """After reloading, new snapshots are numbered after the last existing one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            _init_handler(h1)
            h1.take_snapshot("s1")  # snapshot 2 (init is 1)
            h1.take_snapshot("s2")  # snapshot 3

            count_before = h1._snapshot_count

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()

            # _snapshot_count should be set from existing files
            assert h2._snapshot_count == count_before

            # The next snapshot should have a higher number
            h2.take_snapshot("after_reload")
            snapshots_dir = h2.run_dir / "snapshots"
            names = sorted(p.name for p in snapshots_dir.iterdir())
            last_count = int(names[-1].split("_")[0])
            assert last_count == count_before + 1

    def test_no_snapshot_name_collision_after_reload(self):
        """Snapshot directory names after reload do not collide with pre-reload ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            _init_handler(h1)
            h1.take_snapshot("a")
            h1.take_snapshot("b")

            h2 = _make_handler(tmpdir, d=3, run_uuid=h1.run_uuid)
            h2.load_state()
            h2.take_snapshot("c")
            h2.take_snapshot("d")

            snapshots_dir = h2.run_dir / "snapshots"
            names = [p.name for p in snapshots_dir.iterdir()]
            # All names must be unique
            assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# 9. Max snapshots / cleanup
# ---------------------------------------------------------------------------

class TestSnapshotCleanup:

    def test_max_snapshots_limits_files(self):
        """With max_snapshots=3, only 3 snapshot directories are kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3, max_snapshots=3)
            _init_handler(h1)  # triggers take_snapshot("init")
            for i in range(5):
                h1.take_snapshot(f"extra_{i}")

            snapshots_dir = h1.run_dir / "snapshots"
            kept = list(snapshots_dir.iterdir())
            assert len(kept) <= 3

    def test_latest_txt_points_to_most_recent(self):
        """latest.txt always points to the most recently saved snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, d=3)
            _init_handler(h1)
            h1.take_snapshot("first")
            h1.take_snapshot("second")
            h1.take_snapshot("third")

            latest = (h1.run_dir / "latest.txt").read_text().strip()
            assert "third" in latest


# ---------------------------------------------------------------------------
# 10. Round-trip: save partial run, reload, continue, save again
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_full_round_trip_data_continuity(self):
        """
        Simulate saving mid-run, reloading, adding more data, and saving again.
        Final data contains exactly init + batch1 + batch2 with no duplication.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            n_init = 8
            n_batch1 = 4
            n_batch2 = 3

            # --- Phase 1: initial run ---
            h1 = _make_handler(tmpdir, d=d)
            X_init, Y_init, _ = _init_handler(h1, n=n_init, d=d, seed=0)
            X_b1 = _simplex_points(n_batch1, d, seed=10)
            Y_b1 = torch.ones(n_batch1, 1, dtype=torch.float64) * 10
            h1.add_all_points(X_b1, X_b1.clone(), Y_b1)
            h1.update_iteration_state(0, 0, 2, 0)
            h1.take_snapshot("mid_run")

            total_before = n_init + n_batch1

            # --- Phase 2: reload and continue ---
            h2 = _make_handler(tmpdir, d=d, run_uuid=h1.run_uuid)
            h2.load_state()

            assert h2.X_all_actual.shape[0] == total_before

            X_b2 = _simplex_points(n_batch2, d, seed=20)
            Y_b2 = torch.ones(n_batch2, 1, dtype=torch.float64) * 20
            h2.add_all_points(X_b2, X_b2.clone(), Y_b2)
            h2.update_iteration_state(0, 0, 3, 0)
            h2.take_snapshot("continued")

            assert h2.X_all_actual.shape[0] == total_before + n_batch2

        # Data integrity: first n_init rows match X_init
        assert torch.allclose(h2.X_all_actual[:n_init], X_init)
        # Next n_batch1 rows match X_b1
        assert torch.allclose(h2.X_all_actual[n_init:n_init + n_batch1], X_b1)
        # Last n_batch2 rows match X_b2
        assert torch.allclose(h2.X_all_actual[n_init + n_batch1:], X_b2)

    def test_needle_indices_valid_after_reload_and_new_data(self):
        """
        needle_indices must remain valid (< total points) after reload and new additions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = _make_handler(tmpdir, d=d)
            X, Y, _ = _init_handler(h1, n=10, d=d)
            h1.add_needle(X[3], float(Y[3].item()), 0.05, 0, 0, 2)
            h1.take_snapshot("needle_idx")

            h2 = _make_handler(tmpdir, d=d, run_uuid=h1.run_uuid)
            h2.load_state()

            new_X = _simplex_points(5, d, seed=55)
            new_Y = torch.ones(5, 1, dtype=torch.float64)
            h2.add_all_points(new_X, new_X.clone(), new_Y)

            total = h2.X_all_actual.shape[0]
            for idx in h2.needle_indices.flatten().tolist():
                assert 0 <= idx < total
