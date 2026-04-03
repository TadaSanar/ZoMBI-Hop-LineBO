"""
Tests for permanent snapshot flag and cleanup logic.

Covers:
- take_snapshot(permanent=True) writes a 'permanent' marker file
- take_snapshot(permanent=False) leaves no marker file
- _cleanup_old_snapshots never removes permanent snapshots
- _cleanup_old_snapshots only trims non-permanent, keeping the last max_snapshots of them
- push_checkpoint(is_permanent=True) forwards the flag
- save_init creates a permanent snapshot
- All ZoMBI-Hop event labels (needle, failed, finished, zoomed_out, timeout, final)
  survive cleanup when flagged permanent
- Stress: many iterations interleaved with permanent events;
  permanent ones always survive, non-permanent roll off to max_snapshots
- Edge cases: max_snapshots=1, all-permanent, no-permanent
- num_iterations_saved wiring through ZoMBIHop → DataHandler
- Reload works correctly after aggressive cleanup
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.datahandler import DataHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simplex_points(n: int, d: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    X = torch.rand(n, d, dtype=torch.float64)
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
    X = _simplex_points(n, d, seed=seed)
    Y = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    bounds = torch.zeros(2, d, dtype=torch.float64)
    bounds[1] = 1.0
    handler.save_init(X, X.clone(), Y, bounds)
    return X, Y, bounds


def _all_snapshot_dirs(handler: DataHandler):
    snapshots_dir = handler.run_dir / "snapshots"
    return sorted(snapshots_dir.iterdir()) if snapshots_dir.exists() else []


def _is_permanent(snapshot_dir: Path) -> bool:
    return (snapshot_dir / "permanent").exists()


def _permanent_snapshots(handler: DataHandler):
    return [s for s in _all_snapshot_dirs(handler) if _is_permanent(s)]


def _nonpermanent_snapshots(handler: DataHandler):
    return [s for s in _all_snapshot_dirs(handler) if not _is_permanent(s)]


# ---------------------------------------------------------------------------
# 1. Marker file presence
# ---------------------------------------------------------------------------

class TestPermanentMarkerFile:

    def test_permanent_true_creates_marker(self):
        """take_snapshot(permanent=True) writes a 'permanent' file in the snapshot dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir)
            _init_handler(h)
            h.take_snapshot("event", permanent=True)

            snapshots = _all_snapshot_dirs(h)
            event_snaps = [s for s in snapshots if "event" in s.name]
            assert len(event_snaps) == 1
            assert (event_snaps[0] / "permanent").exists()

    def test_permanent_false_no_marker(self):
        """take_snapshot(permanent=False) leaves no 'permanent' file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir)
            _init_handler(h)
            h.take_snapshot("regular", permanent=False)

            snapshots = _all_snapshot_dirs(h)
            regular_snaps = [s for s in snapshots if "regular" in s.name]
            assert len(regular_snaps) == 1
            assert not (regular_snaps[0] / "permanent").exists()

    def test_default_is_nonpermanent(self):
        """take_snapshot with no permanent kwarg defaults to non-permanent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir)
            _init_handler(h)
            h.take_snapshot("default_snap")

            snapshots = _all_snapshot_dirs(h)
            default_snaps = [s for s in snapshots if "default_snap" in s.name]
            assert len(default_snaps) == 1
            assert not (default_snaps[0] / "permanent").exists()

    def test_save_init_produces_permanent_init_snapshot(self):
        """save_init() takes a permanent 'init' snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir)
            _init_handler(h)

            init_snaps = [s for s in _all_snapshot_dirs(h) if "init" in s.name]
            assert len(init_snaps) == 1
            assert _is_permanent(init_snaps[0])

    def test_multiple_permanent_all_marked(self):
        """Several permanent snapshots all have the marker file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir)
            _init_handler(h)
            for label in ["needle", "failed", "final", "timeout"]:
                h.take_snapshot(label, permanent=True)

            for s in _all_snapshot_dirs(h):
                assert _is_permanent(s), f"{s.name} is missing the permanent marker"


# ---------------------------------------------------------------------------
# 2. push_checkpoint backward-compat wrapper forwards permanent flag
# ---------------------------------------------------------------------------

class TestPushCheckpointForwardsPermanent:

    def test_push_checkpoint_is_permanent_true(self):
        """push_checkpoint(is_permanent=True) results in a permanent snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir)
            _init_handler(h)
            h.push_checkpoint("compat_perm", is_permanent=True)

            snaps = [s for s in _all_snapshot_dirs(h) if "compat_perm" in s.name]
            assert len(snaps) == 1
            assert _is_permanent(snaps[0])

    def test_push_checkpoint_is_permanent_false(self):
        """push_checkpoint(is_permanent=False) results in a non-permanent snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir)
            _init_handler(h)
            h.push_checkpoint("compat_nonperm", is_permanent=False)

            snaps = [s for s in _all_snapshot_dirs(h) if "compat_nonperm" in s.name]
            assert len(snaps) == 1
            assert not _is_permanent(snaps[0])

    def test_push_checkpoint_default_is_nonpermanent(self):
        """push_checkpoint with no is_permanent arg defaults to non-permanent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir)
            _init_handler(h)
            h.push_checkpoint("compat_default")

            snaps = [s for s in _all_snapshot_dirs(h) if "compat_default" in s.name]
            assert not _is_permanent(snaps[0])


# ---------------------------------------------------------------------------
# 3. Cleanup: permanent snapshots are never deleted
# ---------------------------------------------------------------------------

class TestCleanupNeverDeletesPermanent:

    def test_single_permanent_survives_aggressive_cleanup(self):
        """A permanent snapshot survives when max_snapshots=1 and many others are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=1)
            _init_handler(h)  # permanent init already saved

            # Add one permanent event
            h.take_snapshot("needle", permanent=True)

            # Flood with non-permanent snapshots to trigger lots of cleanup
            for i in range(20):
                h.take_snapshot(f"iter_{i}", permanent=False)

            perm_snaps = _permanent_snapshots(h)
            perm_names = [s.name for s in perm_snaps]
            assert any("init" in n for n in perm_names), "init snapshot was deleted"
            assert any("needle" in n for n in perm_names), "needle snapshot was deleted"

    def test_all_permanent_events_survive_tight_cleanup(self):
        """All zombihop permanent event labels survive with max_snapshots=2."""
        permanent_labels = [
            "act0_timeout",
            "act0_z0_i3_timeout",
            "act0_z0_i5_failed",
            "act0_z0_i7_needle",
            "act0_z0_zoomed_out",
            "act0_z0_finished",
            "final",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=2)
            _init_handler(h)

            # Save permanent events interspersed with non-permanent iterations
            for i in range(5):
                h.take_snapshot(f"act0_z0_i{i}", permanent=False)
            for label in permanent_labels:
                h.take_snapshot(label, permanent=True)
            for i in range(5, 20):
                h.take_snapshot(f"act0_z0_i{i}", permanent=False)

            perm_names = {s.name for s in _permanent_snapshots(h)}
            for label in permanent_labels:
                assert any(label in n for n in perm_names), (
                    f"Permanent snapshot '{label}' was incorrectly deleted"
                )

    def test_permanent_count_never_shrinks(self):
        """The number of permanent snapshots only ever increases, never decreases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=2)
            _init_handler(h)
            permanent_count = 1  # init

            for i in range(30):
                if i % 5 == 0:
                    h.take_snapshot(f"perm_{i}", permanent=True)
                    permanent_count += 1
                else:
                    h.take_snapshot(f"iter_{i}", permanent=False)

                current_perm = len(_permanent_snapshots(h))
                assert current_perm == permanent_count, (
                    f"After step {i}: expected {permanent_count} permanent, got {current_perm}"
                )


# ---------------------------------------------------------------------------
# 4. Cleanup: non-permanent snapshots are trimmed correctly
# ---------------------------------------------------------------------------

class TestCleanupTrimsNonPermanent:

    def test_nonpermanent_count_bounded_by_max_snapshots(self):
        """Number of non-permanent snapshots never exceeds max_snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=5)
            _init_handler(h)  # permanent

            for i in range(20):
                h.take_snapshot(f"iter_{i}", permanent=False)
                nonperm = _nonpermanent_snapshots(h)
                assert len(nonperm) <= 5, (
                    f"After iter {i}: {len(nonperm)} non-permanent snapshots, limit is 5"
                )

    def test_oldest_nonpermanent_deleted_first(self):
        """When cleanup runs, the oldest non-permanent snapshots are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=3)
            _init_handler(h)

            for i in range(6):
                h.take_snapshot(f"iter_{i:02d}", permanent=False)

            nonperm = _nonpermanent_snapshots(h)
            names = [s.name for s in nonperm]
            # The three kept should be the last 3 (iter_03, iter_04, iter_05)
            assert all("iter_0" not in n and "iter_1" not in n or
                       n.endswith("iter_03") or n.endswith("iter_04") or n.endswith("iter_05")
                       for n in names)
            assert len(nonperm) == 3

    def test_max_snapshots_1_keeps_only_latest_nonpermanent(self):
        """With max_snapshots=1, only the single most recent non-permanent snapshot is kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=1)
            _init_handler(h)

            for i in range(10):
                h.take_snapshot(f"iter_{i:02d}", permanent=False)

            nonperm = _nonpermanent_snapshots(h)
            assert len(nonperm) == 1
            assert "iter_09" in nonperm[0].name

    def test_no_permanent_snapshots_behaves_like_original(self):
        """Without any permanent snapshots, cleanup behaves exactly as before the change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Disable permanent init by not using save_enabled path
            h = DataHandler(
                directory=tmpdir,
                device="cpu",
                dtype=torch.float64,
                d=3,
                max_snapshots=4,
            )
            X = _simplex_points(5, 3)
            Y = torch.arange(5, dtype=torch.float64).unsqueeze(1)
            bounds = torch.zeros(2, 3, dtype=torch.float64)
            bounds[1] = 1.0
            h.save_init(X, X, Y, bounds)  # This creates permanent init

            # Override: manually remove permanent marker from init to simulate
            # the old behavior (all non-permanent)
            for s in _all_snapshot_dirs(h):
                marker = s / "permanent"
                if marker.exists():
                    marker.unlink()

            for i in range(10):
                h.take_snapshot(f"step_{i}", permanent=False)

            all_snaps = _all_snapshot_dirs(h)
            assert len(all_snaps) <= 4

    def test_permanent_snapshots_not_counted_toward_limit(self):
        """Permanent snapshots don't consume slots from the max_snapshots budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=3)
            _init_handler(h)  # 1 permanent

            # Add 5 permanent events
            for i in range(5):
                h.take_snapshot(f"perm_event_{i}", permanent=True)

            # Now add non-permanent; should still keep 3 non-permanent
            for i in range(10):
                h.take_snapshot(f"iter_{i:02d}", permanent=False)

            nonperm = _nonpermanent_snapshots(h)
            assert len(nonperm) == 3


# ---------------------------------------------------------------------------
# 5. ZoMBI-Hop event labels stress test
# ---------------------------------------------------------------------------

class TestZoMBIHopEventLabels:
    """
    Simulate a realistic ZoMBI-Hop run:
      init (permanent) → many iterations → needle (permanent) →
      more iterations → failed (permanent) → more iterations → final (permanent)
    Then verify all permanent events are intact and non-permanent count ≤ max_snapshots.
    """

    def _simulate_activation(self, h: DataHandler, activation: int,
                              n_zooms: int = 2, n_iters: int = 8,
                              needle_at_iter: int = 5):
        """Simulate one activation: zooms, iterations, and a needle event."""
        for zoom in range(n_zooms):
            for iteration in range(n_iters):
                h.update_iteration_state(activation, zoom, iteration, 0)
                h.take_snapshot(f"act{activation}_z{zoom}_i{iteration}", permanent=False)
                if zoom == n_zooms - 1 and iteration == needle_at_iter:
                    h.take_snapshot(
                        f"act{activation}_z{zoom}_i{iteration}_needle", permanent=True
                    )
                    return  # needle found, stop

    def test_full_run_simulation(self):
        """
        Three activations with needle events, then overpenalization finish.
        All permanent events survive; non-permanent capped at max_snapshots.
        """
        max_snaps = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=max_snaps)
            _init_handler(h)  # permanent init

            expected_permanent_labels = []
            for act in range(3):
                self._simulate_activation(h, act)
                label = f"act{act}_z1_i5_needle"
                expected_permanent_labels.append(label)

            # Final overpenalization
            h.take_snapshot("act3_z0_finished", permanent=True)
            expected_permanent_labels.append("act3_z0_finished")

            h.take_snapshot("final", permanent=True)
            expected_permanent_labels.append("final")

            perm_names = {s.name for s in _permanent_snapshots(h)}
            assert any("init" in n for n in perm_names)
            for label in expected_permanent_labels:
                assert any(label in n for n in perm_names), (
                    f"Expected permanent snapshot '{label}' not found. "
                    f"Permanent snapshots: {perm_names}"
                )

            nonperm = _nonpermanent_snapshots(h)
            assert len(nonperm) <= max_snaps

    def test_timeout_snapshots_permanent(self):
        """Activation-level and iteration-level timeout snapshots are permanent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=2)
            _init_handler(h)

            # Simulate many iterations before timeout
            for i in range(20):
                h.take_snapshot(f"act0_z0_i{i}", permanent=False)

            h.take_snapshot("act0_z0_i15_timeout", permanent=True)
            h.take_snapshot("act0_timeout", permanent=True)

            # More iterations after (different activation)
            for i in range(20):
                h.take_snapshot(f"act1_z0_i{i}", permanent=False)

            perm_names = {s.name for s in _permanent_snapshots(h)}
            assert any("act0_z0_i15_timeout" in n for n in perm_names)
            assert any("act0_timeout" in n for n in perm_names)

    def test_failed_activation_snapshot_permanent(self):
        """Failed activation snapshots (_failed) survive aggressive cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=2)
            _init_handler(h)

            for i in range(15):
                h.take_snapshot(f"act0_z0_i{i}", permanent=False)

            h.take_snapshot("act0_z0_i8_failed", permanent=True)

            for i in range(15):
                h.take_snapshot(f"act1_z0_i{i}", permanent=False)

            perm_names = {s.name for s in _permanent_snapshots(h)}
            assert any("act0_z0_i8_failed" in n for n in perm_names)

    def test_zoomed_out_snapshot_permanent(self):
        """Zoom-out (overpenalization) snapshots survive cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=2)
            _init_handler(h)

            for i in range(15):
                h.take_snapshot(f"act0_z0_i{i}", permanent=False)

            h.take_snapshot("act0_z1_zoomed_out", permanent=True)

            for i in range(15):
                h.take_snapshot(f"act1_z0_i{i}", permanent=False)

            perm_names = {s.name for s in _permanent_snapshots(h)}
            assert any("zoomed_out" in n for n in perm_names)


# ---------------------------------------------------------------------------
# 6. Stress tests
# ---------------------------------------------------------------------------

class TestStressCleanup:

    def test_stress_many_iterations_permanent_invariant(self):
        """
        500 non-permanent + 20 permanent snapshots with max_snapshots=10.
        After all saves: exactly 20 permanent, at most 10 non-permanent.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=10)
            _init_handler(h)

            perm_count = 1  # init is permanent
            for i in range(500):
                is_perm = (i % 25 == 0)  # permanent every 25 steps
                h.take_snapshot(f"step_{i:03d}", permanent=is_perm)
                if is_perm:
                    perm_count += 1

            actual_perm = len(_permanent_snapshots(h))
            actual_nonperm = len(_nonpermanent_snapshots(h))

            assert actual_perm == perm_count, (
                f"Expected {perm_count} permanent, got {actual_perm}"
            )
            assert actual_nonperm <= 10, (
                f"Expected ≤10 non-permanent, got {actual_nonperm}"
            )

    def test_stress_alternating_permanent_nonpermanent(self):
        """
        Strictly alternating permanent/non-permanent with max_snapshots=5.
        All permanent survive; non-permanent capped at 5.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=5)
            _init_handler(h)

            n_each = 50
            for i in range(n_each):
                h.take_snapshot(f"perm_{i:02d}", permanent=True)
                h.take_snapshot(f"nonperm_{i:02d}", permanent=False)

            perm = _permanent_snapshots(h)
            nonperm = _nonpermanent_snapshots(h)

            # init + n_each permanents
            assert len(perm) == n_each + 1
            assert len(nonperm) <= 5

    def test_stress_burst_then_permanent_then_burst(self):
        """
        Large burst of non-permanent, then one permanent, then another burst.
        The sandwiched permanent must survive both bursts of cleanup.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=3)
            _init_handler(h)

            # First burst
            for i in range(100):
                h.take_snapshot(f"burst1_i{i:03d}", permanent=False)

            # Important needle event in the middle
            h.take_snapshot("needle_between_bursts", permanent=True)

            # Second burst — this triggers lots of additional cleanup
            for i in range(100):
                h.take_snapshot(f"burst2_i{i:03d}", permanent=False)

            perm_names = {s.name for s in _permanent_snapshots(h)}
            assert any("needle_between_bursts" in n for n in perm_names)
            assert len(_nonpermanent_snapshots(h)) <= 3

    def test_stress_all_permanent_no_deletion(self):
        """When all snapshots are permanent, nothing is ever deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=5)
            _init_handler(h)  # permanent

            for i in range(30):
                h.take_snapshot(f"all_perm_{i:02d}", permanent=True)

            all_snaps = _all_snapshot_dirs(h)
            # init + 30 permanents = 31
            assert len(all_snaps) == 31

    def test_stress_snapshot_count_monotonically_increases(self):
        """_snapshot_count always increases, never resets during a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h = _make_handler(tmpdir, max_snapshots=3)
            _init_handler(h)

            prev_count = h._snapshot_count
            for i in range(50):
                is_perm = (i % 7 == 0)
                h.take_snapshot(f"s{i}", permanent=is_perm)
                assert h._snapshot_count > prev_count
                prev_count = h._snapshot_count


# ---------------------------------------------------------------------------
# 7. Reload correctness after cleanup
# ---------------------------------------------------------------------------

class TestReloadAfterCleanup:

    def test_reload_loads_latest_after_nonpermanent_cleanup(self):
        """After non-permanent cleanup, load_state still loads the latest snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, max_snapshots=3)
            _init_handler(h1)

            for i in range(10):
                h1.update_iteration_state(0, 0, i, 0)
                h1.take_snapshot(f"act0_z0_i{i}", permanent=False)

            # Only last 3 non-permanent remain + permanent init
            h2 = _make_handler(tmpdir, run_uuid=h1.run_uuid, max_snapshots=3)
            act, zoom, iteration, _ = h2.load_state()

            # Should load the latest: iteration 9
            assert iteration == 9

    def test_reload_loads_latest_when_latest_is_permanent(self):
        """load_state loads the latest snapshot even if it's a permanent one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, max_snapshots=2)
            _init_handler(h1)

            for i in range(5):
                h1.update_iteration_state(0, 0, i, 0)
                h1.take_snapshot(f"act0_z0_i{i}", permanent=False)

            # Permanent needle is the latest
            h1.update_iteration_state(0, 0, 7, 0)
            h1.take_snapshot("act0_z0_i7_needle", permanent=True)

            h2 = _make_handler(tmpdir, run_uuid=h1.run_uuid, max_snapshots=2)
            act, zoom, it, _ = h2.load_state()
            assert it == 7

    def test_reload_data_intact_after_permanent_cleanup(self):
        """Reloaded X_all_actual and Y_all match the state at the latest snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = _make_handler(tmpdir, d=d, max_snapshots=3)
            X_init, Y_init, _ = _init_handler(h1, d=d)

            # Add data and save many non-permanent snapshots
            X_extra = _simplex_points(5, d, seed=42)
            Y_extra = torch.ones(5, 1, dtype=torch.float64) * 99.0
            h1.add_all_points(X_extra, X_extra.clone(), Y_extra)

            for i in range(10):
                h1.take_snapshot(f"iter_{i}", permanent=False)

            # Expected full data
            X_expected = h1.X_all_actual.clone()
            Y_expected = h1.Y_all.clone()

            h2 = _make_handler(tmpdir, d=d, run_uuid=h1.run_uuid, max_snapshots=3)
            h2.load_state()

            assert torch.allclose(h2.X_all_actual, X_expected)
            assert torch.allclose(h2.Y_all, Y_expected)

    def test_reload_after_cleanup_finds_correct_snapshot_count(self):
        """After reload, _snapshot_count is set correctly from existing snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h1 = _make_handler(tmpdir, max_snapshots=3)
            _init_handler(h1)

            for i in range(10):
                h1.take_snapshot(f"step_{i}", permanent=False)

            count_before_cleanup = h1._snapshot_count

            h2 = _make_handler(tmpdir, run_uuid=h1.run_uuid, max_snapshots=3)
            h2.load_state()

            # Snapshot count is set from the last *existing* snapshot number on disk
            # which may be lower than count_before due to cleanup
            assert h2._snapshot_count > 0
            # New snapshot after reload should not collide
            h2.take_snapshot("post_reload")
            names = sorted(p.name for p in (h2.run_dir / "snapshots").iterdir())
            counts = [int(n.split("_")[0]) for n in names]
            assert counts == sorted(set(counts)), "Snapshot numbers have duplicates"

    def test_permanent_snapshot_data_loadable_after_cleanup(self):
        """A permanent snapshot from an early activation is still loadable by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = 3
            h1 = _make_handler(tmpdir, d=d, max_snapshots=2)
            _init_handler(h1, d=d)

            # Early needle
            h1.update_iteration_state(0, 0, 3, 0)
            h1.take_snapshot("act0_z0_i3_needle", permanent=True)
            needle_snapshot_name = [
                s.name for s in _all_snapshot_dirs(h1)
                if "act0_z0_i3_needle" in s.name
            ][0]

            # Many more iterations to trigger aggressive cleanup
            for i in range(50):
                h1.update_iteration_state(1, 0, i, 0)
                h1.take_snapshot(f"act1_z0_i{i}", permanent=False)

            # Permanent needle must still be on disk
            perm_names = {s.name for s in _permanent_snapshots(h1)}
            assert needle_snapshot_name in perm_names

            # That snapshot dir must contain valid tensors
            snap_dir = h1.run_dir / "snapshots" / needle_snapshot_name
            assert (snap_dir / "tensors.pt").exists()
            assert (snap_dir / "summary.json").exists()
            tensors = torch.load(snap_dir / "tensors.pt", map_location="cpu")
            assert "X_all_actual" in tensors


# ---------------------------------------------------------------------------
# 8. num_iterations_saved wiring through ZoMBIHop
# ---------------------------------------------------------------------------

class TestNumIterationsSavedWiring:
    """
    Verify ZoMBIHop passes num_iterations_saved → DataHandler.max_snapshots
    without running the full optimizer.
    """

    def _make_zombihop(self, tmpdir, **kwargs):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.core.zombihop import ZoMBIHop

        d = 3
        X = _simplex_points(5, d)
        Y = torch.randn(5, 1, dtype=torch.float64)
        bounds = torch.zeros(2, d, dtype=torch.float64)
        bounds[1] = 1.0

        def dummy_objective(x, bounds, acq_fn):
            X_e = x.unsqueeze(0).clone()
            X_a = x.unsqueeze(0).clone()
            Y_o = torch.tensor([0.5], dtype=torch.float64)
            return X_e, X_a, Y_o

        return ZoMBIHop(
            objective=dummy_objective,
            bounds=bounds,
            X_init_actual=X,
            X_init_expected=X,
            Y_init=Y,
            device="cpu",
            dtype=torch.float64,
            checkpoint_dir=tmpdir,
            verbose=False,
            **kwargs,
        )

    def test_num_iterations_saved_sets_max_snapshots(self):
        """num_iterations_saved=7 results in DataHandler.max_snapshots == 7."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zh = self._make_zombihop(tmpdir, num_iterations_saved=7)
            assert zh.data_handler.max_snapshots == 7

    def test_default_num_iterations_saved_is_50(self):
        """Default num_iterations_saved results in DataHandler.max_snapshots == 50."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zh = self._make_zombihop(tmpdir)
            assert zh.data_handler.max_snapshots == 50

    def test_explicit_max_snapshots_overrides_num_iterations_saved(self):
        """Explicit max_snapshots=3 takes priority over num_iterations_saved=50."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zh = self._make_zombihop(tmpdir, num_iterations_saved=50, max_snapshots=3)
            assert zh.data_handler.max_snapshots == 3

    def test_num_iterations_saved_limits_nonpermanent_in_practice(self):
        """DataHandler with max_snapshots from num_iterations_saved actually limits cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zh = self._make_zombihop(tmpdir, num_iterations_saved=4)
            dh = zh.data_handler

            for i in range(20):
                dh.update_iteration_state(0, 0, i, 0)
                dh.take_snapshot(f"act0_z0_i{i}", permanent=False)

            nonperm = _nonpermanent_snapshots(dh)
            assert len(nonperm) <= 4
