---
name: ZoMBI-Hop Major Refactor
overview: "Four coordinated changes across src/ and scripts/: streamline state snapshotting, fix the DB handshake race condition, replace the spherical penalization sweep with a Hessian-ellipsoid method, and introduce an \"old_needles\" over-penalization escape hatch with constrained-hyperrectangle bounds computation."
todos:
  - id: snapshot-streamline
    content: Single take_snapshot after _objective_wrapper returns; remove all other mid-iteration snapshot calls; fix prev_best_Y typo
    status: completed
  - id: handshake-fix
    content: Fix obj_empty never-set bug in communication.py; swap reset_objective before write_compositions in run_zombi_main.py
    status: completed
  - id: ellipsoid-gp
    content: Add determine_penalty_ellipsoid + cached tangent basis + Mahalanobis RepulsiveAcquisition in gp_simplex.py
    status: completed
  - id: ellipsoid-dh
    content: Add needle_M_list, needle_B storage, ellipsoid penalty mask, and serialization in datahandler.py
    status: completed
  - id: old-needles-dh
    content: Add old_needles storage, move_needles_to_old(), and determine_new_bounds_constrained() in datahandler.py
    status: completed
  - id: retry-zombi
    content: Add retry logic (move_needles_to_old + scale raw/step), constrained bounds call, and ellipsoid add_needle call in zombihop.py
    status: completed
isProject: false
---

# ZoMBI-Hop Major Refactor Plan

## 1. Streamline State Snapshotting

**Rule:** One snapshot per received objective, taken immediately after `_objective_wrapper` returns in `zombihop.py`. No other mid-iteration snapshots. Permanent snapshots (needle found, timeout, final) are the exception.

**Fix — [`src/utils/datahandler.py`](zombi_replace/src/utils/datahandler.py)**
- Change `take_snapshot` signature to `take_snapshot(label, activation, zoom, iteration, permanent=False)` — it sets `current_activation/zoom/iteration` internally before saving, replacing the separate `update_iteration_state` call. Keep the old zero-arg form for backward compat.

**Fix — [`src/core/zombihop.py`](zombi_replace/src/core/zombihop.py)**
- Remove all `update_iteration_state(…) + take_snapshot(…)` pairs from within the zoom/iteration loops.
- After each `_objective_wrapper` returns, call a single:
  ```python
  dh.take_snapshot(f"act{activation}_z{zoom}_i{iteration}", activation, zoom, iteration)
  ```
- Keep `permanent=True` snapshots only for: needle found, timeout, and final.
- Also fix the existing `NameError` bug on line 327: `prev_best_y` → `prev_best_Y`.

---

## 2. Verify/Fix the DB Handshake

**Analysis of current handshake correctness:**

```
write_compositions()   ← sends new request
reset_objective()      ← clears table and sets handshake=0
get_y_measurements()   ← polls handshake until =1, reads, sets =0
```

**Bug A — race condition (order):** `write_compositions` is called BEFORE `reset_objective`. If the apparatus responds immediately, the fresh data lands in the DB, then `reset_objective` wipes it, and `get_y_measurements` blocks forever.

**Bug B — broken `obj_empty` guard:** In `objective_receiver` (communication.py, lines 556–571), `obj_empty` is initialised to `True` and **never updated** from the `SELECT *` result. The guard that prevents overwriting an unread table never fires.

**Fixes — [`scripts/run_zombi_main.py`](zombi_replace/scripts/run_zombi_main.py)**
- In `objective()`: swap order → `reset_objective()` first, then `write_compositions()`.

**Fixes — [`scripts/communication.py`](zombi_replace/scripts/communication.py)**
- After `cur.execute("SELECT * FROM objective")` / `obj = cur.fetchall()`, add: `obj_empty = (len(obj) == 0)`.

---

## 3. Hessian-Ellipsoid Penalization

Replaces the radial sweep in `determine_penalty_radius` with one autograd Hessian call, giving anisotropic, principled basin shapes.

### 3a. New method in [`src/utils/gp_simplex.py`](zombi_replace/src/utils/gp_simplex.py)

```python
def determine_penalty_ellipsoid(
    self, needle, drop_fraction=0.25, eigenvalue_floor=1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (M, B). Point x is in basin iff (B^T(x-needle))^T M (B^T(x-needle)) <= 1."""
    d = needle.shape[0]
    # Build tangent basis B (cached on self._tangent_basis)
    P = torch.eye(d) - 1/d
    Q, _ = torch.linalg.qr(P)
    B = Q[:, :d-1]  # (d, d-1)
    # Tangent-space acquisition
    def tilde_alpha(u):
        return self.acq_fn((needle + B @ u).view(1,1,d)).squeeze()
    u0 = torch.zeros(d-1, device=self.device, dtype=self.dtype)
    H = torch.autograd.functional.hessian(tilde_alpha, u0)
    neg_H = -0.5*(H + H.T)
    eigvals, eigvecs = torch.linalg.eigh(neg_H)
    eigvals = eigvals.clamp(min=eigenvalue_floor)
    alpha_peak = abs(self.acq_fn(needle.view(1,1,d)).squeeze().item())
    sigma = self.data_handler.get_input_noise()
    lambda_max = eigvals.max().item()
    Delta = max(drop_fraction * alpha_peak,
                0.5 * lambda_max * (3.0 * sigma)**2)
    M = (eigvecs @ torch.diag(eigvals) @ eigvecs.T) / (2.0 * Delta)
    return M, B
```

- Cache `B` as `self._tangent_basis`; rebuild only when `d` changes.
- Keep `determine_penalty_radius` (it's called in old checkpoints / tests).

Update `RepulsiveAcquisition.forward` to compute Mahalanobis violation instead of Euclidean when ellipsoid params are available:

```python
# violation = max(0, 1 - u^T M u) per needle
u = (X_flat - needle) @ B          # (B_points, d-1)
quad = (u @ M * u).sum(-1)          # (B_points,)
violation = torch.clamp(1.0 - quad, min=0.0)
```

### 3b. Storage in [`src/utils/datahandler.py`](zombi_replace/src/utils/datahandler.py)

Add in `_init_storage`:
```python
self.needle_M_list: List[Optional[torch.Tensor]] = []  # per-needle (d-1,d-1)
self.needle_B: Optional[torch.Tensor] = None           # shared (d, d-1)
```

- `add_needle(…, M=None, B=None)`: store M in `needle_M_list`, set `needle_B = B` if provided.
- `_compute_penalty_mask`: for each needle that has an M, use `u^T M u <= 1`; fall back to sphere for None.
- `take_snapshot`: stack M matrices with `torch.stack` (or save `None` as zeros + a flag tensor) in `tensors.pt`. `_load_tensors` restores them.
- `get_needles_and_ellipsoids()` → returns `(needles, M_list, B)` for the repulsive acquisition.

### 3c. Call site in [`src/core/zombihop.py`](zombi_replace/src/core/zombihop.py)

```python
M, B = self.gp_handler.determine_penalty_ellipsoid(needle_X)
dh.add_needle(needle_X, needle_Y.item(), needle_penalty_radius=0.0,
              activation=activation, zoom=zoom, iteration=iteration,
              M=M, B=B)
```

Also pass `M_list, B` to `create_acquisition` so `RepulsiveAcquisition` gets updated ellipsoids.

---

## 4. Over-Penalization Escape Hatch

### 4a. "Old Needles" in [`src/utils/datahandler.py`](zombi_replace/src/utils/datahandler.py)

New fields in `_init_storage`:
```python
self.old_needles: Optional[torch.Tensor] = None        # (k, d)
self.old_needle_vals: Optional[torch.Tensor] = None    # (k, 1)
self.old_needle_radii: Optional[torch.Tensor] = None   # (k, 1) = mult * sigma
self.old_needle_radius_mult: float = 3.0               # configurable
```

New method:
```python
def move_needles_to_old(self):
    """Move all current needles to old_needles with sphere r = mult*input_noise. Reset current needles."""
    sigma = self.get_input_noise()
    r = self.old_needle_radius_mult * sigma
    # Append current needles to old_needles
    ...
    # Reset self.needles, needle_vals, needle_M_list, etc. to empty
    self._update_penalty_mask()
```

`_compute_penalty_mask` change:
- Regular needles: ellipsoid test (or sphere fallback)
- Old needles: sphere test with `old_needle_radii`
- Both types exclude the point from the unpenalized mask

Serialization: old_needles tensors go into `tensors.pt`; `_load_tensors` restores them.

### 4b. Constrained hyperrectangle in [`src/utils/datahandler.py`](zombi_replace/src/utils/datahandler.py)

Replaces `determine_new_bounds` for the post-convergence bounds update:

```python
def determine_new_bounds_constrained(
    self,
    converged_point: torch.Tensor,
    global_bounds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Largest axis-aligned hyperrectangle R containing converged_point such that:
      1. R does not intersect any regular-needle ellipsoid.
      2. No old_needle is strictly interior to R.
    Algorithm (per-dimension, O(n_needles * d)):
      - Initialise lo[i] = 0, hi[i] = 1 (or global_bounds).
      - For each regular needle, compute its ellipsoid AABB:
          extent_i = sqrt((B M^{-1} B^T)[i,i])
          lo_n[i] = needle[i] - extent_i, hi_n[i] = needle[i] + extent_i
        If the needle is on the same side as converged_point, tighten lo or hi.
      - For each old_needle that is interior to the current box, tighten
        the bound in the dimension with the smallest margin to push the
        old_needle to the boundary.
    """
```

Ellipsoid AABB extent per dimension: `extent_i = sqrt((B @ M_inv @ B.T)[i, i])` where `M_inv = torch.linalg.inv(M)`.

Old-needle boundary constraint: if `all(lo[i] < old_n[i] < hi[i])`, find `i* = argmin(min(old_n[i]-lo[i], hi[i]-old_n[i]))` and set that bound to `old_n[i*]`.

### 4c. Retry logic in [`src/core/zombihop.py`](zombi_replace/src/core/zombihop.py)

New activation-level state: `retry_attempted: bool` per activation.

```python
# After inner zoom loop, if activation_failed and no needle found and not retry_attempted:
if activation_failed and needle is None and not retry_attempted:
    dh.move_needles_to_old()
    # Scale up samples and scale down step for this retry
    saved_raw = dh.raw; saved_step = dh.nat_grad_step
    dh.raw = int(dh.raw * retry_raw_scale)           # e.g. 2.0
    dh.nat_grad_step *= retry_step_scale              # e.g. 0.5
    self.gp_handler.raw_samples = dh.raw
    self.gp_handler.nat_grad_step = dh.nat_grad_step
    retry_attempted = True
    zoom = 0; bounds = self.bounds.clone()
    continue  # re-enter zoom loop for this activation
else:
    # Restore params, advance to next activation
    dh.raw = saved_raw; dh.nat_grad_step = saved_step
    ...
```

Two new configurable params on `ZoMBIHop.__init__` (and `DataHandler`): `retry_raw_scale=2.0`, `retry_step_scale=0.5`. Add `old_needle_radius_mult=3.0` as well.

Post-convergence bounds: call `dh.determine_new_bounds_constrained(needle_X)` instead of `dh.determine_new_bounds()` at the zoom-transition step.

---

## Files Changed

- [`src/utils/datahandler.py`](zombi_replace/src/utils/datahandler.py) — major (new storage, ellipsoid mask, hyperrectangle, old_needles, snapshot update)
- [`src/utils/gp_simplex.py`](zombi_replace/src/utils/gp_simplex.py) — major (ellipsoid determination, Mahalanobis repulsion)
- [`src/core/zombihop.py`](zombi_replace/src/core/zombihop.py) — moderate (snapshot streamlining, retry logic, constrained bounds, bugfix)
- [`scripts/communication.py`](zombi_replace/scripts/communication.py) — minor (obj_empty fix)
- [`scripts/run_zombi_main.py`](zombi_replace/scripts/run_zombi_main.py) — minor (reset_objective order fix)
