"""
Comprehensive test of acquisition optimizer (projected gradient ascent on simplex).

Run: python -m scripts.test_acquisition_optimizer

Runs the same logical checks as tests/test_gp_simplex.py::TestAcquisitionOptimizerInterior:
- Interior maximum: acquisition max at center -> find interior point
- Vertex maximum: acquisition max at vertex -> find vertex
- All candidates on simplex (sum=1, non-negative)
- Bounds respected when using restricted bounds
- 3D interior maximum
- More steps improve or maintain convergence
- Flat acquisition does not crash
- Values match acquisition at candidates

If all pass: SGD + projection are working; edge-only suggestions in ZoMBIHop
are due to the acquisition (e.g. LogEI) favoring vertices.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from utils.datahandler import DataHandler
from utils.gp_simplex import GPSimplex
from utils.simplex import proj_simplex


def make_gp_2d():
    handler = DataHandler(directory=None, device="cpu", dtype=torch.float64, d=2)
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


def make_gp_3d():
    handler = DataHandler(directory=None, device="cpu", dtype=torch.float64, d=3)
    X = torch.tensor(
        [[0.33, 0.33, 0.34], [0.5, 0.25, 0.25], [0.2, 0.4, 0.4]], dtype=torch.float64
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


def make_quadratic_acquisition(center, scale=-1.0):
    class QuadraticAcquisition(torch.nn.Module):
        def forward(self, X):
            x = X.squeeze(1) if X.dim() == 3 else X
            diff = x - center.to(x.device)
            return scale * (diff ** 2).sum(dim=-1)

    return QuadraticAcquisition()


def run_tests():
    results = []
    tol_sum = 1e-5
    tol_nonneg = 1e-6

    # --- 1. Interior maximum (2D) ---
    try:
        gp = make_gp_2d()
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = make_quadratic_acquisition(center)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.6, 0.4]],
            dtype=torch.float64,
        ).unsqueeze(1)
        candidates, values = gp._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.15, max_steps=40
        )
        best = candidates[values.argmax().item()]
        dist = torch.norm(best - center).item()
        ok = (
            candidates.shape[0] > 0
            and dist < 0.15
            and abs(best.sum().item() - 1.0) < 0.01
        )
        results.append(("Interior maximum (2D)", ok, f"dist={dist:.4f}" if ok else f"dist={dist:.4f} (expected <0.15)"))
    except Exception as e:
        results.append(("Interior maximum (2D)", False, str(e)))

    # --- 2. Vertex maximum (2D) ---
    try:
        gp = make_gp_2d()
        center = torch.tensor([1.0, 0.0], dtype=torch.float64)
        acq = make_quadratic_acquisition(center)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor(
            [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]],
            dtype=torch.float64,
        ).unsqueeze(1)
        candidates, values = gp._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.1, max_steps=50
        )
        best = candidates[values.argmax().item()]
        dist = torch.norm(best - center).item()
        ok = (
            candidates.shape[0] > 0
            and dist < 0.2
            and abs(best.sum().item() - 1.0) < 0.01
            and (best >= -0.01).all()
            and (best <= 1.01).all()
        )
        results.append(("Vertex maximum (2D)", ok, f"dist={dist:.4f}" if ok else f"dist={dist:.4f}"))
    except Exception as e:
        results.append(("Vertex maximum (2D)", False, str(e)))

    # --- 3. All candidates on simplex ---
    try:
        gp = make_gp_2d()
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = make_quadratic_acquisition(center)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.6, 0.4]],
            dtype=torch.float64,
        ).unsqueeze(1)
        candidates, _ = gp._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.15, max_steps=30
        )
        ok = candidates.shape[0] > 0
        for i in range(candidates.shape[0]):
            c = candidates[i]
            ok = ok and abs(c.sum().item() - 1.0) < tol_sum and (c >= -tol_nonneg).all()
        results.append(("All on simplex", ok, f"{candidates.shape[0]} candidates"))
    except Exception as e:
        results.append(("All on simplex", False, str(e)))

    # --- 4. Bounds respected ---
    try:
        gp = make_gp_2d()
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = make_quadratic_acquisition(center)
        bounds = torch.tensor([[0.2, 0.2], [0.8, 0.8]], dtype=torch.float64)
        inits = torch.tensor(
            [[0.5, 0.5], [0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3]],
            dtype=torch.float64,
        ).unsqueeze(1)
        candidates, _ = gp._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.1, max_steps=30
        )
        lower, upper = bounds[0], bounds[1]
        ok = candidates.shape[0] > 0
        for i in range(candidates.shape[0]):
            c = candidates[i]
            ok = ok and (c >= lower - 1e-5).all() and (c <= upper + 1e-5).all()
        results.append(("Bounds respected", ok, f"{candidates.shape[0]} candidates"))
    except Exception as e:
        results.append(("Bounds respected", False, str(e)))

    # --- 5. Interior maximum (3D) ---
    try:
        gp = make_gp_3d()
        center = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=torch.float64)
        acq = make_quadratic_acquisition(center)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.33, 0.33, 0.34], [0.6, 0.2, 0.2]],
            dtype=torch.float64,
        ).unsqueeze(1)
        candidates, values = gp._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.1, max_steps=50
        )
        best = candidates[values.argmax().item()]
        dist = torch.norm(best - center).item()
        ok = (
            candidates.shape[0] > 0
            and dist < 0.2
            and abs(best.sum().item() - 1.0) < 0.01
        )
        results.append(("Interior maximum (3D)", ok, f"dist={dist:.4f}" if ok else f"dist={dist:.4f}"))
    except Exception as e:
        results.append(("Interior maximum (3D)", False, str(e)))

    # --- 6. More steps improve convergence ---
    try:
        gp = make_gp_2d()
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = make_quadratic_acquisition(center)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]], dtype=torch.float64).unsqueeze(1)
        _, values_short = gp._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.1, max_steps=10
        )
        _, values_long = gp._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.1, max_steps=60
        )
        best_short = values_short.max().item()
        best_long = values_long.max().item()
        ok = best_long >= best_short - 1e-6
        results.append(("More steps improve", ok, f"short={best_short:.4f} long={best_long:.4f}"))
    except Exception as e:
        results.append(("More steps improve", False, str(e)))

    # --- 7. Flat acquisition no crash ---
    try:
        gp = make_gp_2d()
        class FlatAcquisition(torch.nn.Module):
            def forward(self, X):
                x = X.squeeze(1) if X.dim() == 3 else X
                return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        acq = FlatAcquisition()
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.6, 0.4]],
            dtype=torch.float64,
        ).unsqueeze(1)
        candidates, values = gp._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.05, max_steps=20
        )
        ok = candidates.shape[0] > 0
        for i in range(candidates.shape[0]):
            ok = ok and abs(candidates[i].sum().item() - 1.0) < tol_sum
        results.append(("Flat acquisition no crash", ok, f"{candidates.shape[0]} candidates"))
    except Exception as e:
        results.append(("Flat acquisition no crash", False, str(e)))

    # --- 8. Values match acquisition at candidates ---
    try:
        gp = make_gp_2d()
        center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        acq = make_quadratic_acquisition(center)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        inits = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.6, 0.4]],
            dtype=torch.float64,
        ).unsqueeze(1)
        candidates, values = gp._optimize_acquisition(
            acq=acq, bounds=bounds, initial_conditions=inits, step_size=0.15, max_steps=30
        )
        ok = True
        with torch.no_grad():
            for i in range(candidates.shape[0]):
                x = candidates[i].unsqueeze(0).unsqueeze(0)
                expected = acq(x).squeeze().item()
                actual = values[i].item()
                ok = ok and abs(expected - actual) < 1e-5
        results.append(("Values match acquisition", ok, ""))
    except Exception as e:
        results.append(("Values match acquisition", False, str(e)))

    return results


def main():
    print("Acquisition optimizer (SGD + simplex projection) — comprehensive checks\n")
    results = run_tests()
    all_ok = True
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        detail = f" — {msg}" if msg else ""
        print(f"  [{status}] {name}{detail}")
    print()
    if all_ok:
        print("All checks passed. SGD + projection are working; edge-only suggestions")
        print("in ZoMBIHop are due to the acquisition favoring vertices, not broken optimization.")
    else:
        print("Some checks failed. Review projection, step size, or acquisition.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
