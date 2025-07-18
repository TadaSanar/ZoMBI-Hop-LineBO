"""
Pruning-efficiency benchmark for the **new ZoMBI-Hop + LineBO** implementation.

We sweep `max_gp_points` (≈ “top-k kept”) and record:

* the running best‐objective curve;
* wall-clock runtime;

Results:
* curves are plotted on a single figure;
* each curve is also saved to   pruning_curves/best_hist_k{K}.npy
"""

from __future__ import annotations
import time, pathlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from new_zombihop import ZoMBIHop
from acquisitions import LCB_ada
from computational_simulation.benchmark_10d_linbo import multiwell_ackley10_simplex, dirichlet_init

D, GAMMAS, RESOLUTION = 10, 2, 3
BASE_OBJ = lambda x: multiwell_ackley10_simplex(x, depths=[5,10,15,20,25], extra_wells=5, width=0.05, seed=1)

def make_line_objective(base_fun, n_experiments=10):
    """Wrap a point objective → LineBO batch objective."""
    def _fn(endpoints_batch):
        a, b  = np.asarray(endpoints_batch)         # shape (2, D)
        ts    = np.linspace(0., 1., n_experiments)  # along the line
        pts   = (1-ts[:,None])*a + ts[:,None]*b
        vals  = base_fun(pts).ravel()
        return pts, vals
    return _fn


def run_once(k_keep: int, ax=None, seed: int = 1) -> tuple[np.ndarray, float]:
    """Run ZoMBI-Hop once with GP dataset capped at `k_keep`."""
    X0, Y0 = dirichlet_init(BASE_OBJ, D, 30, seed)
    Y0 = Y0.ravel()

    zombi = ZoMBIHop(
        objective_function = make_line_objective(BASE_OBJ, n_experiments=10),
        dimensions         = D,
        X_init_actual      = X0,
        Y_init             = Y0,
        num_activations    = GAMMAS,
        max_gp_points      = k_keep,
        tolerance          = 0.06,
        resolution         = RESOLUTION,
        num_experiments    = 10,
        linebo_num_lines   = 50,
        bounds             = [(0.,1.)]*D,
    )

    t0 = time.perf_counter()
    zombi.run_zombi_hop(verbose=False)
    elapsed = time.perf_counter() - t0

    best_hist = np.minimum.accumulate(zombi.Y_all)     # running minimum

    # Plot on shared axis if provided
    if ax is not None:
        ax.plot(best_hist, lw=1.3, label=f"k = {k_keep}")

    # persist curve for later inspection
    outdir = pathlib.Path("pruning_curves"); outdir.mkdir(exist_ok=True)
    np.save(outdir / f"best_hist_k{k_keep}.npy", best_hist)

    return best_hist, elapsed


if __name__ == "__main__":
    ks        = [10, 20, 30, 50, 100]
    fig, ax   = plt.subplots(figsize=(6,4))

    runtimes  = [run_once(k, ax=ax, seed=10)[1] for k in ks]
    print("wall-clock seconds:", dict(zip(ks, runtimes)))

    ax.set_xlabel("number of experiments")
    ax.set_ylabel("best objective")
    ax.set_title("Pruning comparison")
    ax.grid(True);  ax.legend(title="max_gp_points")
    fig.tight_layout()
    fig.savefig("new_pruning_comparison.png", dpi=150)
    plt.show()


