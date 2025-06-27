from __future__ import annotations
import time, sys, pathlib
import numpy as np, pandas as pd

from zombihop import ZombiHop
from acquisitions import LCB_ada
from benchmark_10d_linbo import ackley10_simplex, multiwell_ackley10_simplex, dirichlet_init
import matplotlib.pyplot as plt


def run_once(k_keep: int, ax = None, seed: int = 1) -> dict:
    """Run ZoMBI-Hop once and return a dict with runtime and best value."""
    D, DROPS, RES, GAMMAS, ALPHAS = 10, 30, 3, 10, 10
    X0, Y0 = dirichlet_init(lambda x: multiwell_ackley10_simplex(x, depths=[5,10,15], extra_wells=3), D, 300, 1)
    zombi = ZombiHop(
        seed           = seed,
        X_init         = X0,
        Y_init         = Y0,
        Y_experimental = lambda x: multiwell_ackley10_simplex(x, depths=[5,10,15], extra_wells=3),
        Gammas         = GAMMAS,
        alphas         = ALPHAS,
        n_draws_per_activation = 10,
        acquisition_type       = LCB_ada,
        tolerance      = 0.15,
        penalty_width  = 0.1,
        m              = 5,
        k              = k_keep,            # ← variable we sweep
        lower_bound    = np.zeros(D),
        upper_bound    = np.ones(D),
        resolution     = RES,
        sampler        = None,
    )

    t0 = time.perf_counter()
    X_all, Y_all, *_ = zombi.run_experimental(
        n_droplets = DROPS,
        n_vectors  = 11,
        verbose    = False,
        plot       = False,
    )
    elapsed = time.perf_counter() - t0
    best_hist = np.minimum.accumulate(Y_all.values.ravel())

    if ax is not None:
        ax.plot(best_hist,  lw=1.4, label=f"k = {k_keep}")
        ax.set_xlabel("number of experiments")
        ax.set_ylabel("best objective")
        ax.set_title("Pruning comparison")
        ax.grid(True)

    # also save individual curve for later inspection
    outdir = pathlib.Path("pruning_curves")
    outdir.mkdir(exist_ok=True)
    np.save(outdir / f"best_hist_k{k_keep}.npy", best_hist)

    return best_hist, elapsed
# ─────────────────── driver ───────────────────────────────────────────────
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6,4))
    ks = [3,5,10,20,30,100]
    runtimes = [run_once(k, ax=ax, seed=10)[1] for k in ks]
    print(runtimes)
    ax.legend(title="top-k kept")
    fig.tight_layout()
    fig.savefig("pruning_comparison_1.png", dpi=150)
    plt.show()