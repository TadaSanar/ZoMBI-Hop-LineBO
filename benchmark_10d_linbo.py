"""
10‑D benchmarking harness for ZoMBI‑Hop + LineBO.
Runs two synthetic objectives on the 10‑component simplex:

1. Standard Ackley (one global minimum ≈ 0 at the centre).
2. Custom multi‑well Ackley variant with user‑defined well depths.

Usage
-----
$ python benchmark_10d_linebo.py            # default settings
$ python benchmark_10d_linebo.py --wells 3  # 3 additional wells
"""

import utils
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zombihop import ZombiHop
from acquisitions import LCB_ada
from sampler import line_bo_sampler



def ackley10_simplex(x: np.ndarray) -> np.ndarray:
    """
    Ackley where the global minimum (≈0) sits at the centre of the D-simplex,
    i.e.  x* = (1/D,…,1/D) with Σx_i = 1.
    """
    a, b, c = 20.0, 0.2, 2 * np.pi
    d       = x.shape[1]

    centre  = np.full(d, 1.0 / d, dtype=x.dtype)
    y       = x - centre

    sum_sq  = np.sum(y ** 2, axis=1)
    sum_cos = np.sum(np.cos(c * y), axis=1)

    fx = a + np.e \
         - a * np.exp(-b * np.sqrt(sum_sq / d)) \
         - np.exp(sum_cos / d)
    return fx.reshape(-1, 1)

def multiwell_ackley10_simplex(
    x:           np.ndarray,
    extra_wells: int        = 2,
    depths:      float | np.ndarray = 5.0,
    width:       float      = 0.07,
    seed:        int        = 42,
) -> np.ndarray:
    """
    Ackley-on-simplex plus `extra_wells` Gaussian wells of specified `depths`.

    Parameters
    ----------
    x           : (n, D) points on the simplex.
    extra_wells : number of secondary minima to add.
    depths      : scalar or array-like of length `extra_wells`; deeper ⇒ lower.
    width       : shared Gaussian width (L2 σ) of those wells.
    seed        : RNG seed for deterministic well locations.

    Returns
    -------
    f(x)        : (n, 1) array of objective values (lower is better).
    """
    base   = ackley10_simplex(x).ravel()

    # RNG for reproducible secondary-well centres
    rng     = np.random.default_rng(seed)
    centres = rng.dirichlet(np.ones(x.shape[1]), size=extra_wells)

    # broadcast depths → array[extra_wells]
    depths  = np.broadcast_to(np.asarray(depths, dtype=x.dtype), extra_wells)

    bonus   = np.zeros_like(base)
    for c, dpth in zip(centres, depths):
        dist2 = np.sum((x - c) ** 2, axis=1)
        bonus -= dpth * np.exp(-dist2 / (2 * width ** 2))

    return (base + bonus).reshape(-1, 1)


def dirichlet_init(obj_fun, D: int, n_pts: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.ones(D), size=n_pts)
    X = np.round(X, 3)
    X[:, -1] = 1.0 - X[:, :-1].sum(axis=1)  # enforce simplex exactly
    Y = obj_fun(X)
    return pd.DataFrame(X), pd.DataFrame(Y)


def run_one_objective(obj_fun, label: str, args):
    D = 10
    X_init, Y_init = dirichlet_init(obj_fun, D, args.init_pts, args.seed)

    zombi = ZombiHop(
        seed=args.seed,
        X_init=X_init,
        Y_init=Y_init,
        Y_experimental=obj_fun,
        Gammas=args.gammas,
        alphas=args.alphas,
        n_draws_per_activation=args.draws,
        acquisition_type=LCB_ada,
        tolerance=0.15,
        penalty_width=0.1,
        m=5,
        k=5,
        lower_bound=np.zeros(D),
        upper_bound=np.ones(D),
        resolution=args.res,
        sampler=None,
    )

    t0 = time.perf_counter()
    X_all, Y_all, *_ = zombi.run_experimental(
        n_droplets=args.draws,
        n_vectors=args.n_vectors,
        verbose=False,
        plot=False,
    )

    best_hist = np.minimum.accumulate(Y_all.values.ravel())
    time_hist = np.linspace(0, time.perf_counter() - t0, len(best_hist))

    plt.figure(figsize=(6, 4))
    plt.plot(best_hist, lw=1.5)
    plt.xlabel("number of experiments")
    plt.ylabel("best objective (lower is better)")
    plt.title(f"Convergence – {label}")
    plt.grid(True)
    plt.savefig(Path(f"convergence_{label}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(time_hist, best_hist, lw=1.5)
    plt.xlabel("wall-clock time [s]")
    plt.ylabel("best objective")
    plt.title(f"Time vs Experiments – {label}")
    plt.grid(True)
    plt.savefig(Path(f"time_{label}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plots for {label}")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wells", type=int, default=0, help="number of extra wells in custom Ackley")
    p.add_argument("--init-pts", type=int, default=30)
    p.add_argument("--gammas", type=int, default=10)
    p.add_argument("--alphas", type=int, default=10)
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--res", type=int, default=3)
    p.add_argument("--n-vectors", type=int, default=11)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Baseline single‑well Ackley
    run_one_objective(ackley10_simplex, "ackley_single", args)

    # 2) Multi‑well variant
    if args.wells > 0:
        def obj(x):
            return multiwell_ackley10_simplex(x, extra_wells=args.wells)
        run_one_objective(obj, f"ackley_{args.wells}wells", args)


if __name__ == "__main__":
    main()
