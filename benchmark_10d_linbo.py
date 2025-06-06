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


def ackley10(x: np.ndarray) -> np.ndarray:
    """10‑D Ackley (global minimum at 0 when x in [0,1]^10)."""
    a, b, c = 20.0, 0.2, 2 * np.pi
    d = x.shape[1]
    sum_sq = np.sum((x - 0.5) ** 2, axis=1) * 4  # centre the well at 0.5
    sum_cos = np.sum(np.cos(c * (x - 0.5)), axis=1)
    fx = a + np.e - a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d)
    return fx.reshape(-1, 1)


def multiwell_ackley(x: np.ndarray, extra_wells: int = 2, depth: float = 5.0) -> np.ndarray:
    """Ackley with additional Gaussian wells of varying depths.

    Parameters
    ----------
    x           : (n,10) array on simplex
    extra_wells : how many secondary wells to add (random centres)
    depth       : depth (positive) of those wells relative to baseline Ackley
    """
    base = ackley10(x).ravel()
    rng = np.random.default_rng(42)

    # fixed random centres for reproducibility
    centres = rng.dirichlet(np.ones(x.shape[1]), size=extra_wells)
    widths = 0.1  # shared width in L2 norm

    penalty = np.zeros_like(base)
    for c in centres:
        dist2 = np.sum((x - c) ** 2, axis=1)
        penalty -= depth * np.exp(-dist2 / (2 * widths ** 2))

    return (base + penalty).reshape(-1, 1)


def dirichlet_init(D: int, n_pts: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.ones(D), size=n_pts)
    X = np.round(X, 3)
    X[:, -1] = 1.0 - X[:, :-1].sum(axis=1)  # enforce simplex exactly
    Y = np.zeros((n_pts, 1))
    return pd.DataFrame(X), pd.DataFrame(Y)


def run_one_objective(obj_fun, label: str, args):
    D = 10
    X_init, Y_init = dirichlet_init(D, args.init_pts, args.seed)

    zombi = ZombiHop(
        seed=args.seed,
        X_init=X_init,
        Y_init=Y_init,
        Y_experimental=obj_fun,
        Gammas=args.gammas,
        alphas=args.alphas,
        n_draws_per_activation=args.draws,
        acquisition_type=LCB_ada,
        tolerance=0.9,
        penalty_width=0.3,
        m=5,
        k=5,
        lower_bound=np.zeros(D),
        upper_bound=np.ones(D),
        resolution=args.res,
        sampler=None,
    )

    best_hist = []
    time_hist = []
    t0 = time.perf_counter()

    orig_progress = utils.progress_bar  # type: ignore

    def pb(n, T, inc):
        val = orig_progress(n, T, inc)
        # record after every experiment
        if n % 1 == 0:
            best_hist.append(float(zombi.Y_init.min()))
            time_hist.append(time.perf_counter() - t0)
        return val

    utils.progress_bar = pb

    zombi.run_experimental(
        n_droplets=args.draws,
        n_vectors=args.n_vectors,
        verbose=False,
        plot=False,
    )

    best_hist = np.asarray(best_hist)
    time_hist = np.asarray(time_hist)

    # Plot convergence
    plt.figure(figsize=(6, 4))
    plt.plot(best_hist, lw=1.5)
    plt.xlabel("number of experiments")
    plt.ylabel("best objective (lower is better)")
    plt.title(f"Convergence – {label}")
    plt.grid(True)
    fname = Path(f"convergence_{label}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot wall‑clock
    plt.figure(figsize=(6, 4))
    plt.plot(time_hist, best_hist, lw=1.5)
    plt.xlabel("wall‑clock time [s]")
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
    run_one_objective(ackley10, "ackley_single", args)

    # 2) Multi‑well variant
    if args.wells > 0:
        def obj(x):
            return multiwell_ackley(x, extra_wells=args.wells)
        run_one_objective(obj, f"ackley_{args.wells}wells", args)


if __name__ == "__main__":
    main()
