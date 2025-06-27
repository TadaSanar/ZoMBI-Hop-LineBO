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


def ackley10_simplex(x: np.ndarray, a=20.0, b=0.2) -> np.ndarray:
    """
    Standard Ackley shifted onto the D-simplex (Σxᵢ = 1) with adjustable
    parameters `a`, `b`.  Global minimiser is x* = (1/D,…,1/D).
    """
    c      = 2 * np.pi
    d      = x.shape[1]
    centre = np.full(d, 1.0 / d, dtype=x.dtype)
    y      = x - centre

    sum_sq  = np.sum(y ** 2, axis=1)
    sum_cos = np.sum(np.cos(c * y), axis=1)

    fx = a + np.e               \
       - a * np.exp(-b * np.sqrt(sum_sq / d)) \
       -     np.exp(      sum_cos / d)
    return fx.reshape(-1, 1)


def multiwell_ackley10_simplex(x: np.ndarray,
                               a            = 20.0,
                               b            = 0.2,
                               extra_wells  = 2,
                               depths       = 5.0,
                               width        = 0.15,
                               seed         = 42) -> np.ndarray:
    """
    Ackley-on-simplex (with the given a,b) + `extra_wells` Gaussian wells.

    • `depths` can be a scalar or a 1-D array of length `extra_wells`.
    • `width` is the shared L2 σ of the wells.
    """
    base = ackley10_simplex(x, a=a, b=b).ravel()

    rng      = np.random.default_rng(seed)
    centres  = rng.dirichlet(np.ones(x.shape[1]), size=extra_wells)

    depths   = np.broadcast_to(np.asarray(depths, dtype=x.dtype), extra_wells)
    bonus    = np.zeros_like(base)
    for cen, dep in zip(centres, depths):
        dist2 = np.sum((x - cen) ** 2, axis=1)
        bonus -= dep * np.exp(-dist2 / (2 * width ** 2))

    return (base + bonus).reshape(-1, 1)



def dirichlet_init(obj_fun, D: int, n_pts: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.ones(D), size=n_pts)
    X = np.round(X, 3)
    X[:, -1] = 1.0 - X[:, :-1].sum(axis=1)  # enforce simplex exactly
    Y = obj_fun(X)
    return pd.DataFrame(X), pd.DataFrame(Y)


def run_one(obj_fun, label, args):
    D = 10
    X_init, Y_init = dirichlet_init(obj_fun, D, args.init_pts, args.seed)

    zombi = ZombiHop(
        seed              = args.seed,
        X_init            = X_init,
        Y_init            = Y_init,
        Y_experimental    = obj_fun,
        Gammas            = args.gammas,
        alphas            = args.alphas,
        n_draws_per_activation = args.draws,
        acquisition_type  = LCB_ada,
        tolerance         = 0.15,
        penalty_width     = 0.1,
        m                 = 5,
        k                 = 5,
        lower_bound       = np.zeros(D),
        upper_bound       = np.ones(D),
        resolution        = args.res,
        sampler           = None,
    )

    t0 = time.perf_counter()
    X_all, Y_all, *_ = zombi.run_experimental(
        n_droplets = args.draws,
        n_vectors  = args.n_vectors,
        verbose    = False,
        plot       = False,
    )
    best_hist = np.minimum.accumulate(Y_all.values.ravel())
    time_hist = np.linspace(0, time.perf_counter() - t0, len(best_hist))

    outdir = Path("./results")
    outdir.mkdir(exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.plot(best_hist, lw=1.5)
    plt.xlabel("experiments");  plt.ylabel("best f(x)")
    plt.title(f"Convergence – {label}")
    plt.grid(True)
    plt.savefig(outdir / f"convergence_{label}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(time_hist, best_hist, lw=1.5)
    plt.xlabel("wall-clock [s]");  plt.ylabel("best f(x)")
    plt.title(f"Time vs Experiments – {label}")
    plt.grid(True)
    plt.savefig(outdir / f"time_{label}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓  saved plots for {label}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wells", type=int, default=0, help="number of extra wells in custom Ackley")
    p.add_argument("--init-pts", type=int, default=50)
    p.add_argument("--gammas", type=int, default=10)
    p.add_argument("--alphas", type=int, default=10)
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--res", type=int, default=3)
    p.add_argument("--n-vectors", type=int, default=11)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    for a_val in [5,10,20,30,50]:
        for b_val in [0.1,0.2,0.5,1]:
            # multiwell benchmark
            if args.wells > 0:
                run_one(lambda x, a=a_val, b=b_val:
                            multiwell_ackley10_simplex(
                                x, a=a, b=b,
                                extra_wells=args.wells,
                                depths=np.linspace(5, 5*args.wells, num=args.wells)),
                        label=f"ackleyMW{args.wells}_a{a_val}_b{b_val}",
                        args=args)
                
            # single well benchmark
            else: 
                run_one(lambda x, a=a_val, b=b_val:
                            ackley10_simplex(x, a=a, b=b),
                        label=f"ackley_a{a_val}_b{b_val}",
                        args=args)



if __name__ == "__main__":
    main()
