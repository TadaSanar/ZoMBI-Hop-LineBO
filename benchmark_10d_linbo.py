"""
10-D benchmarking harness for the new ZoMBI-Hop + LineBO implementation.

Runs two synthetic objectives on the 10-component simplex:

1. Standard Ackley (one global minimum ≈ 0 at the centre).
2. Custom multi-well Ackley variant with user-defined well depths.

Usage
-----
$ python benchmark_10d_linebo.py            # default settings
$ python benchmark_10d_linebo.py --wells 3  # 3 additional wells
"""

import argparse, time
from pathlib import Path

import numpy as np, pandas as pd, matplotlib.pyplot as plt

from new_zombihop import ZoMBIHop                # ← new algorithm
# ------------------------------------------------------------------

def ackley10_simplex(x: np.ndarray, a=20., b=0.2) -> np.ndarray:
    c, d   = 2*np.pi, x.shape[1]
    centre = np.full(d, 1./d, dtype=x.dtype)
    y      = x - centre
    s2     = np.sum(y**2, axis=1)
    sc     = np.sum(np.cos(c*y), axis=1)
    fx     = a + np.e - a*np.exp(-b*np.sqrt(s2/d)) - np.exp(sc/d)
    return fx.reshape(-1, 1)


def multiwell_ackley10_simplex(x: np.ndarray, a=20., b=0.2,
                               extra_wells=2, depths=5., width=0.2, seed=42):
    base = ackley10_simplex(x, a=a, b=b).ravel()
    rng  = np.random.default_rng(seed)
    centres = rng.dirichlet(0.05*np.ones(x.shape[1]), size=extra_wells)

    depths = np.broadcast_to(np.asarray(depths, dtype=x.dtype), extra_wells)
    bonus  = np.zeros_like(base)
    for cen, dep in zip(centres, depths):
        dist2 = np.sum((x-cen)**2, axis=1)
        bonus -= dep * np.exp(-dist2/(2*width**2))
    return bonus.reshape(-1, 1)


# ------------------------------------------------------------------
def dirichlet_init(obj_fun, D: int, n_pts: int, seed: int):
    rng = np.random.default_rng(seed+1)
    X   = rng.dirichlet(0.05*np.ones(D), size=n_pts)
    Y   = obj_fun(X)
    return X, Y.ravel()


def run_one(obj_fun, label, args):
    D                      = 10
    X_init, Y_init         = dirichlet_init(obj_fun, D, args.init_pts, args.seed)
    n_experiments          = args.draws

    def line_objective(endpoints_batch):
        a, b  = np.asarray(endpoints_batch)
        ts    = np.linspace(0., 1., n_experiments)
        pts   = (1-ts[:,None])*a + ts[:,None]*b
        vals  = obj_fun(pts).ravel()
        return pts, vals

    zombi = ZoMBIHop(
        objective_function   = line_objective,
        dimensions           = D,
        X_init_actual        = X_init,
        Y_init               = Y_init,
        num_activations      = args.gammas,
        max_gp_points        = 30,
        tolerance            = 0.06,
        resolution           = args.res,
        num_experiments      = n_experiments,
        linebo_num_lines     = args.n_vectors,
        bounds               = [(0.,1.)]*D,
    )

    t0 = time.perf_counter()
    res = zombi.run_zombi_hop(verbose=False)
    elapsed = time.perf_counter() - t0

    best_hist = np.minimum.accumulate(zombi.Y_all)
    outdir    = Path("./results1"); outdir.mkdir(exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.plot(best_hist, lw=1.5)
    plt.xlabel("experiments"); plt.ylabel("best f(x)")
    plt.title(f"Convergence – {label}")
    plt.grid(True)
    plt.savefig(outdir / f"convergence_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_idx  = np.argmin(zombi.Y_all)
    best_x    = zombi.X_all_actual[best_idx]
    best_fx   = zombi.Y_all[best_idx]

    row = (
        pd.DataFrame([best_x])
        .assign(best_fx = best_fx, label = label, secs = elapsed)
        .loc[:, ["label", *range(D), "best_fx", "secs"]]
    )

    csv = outdir / "best_points.csv"
    row.to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"✓ saved results for {label}")


# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wells",      type=int, default=0)
    p.add_argument("--init-pts",   type=int, default=30)
    p.add_argument("--gammas",     type=int, default=2)
    p.add_argument("--draws",      type=int, default=24)
    p.add_argument("--res",        type=int, default=3)
    p.add_argument("--n-vectors",  type=int, default=50)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    for a_val in [10, 20, 50]:
        for b_val in [0.1, 0.2, 0.5]:

            if args.wells > 0:   # multi-well variant
                run_one(lambda x, a=a_val, b=b_val:
                            multiwell_ackley10_simplex(
                                x, a=a, b=b,
                                extra_wells=args.wells,
                                depths=np.linspace(5, 5*args.wells, args.wells)),
                        label=f"ackleyMW{args.wells}_a{a_val}_b{b_val}", args=args)

            else:                # single-well (standard) Ackley
                run_one(lambda x, a=a_val, b=b_val:
                            ackley10_simplex(x, a=a, b=b),
                        label=f"ackley_a{a_val}_b{b_val}", args=args)


if __name__ == "__main__":
    main()