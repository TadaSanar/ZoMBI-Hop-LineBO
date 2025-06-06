import numpy as np
import pandas as pd

from zombihop import ZombiHop                # ZoMBI core
from acquisitions import LCB_ada             # acquisition fn
from bo_gpy_dyes import ackley               # surrogate objective

# ─── Utility: generate a Dirichlet initial set ───────────────────────────

def build_initial_data(D: int, n_pts: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.ones(D), size=n_pts)
    X = np.round(X, 3)
    X[:, -1] = 1.0 - X[:, :-1].sum(axis=1)  # enforce simplex exactly
    return pd.DataFrame(X), pd.DataFrame(rng.uniform(1, 4.5, size=n_pts))


# ─── Main entry point ────────────────────────────────────────────────────

def main():
    # Basic config – tweak as needed
    D, DROPLETS, RESOLUTION, SEED = 10, 30, 3, 1

    X_init, Y_init = build_initial_data(D, DROPLETS, SEED)

    zombi = ZombiHop(
        seed=SEED,
        X_init=X_init,
        Y_init=Y_init,
        Y_experimental=ackley,     # ← immediate computational evaluation
        Gammas=10,
        alphas=10,
        n_draws_per_activation=10,
        acquisition_type=LCB_ada,
        tolerance=0.9,
        penalty_width=0.3,
        m=5,
        k=5,
        lower_bound=np.zeros(D),
        upper_bound=np.ones(D),
        resolution=RESOLUTION,
        sampler=None,              # ← no external sampler, use Y_experimental
    )

    X_all, Y_all, locs, needles = zombi.run_experimental(
        n_droplets=DROPLETS,
        n_vectors=11,
        verbose=True,
        plot=True,
    )

    needles.to_csv("needles.csv", index=False)
    Y_all.to_csv("Y_all.csv", index=False)
    print("Finished – saved needles.csv & Y_all.csv")


if __name__ == "__main__":
    main()
