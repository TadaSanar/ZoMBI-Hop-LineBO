import numpy as np
from utils import project_simplex, GP_pred, reset_GP
from zombihop import ZombiHop
from acquisitions import LCB_ada
from benchmark_10d_linbo import ackley10_simplex

D = 10
edge_1   = np.array([[1.0] + [0.0]*(D-1)])
edge_2   = np.array([[0.0, 1.0] + [0.0]*(D-2)])
mid_face = np.array([[0.5, 0.5] + [0.0]*(D-2)])

X_test = np.vstack([edge_1, edge_2, mid_face])
X_proj = np.vstack([project_simplex(x) for x in X_test])
print("Row-sums after projection:", X_proj.sum(axis=1))

# Build a tiny GP on random data then predict the three points
rng = np.random.default_rng(0)
X_init = rng.dirichlet(np.ones(D), size=20)
Y_init = ackley10_simplex(X_init).astype(np.float32)
gp = reset_GP(X_init, Y_init)
mu, std = GP_pred(X_proj, gp, dtype=np.float32)
print("GP μ :", mu)
print("GP σ :", std.ravel())