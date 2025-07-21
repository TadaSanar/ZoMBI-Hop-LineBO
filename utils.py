import pandas as pd
import numpy as np
import sys
import time
import GPy
import itertools
import matplotlib.pyplot as plt
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model    import BayesianRidge


def memory_prune(k, tolerance, X_updated, Y_updated):
    # Prune memory after each set of needle-finding zoom ins. Keep only the top k-number of points
    Y_new = np.array(Y_updated.copy()).reshape(-1) # only grab new values generated this iteration
    X_new = np.array(X_updated.copy())
#     Y_new = np.array(Y_updated[prior_n:].copy()).reshape(-1) # only grab new values generated this iteration
#     X_new = np.array(X_updated[prior_n:].copy())
    top_k = []
    top_kX = []
    while len(top_k) < k:
        ymin_index = np.argmin(Y_new)
        top_k.append(Y_new[ymin_index]) # append current best
        top_kX.append(X_new[ymin_index])
        Y_new = np.delete(Y_new, ymin_index) # delete appended best from Y_new
        X_new = np.delete(X_new, ymin_index, axis = 0) # delete from X_new as well
        similar_values = np.where((np.abs(Y_new - top_k[-1])/np.abs(top_k[-1])) < tolerance)
        Y_new = np.delete(Y_new, similar_values) # delete values similar to best from Y_new
        X_new = np.delete(X_new, similar_values, axis = 0) # delete from X_new as well
        # then repeat
        if len(Y_new) < 1:
            break # break if not enough unique values within tolerance
    df_kX = pd.DataFrame(top_kX, columns = X_updated.columns.values)[::-1] # convert to df
    df_k = pd.DataFrame(top_k, columns = Y_updated.columns.values)[::-1] # convert to df
    return df_kX, df_k


def create_penalty_mask(needle_locs, dimension_meshes, ftype, penalty_width):
    """
    needle_locs: (n_needles, D) array or DataFrame
    dimension_meshes: (N_mesh, D) array
    penalty_width: scalar
    """
    # Determine D from the mesh (or needle_locs)
    D = dimension_meshes.shape[1]

    # Build a D-vector of tolerances
    tol = np.abs(penalty_width * np.ones(D, dtype=ftype))

    # Convert needle_locs into a (n_needles, D) array
    Xn = np.array(needle_locs, dtype=ftype)  # shape (n_needles, D)

    # Upper & lower bounding planes for every needle
    X_upper = (Xn + tol).T  # shape (D, n_needles)
    X_lower = (Xn - tol).T

    # Now for every mesh point we check: is it outside the tol-tube?
    # dimension_meshes is (N_mesh, D).  We want a mask of shape (N_mesh,1).
    inside_any = np.any(
        np.all((dimension_meshes[:, :, None] >= X_lower[None, :, :]) &
               (dimension_meshes[:, :, None] <= X_upper[None, :, :]),
               axis=1),
        axis=1
    )
    # penalty_mask is 1 where *no* needle penalizes
    penalty_mask = (~inside_any).astype(ftype).reshape(-1, 1)
    return penalty_mask



# def bounded_mesh(dims, lower_bound, upper_bound, ftype, resolution=10):
#     # Compute GP within bounds
#     dim_array = []
#     for d in range(dims):
#         dim_array.append(np.linspace(lower_bound[d], upper_bound[d], resolution, dtype=ftype))
#     dimension_meshes = np.array(list(itertools.product(*dim_array)))
#     return dimension_meshes


def bounded_mesh(dims, lower_bound, upper_bound, ftype, resolution=10):
    """
    Generate a grid of points strictly inside [lower_bound, upper_bound]^dims
    by dropping the exact endpoints, so no BO ask ever lands on a corner.
    """
    dim_array = []
    for d in range(dims):
        lb, ub = lower_bound[d], upper_bound[d]
        # create resolution+2 linearly spaced pts, then drop first/last
        if resolution >= 2:
            pts = np.linspace(lb, ub, resolution+2, dtype=ftype)[1:-1]
        else:
            # fallback: really just the midpoint
            pts = np.array([(lb+ub)/2], dtype=ftype)
        dim_array.append(pts)

    # cartesian product
    return np.array(list(itertools.product(*dim_array)))


def m_norm(X):
    '''
    multi-dimensional normalization, takes pd dataframe as input.
    '''
    for m in range(X.shape[1]):
        X.iloc[:, m] = (X.iloc[:, m] - np.min(X.iloc[:, m])) / (np.max(X.iloc[:, m]) - np.min(X.iloc[:, m]))
    return X


def bounded_LHS(N, min_vector, max_vector):
    '''
    Runs bounded Latin Hypercube Sampling (LHS) using the iteratively zoomed-in search space bounds and memory points.
    Used to initialize the start of the forward experiments within each ZoMBI activation.
    :param N:               Number of LHS datapoints to output
    :param min_vector:      A (d,) array of lower bounds
    :param max_vector:      A (d,) array of upper bounds
    :return:                An (N,d) array of X datapoints
    '''
    samples = []
    for n in range(N):
        # randomly select points using uniform sampling within limit ranges
        samples.append(np.random.uniform(low=min_vector, high=max_vector).T)
    return np.vstack(samples)


def project_simplex(v):
    """
    Project a point onto the simplex with a constraint that sum equals 1 and all components are non-negative.
    :param v: A 1-dimensional numpy array.
    :return: A 1-dimensional numpy array, the projected point.
    """
    if v.sum() == 1 and np.alltrue(v >= 0):
        # Already on the simplex
        return v
    else:
        # Sort the input vector
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(1, len(u) + 1)
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w


def reset_GP(X, Y):
    """
    Build and return a full (non-sparse) GaussianProcessRegressor,
    projecting inputs onto the simplex first.
    
    Inputs:
      X : (n, d) array or pd.DataFrame
      Y : (n, 1) array-like or pd.DataFrame
    Output:
      A fitted sklearn GaussianProcessRegressor.
    """
    # 1) Project X onto the simplex
    if isinstance(X, pd.DataFrame):
        Xp = np.vstack([project_simplex(row) for row in X.values])
    else:
        arr = np.asarray(X)
        Xp = np.vstack([project_simplex(row) for row in arr])
    
    # 2) Coerce Y into shape (n, 1)
    Yarr = np.asarray(Y).reshape(-1, 1)
    
    # 3) Build and fit the GP
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * \
             Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.0)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=30,
        alpha=1e-3,
        normalize_y=True,
    )
    gp.fit(Xp, Yarr)
    return gp


def GP_pred(X, GP_model, dtype):
    """
    Predict posterior mean and std from the full GP, projecting X onto
    the simplex first. Returns:
      mean: (n,) array of dtype
      std:  (n,1) array of dtype
    """
    # 1) Coerce into numpy array of shape (n, d)
    if isinstance(X, pd.DataFrame):
        arr = X.values
    else:
        arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    
    # 2) Project onto simplex
    Xp = np.vstack([project_simplex(row) for row in arr])
    
    # 3) Predict
    print("starting GP")
    mean, std = GP_model.predict(Xp, return_std=True)
    print("finished GP")
    
    # 4) Cast and reshape
    mean = np.asarray(mean).reshape(-1).astype(dtype)
    std  = np.asarray(std).reshape(-1, 1).astype(dtype)
    return mean, std


def initialize_arrays(X_init, Y_init):
    # initializes all storage arrays for different parts of data logging / algorithm memory
    X_intermediate = pd.DataFrame([],
                                  columns=X_init.columns.values)  # contains only intermediate points during ZoMBI activations => reset after each hop (Gamma).
    Y_intermediate = pd.DataFrame([], columns=Y_init.columns.values)

    X_all = X_init.copy().reset_index(
        drop=True)  # keep track of all data points, including all intermediate that eventually get pruned
    Y_all = Y_init.copy().reset_index(
        drop=True)  # keep track of all data points, including all intermediate that eventually get pruned

    X_GPmemory = X_all.copy()  # this is the memory that the GP has between iterations => contains initial dataset, all intermediate points until the next hop, then reset to keep only init + top-k points
    Y_GPmemory = Y_all.copy()

    X_BOUNDmemory = X_all.copy()  # this is the memory that the zooming boundary selection has between iterations => contains initial dataset, and only intermediate points of current iteration
    Y_BOUNDmemory = Y_all.copy()

    X_final = X_all.copy()  # contains only the final memory-pruned points
    Y_final = Y_all.copy()

    needles = pd.DataFrame([], columns=Y_all.columns.values)  # init df for needles to be stored
    needle_locs = pd.DataFrame([], columns=X_all.columns.values)  # init df for needle locations to be stored

    return X_intermediate, Y_intermediate, X_all, Y_all, X_GPmemory, Y_GPmemory, X_BOUNDmemory, Y_BOUNDmemory, X_final, Y_final, needles, needle_locs


def plot_penalty_dist(penalty_mask, lower_bound, upper_bound):
    plt.hist(penalty_mask)
    plt.yscale('log')
    plt.grid()
    plt.title(
        f'Percentage of Penalized Values in Current Bounds = {np.round(np.sum(~penalty_mask.astype(bool)) / len(penalty_mask) * 100, 4)}%' + '\n' + f'Upper Bound: {np.round(np.array(upper_bound), 3)}' + '\n' + f'Lower Bound: {np.round(np.array(lower_bound), 3)}')
    plt.show()
    plt.close()


def progress_bar(n, T, inc):
    '''
    Progress bar for optimization procedure.
    :param n:           The current experiment number
    :param T:           The total number of experiments to run
    :param inc:         Increment period for updating progress bar, 0.1/1 => 10 updates
    :return:            Live updating progress bar
    '''

    p = round(n / T, 2)
    total = 40
    step = int(p * total)
    block = '\u275A'
    space = '\u205F\u205F'
    if p >= inc:
        sys.stdout.write(
            f"\r{'[' + block * step + space * (total - step) + '] '}{n + 1} / {T} experiments completed . . .{' ' * step}")
        sys.stdout.flush()
        inc += 0.1
    elif (1 - inc) < 0.1 and inc <= 1:
        step = int(total)
        sys.stdout.write(f"\r{'[' + block * step + space * (total - step) + ']'} {T} / {T} Complete!{' ' * step}")
        sys.stdout.flush()
        inc += 0.1
    return inc