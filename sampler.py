#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:04:44 2023

@author: atiihone
"""

import joblib
import numpy as np
import os
import sys
import sqlite3
import time

from acquisitions import *
import communication

#sys.path.append('./../Line-BO/HPER')

from linebo_fun_aleks import extract_inlet_outlet_points, compute_x_coords_along_lines, calc_K, integrate_all_K_over_acqf, choose_K

#poisson_model = joblib.load(os.getcwd()+'/../data/poisson_RF_trained.pkl')

# def sample_y(x, model):
#     """
#     Retrieves sample y values from points x using one of the target functions
#     defined in dictionary target_funs.
    
#     Parameters
#     ----------
#     x : Numpy array of shape (n_samples, n_dimensions)
#         Input x datapoints to be sampled.
#     target_fun_idx : Integer
#         Index of the desired target function in target_funs.
#     target_funs : Dictionary {integer: string}
#         Dictionary of target function options. # To do: Should be a global variable?

#     Raises
#     ------
#     Exception
#         Exception raised when the sampling for the requested target function
#         has not been implemented.

#     Returns
#     -------
#     y : Numpy array of shape (n_samples, 1)
#         The sampled function values.

#     """

#     y = model(x)
    
#     return y


# def get_y_measurements(x):
#     """
#     Blocks until the SQLite file './sql/objective.db' contains
#     one row of length x.shape[0], then returns it as a 1D numpy array,
#     and clears the table to signal “ready for next iteration.”
#     Also resets the compositions.db file.
#     """
#     db_path = './sql/objective.db'

#     while True:
#         if not os.path.exists(db_path):
#             time.sleep(10)
#             continue

#         conn = sqlite3.connect(db_path)
#         cur = conn.cursor()

#         cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
#         tbl = cur.fetchone()

#         if tbl:
#             table_name = tbl[0]
#             cur.execute(f"SELECT * FROM {table_name};")
#             rows = cur.fetchall()
#             if len(rows) == 1:
#                 row = rows[0]
#                 if len(row) != x.shape[0]:
#                     raise ValueError(f"Expected {x.shape[0]} columns, got {len(row)}")

#                 y = np.array(row, dtype=float)
#                 conn.close()

#                 # Reset both databases
#                 communication.reset_objective("./sql/objective.db")
#                 communication.reset_compositions("./sql/compositions.db")

#                 return y

#         conn.close()
#         time.sleep(10)




# ─── get_y_measurements now returns (y, comps) and clears both tables ───────
def get_y_measurements(x):
    """
    Blocks until './sql/objective.db' has one row of length x.shape[0],
    then returns (y_array, compositions_array) and clears both tables.
    """
    db = './sql/objective.db'
    while True:
        if not os.path.exists(db):
            time.sleep(1); continue

        conn = sqlite3.connect(db)
        cur  = conn.cursor()

        # 1) objective row
        cur.execute("SELECT * FROM objective LIMIT 1")
        row = cur.fetchone()
        if not row:
            conn.close()
            time.sleep(1)
            continue
        if len(row) != x.shape[0]:
            conn.close()
            raise ValueError(f"Expected {x.shape[0]} cols, got {len(row)}")

        y = np.array(row, dtype=float)

        # 2) compositions matrix
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='compositions'"
        )
        if cur.fetchone():
            cur.execute("SELECT * FROM compositions")
            comp_rows = cur.fetchall()
            comps = np.array(comp_rows, dtype=float) if comp_rows else None
        else:
            comps = None

        # 3) clear both tables so next packet is fresh
        cur.execute("DELETE FROM objective")
        cur.execute("DELETE FROM compositions")
        conn.commit()
        conn.close()

        # 4) reset the external compositions.db for the sender side
        communication.reset_compositions("./sql/compositions.db")

        return y, comps




'''
def acq_from_zombihop_GP(acq_object, x, acq_params):
    
    self.acquisition_type(X=dimension_meshes, GP_model=GP,  n=n, fX_best=Y_BOUNDmemory.min(), ratio=self.ratio, decay=self.decay, xi=self.xi, ftype=self.ftype)
'''
'''
def old_choose_K_acqarray(acquisitions, p, K_cand, x_acquisitions, emax = 1, 
                      emin = 0, M = 2, acq_max = True, 
                      selection_method = 'integrate_acq', emax_global = None,
                      emin_global = None):
    """
    Note that the selection method 'integrate_acq' is straightforward
    integration here, so it results in the preference toward longer lines
    rather than short ones (even if their acquisition function values would be 
    the same). This is good for regularization, as short lines provide less
    information.

    Parameters
    ----------
    BO_bject : Numpy array (X**N)
        Numpy array containing the acquisition function values over the whole
        search space as a Numpy array.
    p : TYPE
        DESCRIPTION.
    K_cand : TYPE
        DESCRIPTION.
    emax : TYPE, optional
        DESCRIPTION. The default is 1.
    emin : TYPE, optional
        DESCRIPTION. The default is 0.
    M : TYPE, optional
        DESCRIPTION. The default is 2.
    acq_max : TYPE, optional
        DESCRIPTION. The default is True.
    selection_method : TYPE, optional
        DESCRIPTION. The default is 'integrate_acq'.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    A_sel : TYPE
        DESCRIPTION.
    B_sel : TYPE
        DESCRIPTION.
    tA_sel : TYPE
        DESCRIPTION.
    tB_sel : TYPE
        DESCRIPTION.
    K_sel : TYPE
        DESCRIPTION.

    """
    
    A, B, tA, tB = extract_inlet_outlet_points(p, K_cand = K_cand, emax = emax,
                                       emin = emin, M = M)
    
    if selection_method == 'integrate_acq':
        
        acq_object = {'x': x_acquisitions, 'y': acquisitions}
        
        I_all = np.empty((K_cand.shape[0],1)) 
        
        for i in range(K_cand.shape[0]):
            
            I_all[i] = integrate_over_acqf(p, K_cand[[i],:], tA[i,:], tB[i,:], 
                                           100, acq_object, acq_max = acq_max)
            
        idx = np.argmax(I_all, axis = 0)
        
    elif selection_method == 'random':
        
        idx = np.random.randint(0, A.shape[0])
        
    else:
        
        raise Exception("Not implemented.")
        
    if emax_global is None:
        
        A_sel = A[idx, :]
        B_sel = B[idx, :]
        tA_sel = tA[idx]
        tB_sel = tB[idx]
        K_sel = K_cand[idx, :]
        
    else:
        
        # TO DO:
        # Change K_cand = K_cand[idx, :] below (did not work because the ) 
            
        # Extend the chosen line to the global search space.
        A, B, tA, tB = extract_inlet_outlet_points(p, K_cand = K_cand[idx,:], 
                                                   emax = emax_global,
                                                   emin = emin_global, M = M)
        
        A_sel = A[[0], :]
        B_sel = B[[0], :]
        tA_sel = tA[[0]]
        tB_sel = tB[[0]]
        K_sel = K_cand[idx, :]
    
    return A_sel, B_sel, tA_sel, tB_sel, K_sel
'''
def acq_fun_zombihop(x, acq_params):
    
    acquisition_type = acq_params['acq_type'] 
    acq_GP = acq_params['acq_GP']
    acq_n = acq_params['acq_n'] 
    acq_fX_best = acq_params['acq_fX_best']
    acq_ratio = acq_params['acq_ratio'] 
    acq_decay = acq_params['acq_decay'] 
    acq_xi = acq_params['acq_xi']
    acq_ftype = acq_params['acq_ftype']
    
    y = acquisition_type(X=x, GP_model=acq_GP,  n=acq_n, fX_best=acq_fX_best,
                         ratio=acq_ratio, decay=acq_decay, xi=acq_xi, 
                         ftype=acq_ftype)
    
    return y
'''
def old_choose_K_acq_zombihop(acq_params, p, K_cand, 
                          emax = 1, emin = 0, emax_global = None,
                          emin_global = None, M = 2, acq_max = True, 
                          selection_method = 'integrate_acq'):
    """
    Note that the selection method 'integrate_acq' is straightforward
    integration here, so it results in the preference toward longer lines
    rather than short ones (even if their acquisition function values would be 
    the same). This is good for regularization, as short lines provide less
    information.

    Parameters
    ----------
    BO_bject : Numpy array (X**N)
        Numpy array containing the acquisition function values over the whole
        search space as a Numpy array.
    p : TYPE
        DESCRIPTION.
    K_cand : TYPE
        DESCRIPTION.
    emax : TYPE, optional
        DESCRIPTION. The default is 1.
    emin : TYPE, optional
        DESCRIPTION. The default is 0.
    M : TYPE, optional
        DESCRIPTION. The default is 2.
    acq_max : TYPE, optional
        DESCRIPTION. The default is True.
    selection_method : TYPE, optional
        DESCRIPTION. The default is 'integrate_acq'.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    A_sel : TYPE
        DESCRIPTION.
    B_sel : TYPE
        DESCRIPTION.
    tA_sel : TYPE
        DESCRIPTION.
    tB_sel : TYPE
        DESCRIPTION.
    K_sel : TYPE
        DESCRIPTION.

    """
    
    A, B, tA, tB = extract_inlet_outlet_points(p, K_cand = K_cand, emax = emax,
                                       emin = emin, M = M)
    
    if selection_method == 'integrate_acq':
        
        I_all = np.empty((K_cand.shape[0],1)) 
        
        for i in range(K_cand.shape[0]):
            
            I_all[i] = integrate_over_acqf(p, K_cand[[i],:], tA[i,:], tB[i,:], 
                                           100, acq_object = acq_fun_zombihop, 
                                           acq_max = acq_max, 
                                           acq_params = acq_params)
            
        idx = np.argmax(I_all, axis = 0)
        
    elif selection_method == 'random':
        
        idx = np.random.randint(0, A.shape[0])
        
    else:
        
        raise Exception("Not implemented.")
        
    if emax_global is None:
        
        A_sel = A[idx, :]
        B_sel = B[idx, :]
        tA_sel = tA[idx]
        tB_sel = tB[idx]
        K_sel = K_cand[idx, :]
        
    else:
        
        # TO DO:
        # Change K_cand = K_cand[idx, :] below (did not work because the ) 
            
        # Extend the chosen line to the global search space.
        A, B, tA, tB = extract_inlet_outlet_points(p, K_cand = K_cand[idx,:], 
                                                   emax = emax_global,
                                                   emin = emin_global, M = M)
        
        A_sel = A[[0], :]
        B_sel = B[[0], :]
        tA_sel = tA[[0]]
        tB_sel = tB[[0]]
        K_sel = K_cand[idx, :]
    
    return A_sel, B_sel, tA_sel, tB_sel, K_sel
'''
# def choose_K_acq_zombihop(acq_params,
#                           p,
#                           K_cand,
#                           emax=1,
#                           emin=0,
#                           emax_global=None,
#                           emin_global=None,
#                           M=2,
#                           acq_max=True,
#                           selection_method='integrate_acq_line',
#                           constrain_sum_x=False,
#                           plotting=False,
#                           max_candidates=5000):
#     """
#     Subsample huge K_cand down to at most max_candidates, then call our
#     vectorized choose_K() on the subset.  Map the chosen direction back
#     into the full K_cand if you later need to extend to global bounds.
#     """

#     if constrain_sum_x:
#         raise ValueError("constrain_sum_x not supported in ZoMBI-Hop wrapper")

#     # Decide which acquisition-function wrapper to forward:
#     BO_object_zombihop = None if acq_params['acq_type'] is None else acq_fun_zombihop

#     # --- 1) SUBSAMPLE candidate directions if there are too many ---
#     K_total = K_cand.shape[0]
#     if K_total > max_candidates:
#         sample_idx = np.random.choice(K_total, max_candidates, replace=False)
#         K_sub = K_cand[sample_idx]
#     else:
#         sample_idx = np.arange(K_total)
#         K_sub = K_cand

#     # --- 2) Call the fast, vectorized core choose_K on the (smaller) subset ---
#     A_sub, B_sub, tA_sub, tB_sub, K_subsel = choose_K(
#         BO_object=BO_object_zombihop,
#         p=p,
#         K_cand=K_sub,
#         emax=emax,
#         emin=emin,
#         M=M,
#         acq_max=acq_max,
#         selection_method=selection_method,
#         constrain_sum_x=constrain_sum_x,
#         plotting=plotting,
#         acq_params=acq_params
#     )

#     # --- 3) Figure out which index in the subset was chosen ---
#     # (there might be duplicates in K_sub, so we pick the first match)
#     local_idx = np.where((K_sub == K_subsel).all(axis=1))[0][0]
#     global_idx = sample_idx[local_idx]

#     # These are your local selections:
#     A_sel  = A_sub
#     B_sel  = B_sub
#     tA_sel = tA_sub
#     tB_sel = tB_sub
#     K_sel  = K_subsel

#     # --- 4) If you have a global boundary, extend that same direction globally ---
#     if emax_global is not None:
#         # pick out the single global K row by index
#         K_one = K_cand[[global_idx], :]
#         A_sel, B_sel, tA_sel, tB_sel, K_sel = extract_inlet_outlet_points(
#             p,
#             K_cand=K_one,
#             emax=emax_global,
#             emin=emin_global,
#             M=M,
#             constrain_sum_x=constrain_sum_x,
#             plotting=plotting
#         )

#     return A_sel, B_sel, tA_sel, tB_sel, K_sel


# Modify to output the top two performing A_sel and B_sel from the space angle integration process
def choose_K_acq_zombihop(acq_params,
                          p,
                          K_cand,
                          emax=1,
                          emin=0,
                          emax_global=None,
                          emin_global=None,
                          M=2,
                          acq_max=True,
                          selection_method='integrate_acq_line',
                          constrain_sum_x=False,
                          plotting=False,
                          max_candidates=5000):
    """
    Returns:
      A_sel,  B_sel,  tA_sel,  tB_sel,  K_sel,         # best
      A_cache,B_cache,tA_cache,tB_cache,K_cache        # second-best
    """
    if constrain_sum_x:
        raise ValueError("constrain_sum_x not supported here")

    # Which acquisition‐wrapper to call (or None for array‐based acquisitions)
    acq_fun = acq_fun_zombihop if acq_params['acq_type'] else None
    
    '''
    # Subsampling not needed here anymore because K_cands should be defined
    # so that the matrix is not huge.
    # 1) SUBSAMPLE if huge
    K_total = K_cand.shape[0]
    if K_total > max_candidates:
        idxs = np.random.choice(K_total, max_candidates, replace=False)
    else:
        idxs = np.arange(K_total)
    K_sub = K_cand[idxs]
    '''
    K_sub = K_cand
    
    # 2) COMPUTE all inlet/outlet points for subset
    A_sub, B_sub, tA_sub, tB_sub = extract_inlet_outlet_points(
        p, K_cand=K_sub, emax=emax, emin=emin, M=M,
        constrain_sum_x=True, plotting=plotting
    )

    # 3) INTEGRATE acquisition on each line
    # I_all = np.empty(K_sub.shape[0], dtype=float)
    # for i in range(K_sub.shape[0]):
    #     I_all[i] = integrate_over_acqf(
    #         p,
    #         K_sub[[i], :],
    #         tA_sub[i],
    #         tB_sub[i],
    #         100,                  # you can adjust #points here
    #         acq_fun,
    #         acq_max,
    #         acq_params
    #     )

    I_all = integrate_all_K_over_acqf(
    p,
    K_sub,
    tA_sub,
    tB_sub,
    n_points=100,     # same as your old “100” here
    acq_fun=acq_fun,
    acq_max=acq_max,
    acq_params=acq_params
)

    # 4) PICK top-2
    order = np.argsort(I_all)[::-1]
    i0, i1 = order[0], order[1]

    # best
    A_sel,  B_sel  = A_sub[i0],  B_sub[i0]
    tA_sel, tB_sel = tA_sub[i0], tB_sub[i0]
    K_sel         = K_sub[i0]

    # second best (cache)
    A_cache,  B_cache  = A_sub[i1],  B_sub[i1]
    tA_cache, tB_cache = tA_sub[i1], tB_sub[i1]
    K_cache           = K_sub[i1]

    # 5) EXTEND into global bounds if needed
    if emax_global is not None:
        # best
        A_sel, B_sel, tA_sel, tB_sel = extract_inlet_outlet_points(
            p, K_cand=K_cand[[idxs[i0]]], emax=emax_global, emin=emin_global,
            M=M, constrain_sum_x=False, plotting=plotting
        )
        K_sel = K_cand[idxs[i0]]

        # cache
        A_cache, B_cache, tA_cache, tB_cache = extract_inlet_outlet_points(
            p, K_cand=K_cand[[idxs[i1]]], emax=emax_global, emin=emin_global,
            M=M, constrain_sum_x=False, plotting=plotting
        )
        K_cache = K_cand[idxs[i1]]

    return (A_sel,  B_sel,  tA_sel,  tB_sel,  K_sel,
            A_cache,B_cache,tA_cache,tB_cache,K_cache)


def search_space_to_nparr(emin, emax, N):
    
    # Ensure search space boundaries are provided as a 2D numpy array. 
    
    if isinstance(emin, (float, int)) is True:
        
        emin_new = np.tile([[emin]], N)
        
    elif len(emin.shape) < 2:
        
        # Works for numpy arrays and pandas dataframes
        emin_new = np.reshape(np.array(emin), [-1, N])
    
    else:
        
        raise Exception("Not implemented.")
        
    if isinstance(emax, (float, int)) is True:
        
        emax_new = np.tile([[emax]], N)
        
    elif len(emax.shape) < 2:
        
        # Works for numpy arrays and pandas dataframes
        emax_new = np.reshape(np.array(emax), [-1, N])
    
    else:
        
        raise Exception("Not implemented.")
        
    return emin_new, emax_new

# def line_bo_sampler(X_ask, x_acquisitions, acquisitions, model, 
#                     emin = 0, emax = 1, emin_global = 0, emax_global = 1,
#                     n_droplets = 30, M = 21, acq_max = False, 
#                     selection_method = 'integrate_acq_line', acq_GP = None, 
#                     acq_type = None, acq_n = None, acq_fX_best = None, 
#                     acq_ratio = None, acq_decay = None, acq_xi = None, 
#                     acq_ftype = None, plotting = False):
#     """
#     This is the main Line-BO sampler to be used with ZoMBI-Hop. You should not
#     need to modify any other functions than this to switch between the
#     features Armi has implemented by 6/19/2024.

#     Parameters
#     ----------
#     X_ask : TYPE
#         Point P suggested by ZoMBI-Hop.
#     x_acquisitions : TYPE
#         Locations of the points included into the ZoMBI-Hop acquisitions matrix.
#     acquisitions : TYPE
#         ZoMBI-Hop acquisitions matrix.
#     model : TYPE
#         Callable for sampling the ground truth ("performing the measurement").
#     emin : TYPE, optional
#         Lower boundary of the (local, zoomed-in) search space. The default is 0.
#     emax : TYPE, optional
#         Lower boundary of the (local, zoomed-in) search space. The default is 1.
#     emin_global : TYPE, optional
#         Lower boundary of the global search space. The default is 0.
#     emax_global : TYPE, optional
#         Upper boundary of the global search space.  The default is 1.
#     n_droplets : TYPE, optional
#         Number of droplets "to be synthesized" when sampling from 'model'.
#         The default is 3.
#     M : TYPE, optional
#         Number of space angles to be considered per dimension. The default is 21.
#     acq_max : TYPE, optional
#         Set to True if the most promising value of the acquisition function is
#         its maximum. Set to False if it is its minimum. The default is False.
#     selection_method : TYPE, optional
#         Method of selecting the best line. Options are 'integrate_acq_line' and
#         'random_line'. The default is 'integrate_acq_line'.
#     acq_GP : TYPE, optional
#         The GP surrogate model that will be forwarded to the acquisition function.
#         Will be used only if the acquisitions are defined as a callable function
#         (see argument 'acq_type'). See the description from ZoMBI-Hop 
#         documentation. The default is None.
#     acq_type : TYPE, optional
#         The type of the acquisitions object. Can be either a callable ZoMBI-Hop
#         -compatible acquisition function, or None, in which case the ZoMBI-Hop
#         acquisitions matrix defined by 'acquisitions' and 'x_acquisitions' will
#         be used in Line-BO. The default is None.
#     acq_n : TYPE, optional
#         Acquisition argument 'n' that will be forwarded to the acquisition function.
#         Will be used only if the acquisitions are defined as a callable function.
#         See the description from ZoMBI-Hop documentation.  The default is None.
#     acq_fX_best : TYPE, optional
#         Acquisition argument 'fX_best' that will be forwarded to the acquisition function.
#         Will be used only if the acquisitions are defined as a callable function.
#         See the description from ZoMBI-Hop documentation.  The default is None.
#     acq_ratio : TYPE, optional
#         Acquisition argument 'ratio' that will be forwarded to the acquisition function.
#         Will be used only if the acquisitions are defined as a callable function.
#         See the description from ZoMBI-Hop documentation.  The default is None.
#     acq_decay : TYPE, optional
#         Acquisition argument 'decay' that will be forwarded to the acquisition function.
#         Will be used only if the acquisitions are defined as a callable function.
#         See the description from ZoMBI-Hop documentation.  The default is None.
#     acq_xi : TYPE, optional
#         Acquisition argument 'xi' that will be forwarded to the acquisition function.
#         Will be used only if the acquisitions are defined as a callable function.
#         See the description from ZoMBI-Hop documentation.  The default is None.
#     acq_ftype : TYPE, optional
#         Acquisition argument 'ftype' that will be forwarded to the acquisition function.
#         Will be used only if the acquisitions are defined as a callable function.
#         See the description from ZoMBI-Hop documentation.  The default is None.
#     plotting : TYPE, optional
#         The level of producing plots during the ZoMBI-Hop run. Options are:
#         'plot_all', 'plot_few', 'plot_none'. The default is 'plot_all'.

#     Returns
#     -------
#     X_tell : TYPE
#         DESCRIPTION.
#     Y_tell : TYPE
#         DESCRIPTION.

#     """
        
#     # Ensure search space boundaries are provided as a 2D numpy array. 

#     print('search space to np array')
#     emin, emax = search_space_to_nparr(emin, emax, X_ask.shape[1])
    
#     # NOTE
#     # This is where emin and emax could be increased by a marging of numerical
#     # error to make sure the P points are always strictly inside the search
#     # space boundaries.
    
    
#     # Ensure search space boundaries are provided as a 2D numpy array.
#     # (The global boundaries are disregarded in the rest of the code if None.)
#     if emin_global is not None:
        
#         emin_global, emax_global = search_space_to_nparr(emin_global, 
#                                                          emax_global, 
#                                                          X_ask.shape[1])
    
#     idx_to_compute = list(np.arange(X_ask.shape[0]))
#     #idx_to_compute = np.reshape(idx_to_compute, (-1,1))
#     N = X_ask.shape[1]
    
#     # Initialize K candidate points that define the angle of the line together
#     # with BO-suggested point P.
#     print('calc K')
#     K_cand = calc_K(N, M, plotting = False) # TO DO: We could change sampler to an object and calculate K only once.
        
#     # The inlet and outlet points of the line for the next round.
    
#     if acq_type is None:
        
#         # No acquisition function provided; use acquisition array instead.
#         # This option is slower.
        
#         acq_params = {'acq_type': None,
#                       'acq_dictionary': {'x': x_acquisitions, 
#                                          'y': acquisitions}}
        
#     else:
        
#         # Use acquisition function.
#         acq_params = {'acq_type': acq_type, 'acq_GP': acq_GP,
#                       'acq_n': acq_n, 
#                       'acq_fX_best': acq_fX_best,
#                       'acq_ratio': acq_ratio, 'acq_decay': acq_decay, 
#                       'acq_xi': acq_xi, 'acq_ftype': acq_ftype}
        
#     print('choose K acq')
#     A_sel, B_sel, tA_sel, tB_sel, K_sel = choose_K_acq_zombihop(
#         acq_params = acq_params, p = X_ask, K_cand = K_cand, 
#         emax = emax, emin = emin, 
#         emax_global = emax_global, emin_global = emin_global,
#         M = M, acq_max = acq_max, selection_method = selection_method,
#         plotting = plotting)
    
#     # Compute the equally spaced x values along the selected line (or lines if 
#     # there are multiple init points).
#     print('compute x coords')
#     x = compute_x_coords_along_lines(idx_to_compute, N, n_droplets, 
#                                    tA_sel, tB_sel, X_ask, K_sel)

#     print('start: ', A_sel)
#     print('end: ', B_sel)

#     print('droplets: ', x)

#     # "Do the measurement"
#     # y = sample_y(x, model = model)

#     communication.write_compositions(start=np.round(A_sel,3), end=np.round(B_sel,3), array=np.round(x,3), timestamp=time.time())

#     print('READY FOR EXPERIMENTS')
#     y, x = get_y_measurements(x) # update compositions using those that were actually measured, with their objective values
#     print('y shape ', y.shape)
    
#     X_tell = x
#     Y_tell = np.ravel(y)
    
#     # print("ZoMBI-Hop suggested: \n - Point P=" + str(X_ask) + "\n- emin=" + 
#     #       str(emin) + "\n- emax=" + str(emax) + "\mLine-BO-suggested points:\n-A=" + 
#     #       str(A_sel) + "\n-B=" + str(B_sel))
    
#     return X_tell, Y_tell


# Modify to output the best and second best composition predictions
# Store the second best prediction in the cache to use in DiSCO to prevent backlog
def line_bo_sampler(X_ask, x_acquisitions, acquisitions, model, 
                    emin=0, emax=1, emin_global=0, emax_global=1,
                    n_droplets=30, M=2, acq_max=False, 
                    selection_method='integrate_acq_line', acq_GP=None, 
                    acq_type=None, acq_n=None, acq_fX_best=None, 
                    acq_ratio=None, acq_decay=None, acq_xi=None, 
                    acq_ftype=None, plotting=False, max_candidates=5000):
    """
    Main Line-BO sampler, now returning both best and second-best lines.
    """
    # 1) prepare bounds
    emin, emax = search_space_to_nparr(emin, emax, X_ask.shape[1])
    if emin_global is not None:
        emin_global, emax_global = search_space_to_nparr(
            emin_global, emax_global, X_ask.shape[1]
        )

    N = X_ask.shape[1]
    K_cand = calc_K(N, M, constrain_sum_x=True, plotting=plotting, 
                    generate_randomly=True, max_candidates=max_candidates, 
                    random_generation_type='cartesian',p=X_ask)

    # 2) build acq_params dict
    if acq_type is None:
        ap = {'acq_type': None,
              'acq_dictionary': {'x': x_acquisitions, 'y': acquisitions}}
    else:
        ap = {'acq_type': acq_type, 'acq_GP': acq_GP,
              'acq_n': acq_n, 'acq_fX_best': acq_fX_best,
              'acq_ratio': acq_ratio, 'acq_decay': acq_decay,
              'acq_xi': acq_xi, 'acq_ftype': acq_ftype}

    # 3) pick both best & second-best lines
    (A_sel, B_sel, tA_sel, tB_sel, K_sel,
     A_cache, B_cache, tA_cache, tB_cache, K_cache) = choose_K_acq_zombihop(
        acq_params=ap, p=X_ask, K_cand=K_cand,
        emax=emax, emin=emin, emax_global=emax_global,
        emin_global=emin_global, M=M, acq_max=acq_max,
        selection_method=selection_method, plotting=plotting,
        max_candidates=max_candidates
    )

    # 4) compute droplets for both lines
    idxs = [0]  # single BO-suggested point
    x      = compute_x_coords_along_lines(
        idxs, N, n_droplets, tA_sel, tB_sel, X_ask, K_sel
    )
    x_cache = compute_x_coords_along_lines(
        idxs, N, n_droplets, tA_cache, tB_cache, X_ask, K_cache
    )

    # 5) logging
    print('best start: ', A_sel)
    print('best end:   ', B_sel)
    # print('best drops: ', x)
    print('cache start:', A_cache)
    print('cache end:  ', B_cache)
    # print('cache drops:', x_cache)

    # 6) hand off to experiment
    communication.write_compositions(
        start=np.round(A_sel,3),
        end=np.round(B_sel,3),
        array=np.round(x,3),
        start_cache=np.round(A_cache,3),
        end_cache=np.round(B_cache,3),
        array_cache=np.round(x_cache,3),
        timestamp=time.time()
    )
    y, x_meas = get_y_measurements(x)

    X_tell = x_meas
    Y_tell = y.ravel()

    return X_tell, Y_tell