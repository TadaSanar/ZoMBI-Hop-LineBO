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

from acquisitions import *

#sys.path.append('./../Line-BO/HPER')

from bo_gpy_dyes import extract_inlet_outlet_points, compute_x_coords_along_lines, calc_K, integrate_over_acqf

#poisson_model = joblib.load(os.getcwd()+'/../data/poisson_RF_trained.pkl')

def sample_y(x, model):
    """
    Retrieves sample y values from points x using one of the target functions
    defined in dictionary target_funs.
    
    Parameters
    ----------
    x : Numpy array of shape (n_samples, n_dimensions)
        Input x datapoints to be sampled.
    target_fun_idx : Integer
        Index of the desired target function in target_funs.
    target_funs : Dictionary {integer: string}
        Dictionary of target function options. # To do: Should be a global variable?

    Raises
    ------
    Exception
        Exception raised when the sampling for the requested target function
        has not been implemented.

    Returns
    -------
    y : Numpy array of shape (n_samples, 1)
        The sampled function values.

    """

    y = model(x)
    
    return y

def acq_from_zombihop_GP(acq_object, x, acq_params):
    
    self.acquisition_type(X=dimension_meshes, GP_model=GP,  n=n, fX_best=Y_BOUNDmemory.min(), ratio=self.ratio, decay=self.decay, xi=self.xi, ftype=self.ftype)

def choose_K_acqarray(acquisitions, p, K_cand, x_acquisitions, emax = 1, 
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

def acq_fun_zombihop(acq_params, x):
    
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

def choose_K_acq_zombihop(acq_params, p, K_cand, 
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

def line_bo_sampler(X_ask, x_acquisitions, acquisitions, model, 
                    emin = 0, emax = 1, emin_global = 0, emax_global = 1,
                    n_droplets = 3, M = 1, acq_max = False, 
                    selection_method = 'integrate_acq', acq_GP = None, 
                    acq_type = None, acq_n = None, acq_fX_best = None, 
                    acq_ratio = None, acq_decay = None, acq_xi = None, 
                    acq_ftype = None):
        
    # Ensure search space boundaries are provided as a 2D numpy array. 
    emin, emax = search_space_to_nparr(emin, emax, X_ask.shape[1])
    
    # Ensure search space boundaries are provided as a 2D numpy array.
    # (The global boundaries are disregarded in the rest of the code if None.)
    if emin_global is not None:
        
        emin_global, emax_global = search_space_to_nparr(emin_global, 
                                                         emax_global, 
                                                         X_ask.shape[1])
    
    idx_to_compute = np.arange(X_ask.shape[0])
    N = X_ask.shape[1]
    
    # Initialize K candidate points that define the angle of the line together
    # with BO-suggested point P.
    K_cand = calc_K(N, M, plotting = False) # TO DO: We could change sampler to an object and calculate K only once.
        
    # The inlet and outlet points of the line for the next round.
    if acq_type is None:
        
        # No acquisition function provided; use acquisition array instead.
        # This option is slower.
        
        # TO DO: Move the definition of the acq_object here and use the new
        # choose_K_acq_zombihop().
        acq_object = {'x': x_acquisitions, 'y': acquisitions}
        
        A_sel, B_sel, tA_sel, tB_sel, K_sel = choose_K_acqarray(
        acquisitions, X_ask, K_cand, x_acquisitions, emax = emax, emin = emin, 
        M = M, acq_max = acq_max, selection_method = selection_method, 
        emax_global = emax_global, emin_global = emin_global)
    
    else:
        
        # Use acquisition function.
        acq_params = {'acq_type': acq_type, 'acq_GP': acq_GP,
                      'acq_n': acq_n, 
                      'acq_fX_best': acq_fX_best,
                      'acq_ratio': acq_ratio, 'acq_decay': acq_decay, 
                      'acq_xi': acq_xi, 'acq_ftype': acq_ftype}
        
        A_sel, B_sel, tA_sel, tB_sel, K_sel = choose_K_acq_zombihop(
        acq_params = acq_params, p = X_ask, K_cand = K_cand, 
        emax = emax, emin = emin, 
        emax_global = emax_global, emin_global = emin_global,
        M = M, acq_max = acq_max, selection_method = selection_method)
    
    # Compute the equally spaced x values along the selected line (or lines if 
    # there are multiple init points).
    x = compute_x_coords_along_lines(idx_to_compute, N, n_droplets, 
                                   tA_sel, tB_sel, X_ask, K_sel)
    
    # "Do the measurement"
    y = sample_y(x, model = model)
    
    X_tell = x
    Y_tell = np.ravel(y)
    
    return X_tell, Y_tell


