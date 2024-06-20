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

from linebo_fun import extract_inlet_outlet_points, compute_x_coords_along_lines, calc_K, integrate_over_acqf, choose_K

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
def choose_K_acq_zombihop(acq_params, p, K_cand, emax = 1, emin = 0, emax_global = None,
                          emin_global = None, M = 2, acq_max = True, 
                          selection_method = 'integrate_acq_line', 
                          constrain_sum_x = False, plotting = 'plot_all'):
    """
    Note that the selection method 'integrate_acq_line' is straightforward
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
    
    if constrain_sum_x is True:
        
        raise Exception("Constraint has not been implemented in ZoMBI-Hop so do not use it with Line-BO, either. ")
    
    if acq_params['acq_type'] is None:
        
        # Use dictionary as the acquisition object and "BO_object".
        
        BO_object_zombihop = None
        
    else:
        
        # TO DO: Clarify structure of the code from here onward so that BO_object 
        # would always be either BO_object or None, and acquisition-related stuff
        # would all be forwarded in acq_params.
        
        BO_object_zombihop = acq_fun_zombihop
        
    # Choose K using the local zoomed-in search space boundaries.
    
    # NOTE: Line-BO has not been properly tested for emin other than
    # zero and emax other than 1 even though Line-BO has been written so
    # that it should work with any boundary.
    
    emax_linebo = emax
    emin_linebo = emin
        
    # TO DO: Confirm that the acquisition function parameters are actually
    # delivered to the ZoMBI-Hop acquisition function when using it inside
    # choose_K().
    A_sel, B_sel, tA_sel, tB_sel, K_sel = choose_K(BO_object_zombihop, p = p,
                                                       K_cand = K_cand, 
                                                       emax = emax_linebo,
                                                       emin = emin_linebo,
                                                       M = M, 
                                                       acq_max = acq_max,
                                                       selection_method = selection_method,
                                                       constrain_sum_x = constrain_sum_x,
                                                       plotting = plotting,
                                                       acq_params = acq_params)
        
    if emax_global is not None:
        
        # Use the global ZoMBI-Hop borders for defining points A and B
        # (in contrast to using the zoomed-in local borders).
        
        idx_K_sel = np.ravel(np.argwhere((K_cand == K_sel).all(axis=1)))
        
        # Extend the chosen line to the global search space.
        A_sel, B_sel, tA_sel, tB_sel, K_sel = extract_inlet_outlet_points(p, 
                                                                          K_cand = K_cand[[idx_K_sel],:], 
                                                                          emax = emax_global,
                                                                          emin = emin_global, 
                                                                          M = M)
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
                    n_droplets = 30, M = 21, acq_max = False, 
                    selection_method = 'integrate_acq_line', acq_GP = None, 
                    acq_type = None, acq_n = None, acq_fX_best = None, 
                    acq_ratio = None, acq_decay = None, acq_xi = None, 
                    acq_ftype = None, plotting = 'plot_all'):
    """
    This is the main Line-BO sampler to be used with ZoMBI-Hop. You should not
    need to modify any other functions than this to switch between the
    features Armi has implemented by 6/19/2024.

    Parameters
    ----------
    X_ask : TYPE
        Point P suggested by ZoMBI-Hop.
    x_acquisitions : TYPE
        Locations of the points included into the ZoMBI-Hop acquisitions matrix.
    acquisitions : TYPE
        ZoMBI-Hop acquisitions matrix.
    model : TYPE
        Callable for sampling the ground truth ("performing the measurement").
    emin : TYPE, optional
        Lower boundary of the (local, zoomed-in) search space. The default is 0.
    emax : TYPE, optional
        Lower boundary of the (local, zoomed-in) search space. The default is 1.
    emin_global : TYPE, optional
        Lower boundary of the global search space. The default is 0.
    emax_global : TYPE, optional
        Upper boundary of the global search space.  The default is 1.
    n_droplets : TYPE, optional
        Number of droplets "to be synthesized" when sampling from 'model'.
        The default is 3.
    M : TYPE, optional
        Number of space angles to be considered per dimension. The default is 21.
    acq_max : TYPE, optional
        Set to True if the most promising value of the acquisition function is
        its maximum. Set to False if it is its minimum. The default is False.
    selection_method : TYPE, optional
        Method of selecting the best line. Options are 'integrate_acq_line' and
        'random_line'. The default is 'integrate_acq_line'.
    acq_GP : TYPE, optional
        The GP surrogate model that will be forwarded to the acquisition function.
        Will be used only if the acquisitions are defined as a callable function
        (see argument 'acq_type'). See the description from ZoMBI-Hop 
        documentation. The default is None.
    acq_type : TYPE, optional
        The type of the acquisitions object. Can be either a callable ZoMBI-Hop
        -compatible acquisition function, or None, in which case the ZoMBI-Hop
        acquisitions matrix defined by 'acquisitions' and 'x_acquisitions' will
        be used in Line-BO. The default is None.
    acq_n : TYPE, optional
        Acquisition argument 'n' that will be forwarded to the acquisition function.
        Will be used only if the acquisitions are defined as a callable function.
        See the description from ZoMBI-Hop documentation.  The default is None.
    acq_fX_best : TYPE, optional
        Acquisition argument 'fX_best' that will be forwarded to the acquisition function.
        Will be used only if the acquisitions are defined as a callable function.
        See the description from ZoMBI-Hop documentation.  The default is None.
    acq_ratio : TYPE, optional
        Acquisition argument 'ratio' that will be forwarded to the acquisition function.
        Will be used only if the acquisitions are defined as a callable function.
        See the description from ZoMBI-Hop documentation.  The default is None.
    acq_decay : TYPE, optional
        Acquisition argument 'decay' that will be forwarded to the acquisition function.
        Will be used only if the acquisitions are defined as a callable function.
        See the description from ZoMBI-Hop documentation.  The default is None.
    acq_xi : TYPE, optional
        Acquisition argument 'xi' that will be forwarded to the acquisition function.
        Will be used only if the acquisitions are defined as a callable function.
        See the description from ZoMBI-Hop documentation.  The default is None.
    acq_ftype : TYPE, optional
        Acquisition argument 'ftype' that will be forwarded to the acquisition function.
        Will be used only if the acquisitions are defined as a callable function.
        See the description from ZoMBI-Hop documentation.  The default is None.
    plotting : TYPE, optional
        The level of producing plots during the ZoMBI-Hop run. Options are:
        'plot_all', 'plot_few', 'plot_none'. The default is 'plot_all'.

    Returns
    -------
    X_tell : TYPE
        DESCRIPTION.
    Y_tell : TYPE
        DESCRIPTION.

    """
        
    # Ensure search space boundaries are provided as a 2D numpy array. 
    emin, emax = search_space_to_nparr(emin, emax, X_ask.shape[1])
    
    # NOTE
    # This is where emin and emax could be increased by a marging of numerical
    # error to make sure the P points are always strictly inside the search
    # space boundaries.
    
    
    # Ensure search space boundaries are provided as a 2D numpy array.
    # (The global boundaries are disregarded in the rest of the code if None.)
    if emin_global is not None:
        
        emin_global, emax_global = search_space_to_nparr(emin_global, 
                                                         emax_global, 
                                                         X_ask.shape[1])
    
    idx_to_compute = list(np.arange(X_ask.shape[0]))
    #idx_to_compute = np.reshape(idx_to_compute, (-1,1))
    N = X_ask.shape[1]
    
    # Initialize K candidate points that define the angle of the line together
    # with BO-suggested point P.
    K_cand = calc_K(N, M, plotting = False) # TO DO: We could change sampler to an object and calculate K only once.
        
    # The inlet and outlet points of the line for the next round.
    
    if acq_type is None:
        
        # No acquisition function provided; use acquisition array instead.
        # This option is slower.
        
        acq_params = {'acq_type': None,
                      'acq_dictionary': {'x': x_acquisitions, 
                                         'y': acquisitions}}
        
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
        M = M, acq_max = acq_max, selection_method = selection_method,
        plotting = plotting)
    
    # Compute the equally spaced x values along the selected line (or lines if 
    # there are multiple init points).
    x = compute_x_coords_along_lines(idx_to_compute, N, n_droplets, 
                                   tA_sel, tB_sel, X_ask, K_sel)
    
    # "Do the measurement"
    y = sample_y(x, model = model)
    
    X_tell = x
    Y_tell = np.ravel(y)
    
    print("ZoMBI-Hop suggested: \n - Point P=" + str(X_ask) + "\n - emin=" + 
          str(emin) + "\n - emax=" + str(emax) + "\nLine-BO-suggested points:\n -A=" + 
          str(A_sel) + "\n -B=" + str(B_sel))
    
    return X_tell, Y_tell


