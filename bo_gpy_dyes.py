#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:39:26 2023

@author: atiihone
"""

import GPyOpt
import pickle
import joblib
import pandas as pd
import numpy as np
import n_sphere
from itertools import product
import matplotlib.pyplot as plt
from numpy.linalg import solve, lstsq
import os
import sklearn
from mpl_toolkits.mplot3d import Axes3D
import dill
from datetime import datetime
import matplotlib as mpl
from plotting_v2 import triangleplot

def ackley(x, b=0.5, a=20, c=2*np.pi, limit=15, noise_level = 0):
    """
    x: vector of input values
    """
    x = (x-(1/x.shape[1])).T * limit
    
    d = len(x) # dimension of input vector x
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x*x) / d))
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    
    y = a + np.exp(1) + sum_sq_term + cos_term
    
    if noise_level > 0:
        
        # TO DO: This does not necessarily work in a parallelized implementation.
        # Look how I did it in #HPER.
        noise = np.random.normal(loc=0, scale = a * noise_level)
        
        y = y + noise
        
    return y

def poisson(x, noise_level = 0):
    
    y = poisson_model.predict(x)
    
    if noise_level > 0:
        
        # TO DO: This does not necessarily work in a parallelized implementation.
        # Look how I did it in #HPER.
        noise = np.random.normal(loc=0, scale = noise_level)
        
        y = y + noise
        
    return y

def sample_y(x, target_fun_idx, target_funs = {1: 'ackley', 2: 'poisson', 
             3: 'zombi', 4: 'experimental'}, model = None, noise_level = 0):
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

    if target_funs[target_fun_idx] == 'ackley':
        y = ackley(x, b=0.5, a=20, c=2*np.pi, limit=15, 
                   noise_level = noise_level).reshape((x.shape[0],1))
        
    elif target_funs[target_fun_idx] == 'poisson':
        y = poisson(x, noise_level = noise_level).reshape((x.shape[0],1))
        
    elif target_funs[target_fun_idx] == 'experimental':
        y = np.zeros((x.shape[0],1))
        y[:,:] = np.nan
        
    else:
        raise Exception("Not implemented.")
        
    return y

def build_constraint_str_gpyopt(X_variables, total_sum_btw = [0.995, 1.0], 
                                prefix = 'x[:,', postfix = ']'):

    c1 = ''
    c0 = str(total_sum_btw[0])

    for idx in range(len(X_variables)):
        c1 = c1 + prefix + str(idx) + postfix + ' + '
        c0 = c0 + ' - ' + prefix + str(idx) + postfix

    c1 = c1[0:-2] + '- ' + str(total_sum_btw[1])
    
    constraints = [{'name': 'constr_1', 'constraint': c0},
                   {'name': 'constr_2', 'constraint': c1}]
        
    return constraints

def indicator_constraints_gpyopt(x, constraints):
    """
    Returns array of ones and zeros indicating if x is within the constraints
    This implementation is from GPyOpt.
    """
    x = np.atleast_2d(x)
    I_x = np.ones((x.shape[0],1))
    if constraints is not None:
        for d in constraints:
            try:
                exec('constraint = lambda x:' + d['constraint'], globals())
                ind_x = (constraint(x) <= 0) * 1
                I_x *= ind_x.reshape(x.shape[0],1)
            except:
                print('Fail to compile the constraint: ' + str(d))
                raise
    return I_x

def define_bo(x_all, y_all, emin, emax, N, task_max = True, batch_size = 1,
              constrain_sum_x = False):
    """
    Defines a GPyOpt Bayesian optimization object with the given x, y data,
    search space boundaries & dimensionality, and sets the task as minimization
    or maximization.

    Parameters
    ----------
    x_all : Numpy array of shape (n_samples, n_dimensions)
        Input x data for Bayesian optimization.
    y_all : Numpy array of shape (n_samples, 1)
        Input y data for Bayesian optimization.
    emin : Numpy array of shape (1, n_dimensions)
        Lower boundaries of the search space.
    emax : Numpy array of shape (1, n_dimensions)
        Upper boundaries of the search space.
    N : Int
        Number of dimensions.
    task_max : boolean, optional
        True if the optimization task is maximization, False if minimization.
        The default is True.

    Returns
    -------
    BO_object : GPyOpt.methods.BayesianOptimization
        The Bayesian optimization object that has been initialized and fit with
        the input data. Batch size is one. 

    """
    
    # Set search space boundaries (GPyOpt).
    bounds = []
    for j in range(N):
        bounds.append({'name': str(j), 'type': 'continuous',
                       'domain': [emin[0,j], emax[0,j]]})
    
    constraints = None
    
    if constrain_sum_x == True:
        
        constraints = build_constraint_str_gpyopt(['X' + str(i) for i in range(N)])
    
    # Implemented with f=None because will eventually be used with experimental
    # data.
    BO_object = GPyOpt.methods.BayesianOptimization(f=None,
                                                    domain=bounds,
                                                    constraints=constraints,
                                                    acquisition_type='LCB',
                                                    normalize_Y=True,
                                                    X=x_all,
                                                    Y=y_all,
                                                    evaluator_type='local_penalization',
                                                    batch_size=1,#batch_size,
                                                    #acquisition_jitter=0.01,
                                                    noise_var = 0.1, #10e-12,# GPyOpt assumes normalized Y data at the point when variance is defined.
                                                    optimize_restarts = 25,#25
                                                    max_iters = 3000,#3000
                                                    exact_feval = False,
                                                    maximize = task_max,
                                                    exploration_weight = 2,#2.5, # For LCB: Higher value means more exploration.
                                                    ARD = True)
    
    return BO_object
    
def suggest_next_x(BO_object):
    """
    Suggest the next x datapoint to be sampled using the Bayesian optimization
    object.

    Parameters
    ----------
    BO_object : GPyOpt.methods.BayesianOptimization
        The Bayesian optimization object that has been initialized and fit with
        the input data collected this far.

    Returns
    -------
    p_next : Numpy array of shape (n_samples, n_dimensions)
        The next point(s) to sample suggested by BO_object. 

    """
    p_next = BO_object.suggest_next_locations()
    
    return p_next

def predict_from_BO_object(BO_object, x, unscale_y = True, return_std = False):
    """
    Predict the posterior mean value of the given x datapoint using the given
    Bayesian optimization object.

    Parameters
    ----------
    BO_object : GPyOpt.methods.BayesianOptimization
        The Bayesian optimization object that has been initialized and fit with
        the input data collected this far.
    x : Numpy array of shape (n_samples, n_dimensions)
        The x datapoint(s) to predict.

    Returns
    -------
    y : Numpy array of shape (n_samples, 1)
        The posterior mean value of y datapoint(s) predicted with BO_object.

    """
    y, std = BO_object.model.model.predict(x)
    
    y_train_unscaled = BO_object.Y
    
    if y_train_unscaled.shape[0] > 0:
        
        posterior_mean_true_units = y * np.std(y_train_unscaled) + \
            np.mean(y_train_unscaled)
        posterior_std_true_units = (np.std(y_train_unscaled)) * (std)
        
    if return_std is False:
        
        result = posterior_mean_true_units
        
    else:
        
        result = [posterior_mean_true_units, posterior_std_true_units]
    
    return result

def acq_from_BO_object(BO_object, x):
    """
    Compute the acquisition function value of the given x datapoint(s) using the given
    Bayesian optimization object.

    Parameters
    ----------
    BO_object : GPyOpt.methods.BayesianOptimization
        The Bayesian optimization object that has been initialized and fit with
        the input data collected this far.
    x : Numpy array of shape (n_samples, n_dimensions)
        The x datapoint(s) to predict.

    Returns
    -------
    y : Numpy array of shape (n_samples, 1)
        The acquisition function value of y datapoint(s) predicted with BO_object.

    """
    # TO DO: Is there difference among the two options?
    #a = BO_object.acquisition._compute_acq(x)
    a = BO_object.acquisition.acquisition_function(x)
    
    return a

def transform_to_Nplusone_dim(cart_coords_low_dim):
    
    # Assume the sum of each x coordinate is limited to one. Then, the
    # search space is effectively constrained to N-1 dimensional space that
    # spans points x_i = 1, x_j = 0, i != j, for all i in [1, ... , N].
    
    # True dimensionality of the search space
    N = cart_coords_low_dim.shape[1] + 1 
    # Eigenvectors of the high-dimensional search space.
    eigenvecs = np.diag(np.ones(N))
    
    # Eigenvectors of the constrained subspace.
    # Note! This parts needs to be done with an external constraint 
    # checking function if you want to create flexible constraints.
    # The eigenvecs_subspace_simple need to be chosen so that each one of them
    # fulfills the constraint. Additionally, the way of choosing orthogonal
    # eigenvecs_subspace_new may vary based on the constraint. These vectors
    # work for any sum(x_i) = value constraint.
    
    # Simple implementation that does not sample uniformly in N dimensions
    # (because the angle of the two eigenvecs is not pi/2).
    eigenvecs_base_idx = 0#np.random.randint(0,N)
    eigenvecs_others_idx = list(range(N))
    eigenvecs_others_idx.remove(eigenvecs_base_idx)
    eigenvecs_subspace_simple = (eigenvecs[eigenvecs_others_idx,:] - 
                          eigenvecs[eigenvecs_base_idx,:])
    
    # So let's pick new eigenvecs that have pi/2 angles.
    eigenvecs_base_idx = 0#np.random.randint(0,N)
    eigenvecs_others_idx = list(range(N-1))
    eigenvecs_others_idx.remove(eigenvecs_base_idx)
    eigenvecs_subspace_new = (-0.5*eigenvecs_subspace_simple[eigenvecs_base_idx,:] +
                     eigenvecs_subspace_simple)
    # And let's make them to have the same lengths.
    eigenvecs_subspace_new[eigenvecs_base_idx,:] = 4*eigenvecs_subspace_new[eigenvecs_base_idx,:]
    eigenvecs_subspace_new = (eigenvecs_subspace_new / (np.sqrt(np.sum(
        eigenvecs_subspace_new**2, axis=1))).reshape((N-1,1)))
    
    dot_prod = np.dot(eigenvecs_subspace_new[0,:], eigenvecs_subspace_new.T)
    if ((not np.isclose(dot_prod[eigenvecs_base_idx], 1, atol = 0.02)) or 
        (np.any(~np.isclose(dot_prod[eigenvecs_others_idx], 0, atol = 0.02)))):
        
        raise Exception('Eigenvectors are not orthogonal or scaling is wrong!')
    
    # Cartesian coordinates expressed in the true search space. Note that the
    # points are now "on the right plane" but do not necessarily fulfill the
    # conditions of sum(x_i)=1.
    cart_coords = np.matmul(cart_coords_low_dim, eigenvecs_subspace_new)
    
    print('K candidates after transform to actual dimensionality: ', cart_coords)
    
    if np.isclose(np.sqrt(np.sum(cart_coords**2)), 1, atol = 0.02) is False:
        
        raise Exception('K_cand are not scaled to one!')
    
    # Scaling of the values so that the points are actually within the subspace
    # boundaries is done just for clarity - for the purposes of this code the
    # previous step would already have sufficed.
    #cart_coords = cart_coords / np.reshape(np.sum(cart_coords, axis = 1), (N-1,1))
    
    return cart_coords

def calc_K(dimensionality, M, constrain_sum_x = False, plotting = True):
    """
    Calculate the locations of candidate points K (in cartesian coordinates)
    that will together with point P define candidate lines. Points K will in
    this implementation be fixed during the whole BO search. Each candidate line
    has a parameterized equation: x = p + k * t, i.e., each line goes through
    points P and P + K. Candidate points are chosen with M uniformly spaced
    space angles along each dimension N.

    Parameters
    ----------
    dimensionality : Integer
        Number of dimensions.
    M : Integer
        Number of candidate angles to be tested within each dimension.

    Returns
    -------
    cart_coords : Numpy array of shape (n_samples, n_dimensions)
        Cartesian coordinates of the candidate points that (together with
        a second point P) define the candidate lines to be compared. Candidate
        line is fully defined when it passes point P and one of points K_cand + P.

    """
    
    # The dimensionality of the serach is effectively reduced by one if the
    # search is constrained to sum(X_i) = 1.
    
    N = dimensionality
    
    if constrain_sum_x == True:
        
        N = dimensionality - 1
    
    # In spherical coordinates:
    # rows: points
    # columns: coordinates r, angles
        
    # Number of angles to be tested in total.
    K = M**(N-1)
    
    # Filling in the spherical coordinates of K equally distributed points around
    # the unit circle centered to point P. 
    sph_coords = np.zeros((K,N))
    # Unit circle, r=1. Radius is the first dimensions in n_sphere representation.
    sph_coords[:,0] = 1
    # The angle step will be multiplied with these numbers for each point K in each
    # dimension.
    km = np.array(list(product(range(M), repeat=N-1)))
    # Assign the spherical coordinate angles. There are N-1 angles. N-2 first ones
    # have a range [0,pi], the last one [0,2*pi]. We look at only the first half of
    # the hypersphere (lines will span both halves symmetrically anyway).
    # TO DO check: Which logic is the correct one?
    # ranges here are [0, pi/2] and [0,pi], respectively. 
    # OR
    # ranges here are [0,pi] and [0,pi], respectively (halving one dimension
    # already cuts out half of the hypersphere points, right?).
    #Dimensions 1, ..., N-2
    sph_coords[:, 1:-1] = km[:, 0:-1] * np.pi / (1*M)
    sph_coords[:, -1] = km[:, -1] * np.pi / M
    
    #print('Spherical coordinates of candidate points K:\n', sph_coords, '\n')
    
    cart_coords = n_sphere.convert_rectangular(sph_coords)
    
    # Remove duplicates (e.g., spherical coordinates (1, 0, theta) will result
    # in cartesian coordinates (1, 0, 0) with all theta angles). Note that
    # np.unique also sorts the array at the same time.
    cart_coords = np.unique(cart_coords, axis=0)
    
    if plotting == True:
        
        idx0_for_plot = 0
        idx1_for_plot = N-1
        
        plt.figure()
        plt.scatter(cart_coords[:,idx0_for_plot], cart_coords[:,idx1_for_plot])
        plt.xlabel('$x_' + str(idx0_for_plot) + '$')
        plt.ylabel('$x_' + str(idx1_for_plot) + '$')
        plt.title('All the points K')
        plt.show()
    
    if constrain_sum_x == True:
        
        # Return back to the true dimensionality of the search space.
        cart_coords = transform_to_Nplusone_dim(cart_coords)
    
    #print('Cartesian coordinates of candidate points K:\n', cart_coords, '\n')
    
    # TO DO: Is there some bug here in higher than 3D?
    #back_tr = n_sphere.convert_spherical(cart_coords)
    #print(back_tr, '\n')
    
    return cart_coords

def solve_t_matrix_eq(p, K_cand, e):
    """
    Solve matrix equation Ax = b, where A is a diagonal matrix. Here, A = [...],
    b = [...], x = [...].

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    K_cand : TYPE
        DESCRIPTION.
    e : TYPE
        DESCRIPTION.

    Returns
    -------
    t : TYPE
        DESCRIPTION.

    """
    
    K = K_cand.shape[0]
    N = K_cand.shape[1]
    
    b = np.tile(e - p, K).T

    # Solve t_min values for each point k.
    
    # Ordering: Point 0 dim 0, point 0, dim 1, ... point K dim N.
    diag_A = K_cand.flatten()
    
    # Indices of rows that cannot be solved (e.g. lines travelling along the
    # search space boundary but shifted by some amount, i.e., they will never
    # cross the boundary).
    idcs = np.ravel(np.where((diag_A == 0)))# & np.ravel((b != 0)))[0]
    
    diag_A_temp = np.delete(diag_A, idcs, axis = 0)
    b_temp = np.delete(b, idcs, axis = 0)
    
    # Confirm that the matrix is invertible. We assume that after the deletions
    # above, A is invertible (something is wrong if it is not). Normally, you
    # would check invertibility with a determinant but np.det() rounds to 0
    # with large matrices. So let's utilize diagonal matrix determinant rule
    # instead (A not invertible if det(A) = 0, det(diag) = d_11*d_22*...*d_NN).
    if np.any(diag_A_temp == 0) == True:
        
        # Not invertible, solve with least squares.
        A_temp = np.diag(diag_A_temp)
        t = lstsq(A_temp, b_temp, rcond = None)[0]
        print('Used least-squares solver (why?).')
        print('det(A)=', np.linalg.det(A_temp), 'N_unique_eigenvalues=', 
              np.unique(np.linalg.eig(A_temp)[0]).shape[0])
        print('Shape A_temp: ', A_temp.shape)
        
    else:
        
        diag_Ainv_temp = np.reshape(1/diag_A_temp, (-1,1))
        t = diag_Ainv_temp * b_temp
        
        #Ainv_temp = np.diag(diag_Ainv_temp)
        #t = np.matmul(Ainv_temp, b_temp)
        
        # Invertible matrix, can be solved exactly.
        #t = solve(A_temp, b_temp)
        
    # Fix the dimensionality of the variable.
    for i in idcs:
        t = np.insert(t, i, np.nan, axis=0)
    
    t = np.reshape(t, (K,N))
    
    return t
    
def plot_K_P_in_3D(a, p, title = None, first_point_label = 'K_cand', 
                   plot_triangle = True, lims = [-2,2]):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(elev=45, azim=60)
    
    #ax = plt.figure().add_subplot(projection='3d')
    
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('x2')
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)
    
    for i in range(a.shape[0]):
        
        ax.plot(np.ravel([p[:,0], p[:,0] + a[i,0]]), 
                np.ravel([p[:,1], p[:,1] + a[i,1]]),
                np.ravel([p[:,2], p[:,2] + a[i,2]]), c='b',
                linewidth = 0.5, linestyle = '--')
    
    if plot_triangle is True:
    
        ax.plot([1, 1], [0, 0], [lims[0],0], c = 'k', linewidth = 0.5,
                linestyle = '--')
        ax.plot([0, 0], [1, 1], [lims[0],0], c = 'k', linewidth = 0.5,
                linestyle = '--')
        ax.plot([0, 0], [0, 0], [lims[0],1], c = 'k', linewidth = 0.5,
                linestyle = '--')
        ax.plot([1, 0, 0, 1], [0, 1, 0, 0], [lims[0], lims[0], lims[0], lims[0]], 
                c = 'k', linewidth = 0.5, linestyle = '--')
        ax.plot([1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], c = 'k', 
                label = 'Constraint')
        
    else:
        
        ax.plot(np.ravel([p[:,0], 0]), 
                np.ravel([p[:,1], 0]),
                np.ravel([p[:,2], -2]), c='k',
                linewidth = 0.5, linestyle = '--')
        
    ax.scatter(p[:,0] + a[:,0], p[:, 1] + a[:,1], p[:,2] + a[:,2], c = 'b', 
               label = first_point_label)
    #ax.scatter(a2[:,0], a2[:,1], a2[:,2], c = 'r')
    ax.scatter(p[:,0], p[:,1], p[:,2], c = 'k', label='P')
    
    #ax.set_proj_type('ortho')
    plt.legend()
    plt.title(title)
    plt.show()
    
def extract_inlet_outlet_points(p, K_cand, emax, emin, M,
                                constrain_sum_x = False, plotting = True):
    """
    Extract the points A and B in cartesian coordinates. They are the points
    where the candidate lines K_cand arrive to and leave from the search space,
    respectively. Points A and B can also be defined with parameter tA and tB
    in the parameterized form of the line: x = p + k * t.

    Parameters
    ----------
    p : Numpy array of shape (1, n_dimensions)
        Point P that has been suggested by BO. Candidate line is fully defined
        when it passes point P and one of points K_cand + P.
    K_cand : Numpy array of shape (n_candidates, n_dimensions)
        Candidate points that define the candidate lines. Candidate line is
        fully defined when it passes point P and one of points K_cand + P.
    emin : Numpy array of shape (1, n_dimensions)
        Lower boundaries of the search space.
    emax : Numpy array of shape (1, n_dimensions)
        Upper boundaries of the search space.
    M : Integer
        Number of candidate angles to be tested within each dimension.

    Returns
    -------
    a : Numpy array of shape (n_candidates, n_dimensions)
        The inlet points of the candidate lines, i.e., the points where the
        arrives to the search space.
    b : Numpy array of shape (n_candidates, n_dimensions)
        The outlet points of the candidate lines, i.e., the points where the
        leaves the search space.
    tA : Numpy array of shape (n_candidates, 1)
        The parameterized presentation of each candidate line is x = p + k*t.
        In this representation, tA are the t values of inlet points A.
    tB : Numpy array of shape (n_candidates, 1)
        The parameterized presentation of each candidate line is x = p + k*t.
        In this representation, tB are the t values of outlet points B.

    """
    
    # This function takes the longest time during BO. Here, lstsq takes longest and
    # could be switched to an exact solver to make the code faster (read details
    # later in this function).
        
    # Number of dimensions
    N = p.shape[1]
    # Number of candidate points.
    K = K_cand.shape[0]
    
    # Solve t_min values for each point k. Ordering: Point 0 dim 0, 
    # point 0, dim 1, ... point K dim N.
    tmin = solve_t_matrix_eq(p, K_cand, emin)
    
    '''
    # Solve t_min values for each point k. Ordering: Point 0 dim 0, 
    # point 0, dim 1, ... point K dim N.
    A = np.diag(K_cand.flatten())
    b = np.tile(emin - p, K).T
    
    # Indices of rows that cannot be solved (e.g. lines travelling along the
    # search space boundary but shifted by some amount, i.e., they will never
    # cross the boundary.
    idcs = np.where(np.all(A == 0, axis = 1) & np.ravel((b != 0)))[0]
    #A[idcs, idcs] = np.nan
    #b[idcs,:] = np.nan
    A_temp = np.delete(A, idcs, axis = 0)
    A_temp = np.delete(A_temp, idcs, axis = 1)
    b_temp = np.delete(b, idcs, axis = 0)
    tmin = solve(A_temp,b_temp)
    for i in idcs:
        tmin = np.insert(tmin, i, np.nan)
    tmin = np.reshape(tmin, (K,N))
    
    #tmin = lstsq(A, b, rcond = None)
    #tmin = np.reshape(tmin[0], (K,N))
    '''    
    
    # Solve t_max values for each point k.
    tmax = solve_t_matrix_eq(p, K_cand, emax)
    
    '''
    A = np.diag(K_cand.flatten())
    b = np.tile(emax - p, K).T
    
    idcs = np.where(np.all(A == 0, axis = 1) & np.ravel((b != 0)))[0]
    A_temp = np.delete(A, idcs, axis = 0)
    A_temp = np.delete(A_temp, idcs, axis = 1)
    b_temp = np.delete(b, idcs, axis = 0)
    tmax = solve(A_temp,b_temp)
    for i in idcs:
        tmax = np.insert(tmax, i, np.nan)
    tmax = np.reshape(tmax, (K,N))
    
    #tmax = lstsq(A, b, rcond = None)
    #tmax = np.reshape(tmax[0], (K,N))
    '''
    tall = np.append(tmin, tmax, axis=1)
    
    # Candidates for the point where the line comes into the search space:
    tAcands = np.where(tall<0, tall, np.nan)
    # Candidates for the point where the line gets out from the search space:
    tBcands = np.where(tall>0, tall, np.nan)
    # Optimal t values for points A and B:
    tA = np.nanmax(tAcands, axis = 1, keepdims = True)
    tB = np.nanmin(tBcands, axis = 1, keepdims = True)
    
    # If any of the tA or tB above are nan at this point, it means the only
    # valid tA or tB value was t = 0 (i.e., P is at the boundary of the search
    # space and is thus either also point B or point A).
    # point P is on the edge of the search space, K is also there, and the t
    # value needs to be 0 for either A or B.
    tA[np.isnan(tA)] = 0
    tB[np.isnan(tB)] = 0
    
    # Actual points A and B for each point K.
    a = p + tA * K_cand
    b = p + tB * K_cand
    
    #print('Sums (A, B): ', np.sum(a, axis=1), np.sum(b, axis=1))
    #print('(A, B):\n', a, '\n', b, 'ņ')
    
    if constrain_sum_x == True:
        
        # Need to reduce the lengths of the lines if they do not otherwise fill the
        # constraint sum(x_i) = 1.
        
        for i in range(a.shape[0]):
            
            if test_constraint(a[[i],:]) == False:
                
                print("Constraint not fulfilled for A!")
                #raise Exception("Constraint not fulfilled! a = ", a, ", b = ",
                #                b, ", p = ", p, ", K_cand = ", K_cand, ", tA = ",
                #                tA, ", tB = ", tB)
            
            if test_constraint(b[[i],:]) == False:
                
                print("Constraint not fulfilled for B!")
                #raise Exception("Constraint not fulfilled! a = ", a, ", b = ",
                #                b, ", p = ", p, ", K_cand = ", K_cand, ", tA = ",
                #                tA, ", tB = ", tB)
            
    
    if (N == 3) and (plotting == True):
        
        plot_K_P_in_3D(K_cand, p, 'Real 3D space', first_point_label = 'K_cand')
        plot_K_P_in_3D(K_cand-p, p-p, 'P in origo', first_point_label = 'K_cand', 
                       plot_triangle = False)
        plot_K_P_in_3D(a-p, p, 'Real 3D space', first_point_label = 'A', lims = [0,1])
        plot_K_P_in_3D(b-p, p, 'Real 3D space', first_point_label = 'B', lims = [0,1])
        #print('P: ', p)
        #print('K: ', K_cand-p)
        
    
    # If any of the points are outside the search space at this point, it means
    # P is on the boundary of the search space and, additionally, there is no t
    # value that could make the line candidate get inside the search space. I.e.,
    # P is in these cases always in the corner, I think.
    # Let's set t values for those points to zero (per the algo above, "the other
    # point A or B" is already zero, so in practice the line length goes to zero).
    tA[np.any((a > emax), axis = 1)] = 0
    tA[np.any((a < emin), axis = 1)] = 0
    tB[np.any((b > emax), axis = 1)] = 0
    tB[np.any((b < emin), axis = 1)] = 0
    
    '''
    idx0_for_plot = 0
    idx1_for_plot = 1
    
    plt.figure()
    ##  "BO" point
    plt.scatter(p[:,idx0_for_plot], p[:,idx1_for_plot], c = 'k')
    ## Points K
    plt.scatter(K_cand[:,idx0_for_plot] + p[:,idx0_for_plot], 
                K_cand[:,idx1_for_plot] + p[:,idx1_for_plot], c='b')
    ## Points A
    plt.scatter(a[:,idx0_for_plot], a[:,idx1_for_plot], c = 'r')
    ## Points B
    plt.scatter(b[:,idx0_for_plot], b[:,idx1_for_plot], c = 'm')
    #for i in range(K):
    #    temp = np.append(a[[i],:], b[[i],:], axis = 0)
    #    plt.plot(temp[:,0], temp[:,1])
    #plt.xlabel('Cart. coordinate ' + str(idx0_for_plot))
    #plt.ylabel('Cart. coordinate ' + str(idx1_for_plot))
    #plt.title('All the points K')
    #plt.legend(['BO point P', 'Candidates K', 'Search space inlet points A', 
    #            'Search space outlet points B', 'Candidate line 0',
    #            'Candidate line 1', 'Candidate line 2'])
    #plt.xlim((emin[0,idx0_for_plot], emax[0,idx0_for_plot]))
    #plt.ylim((emin[0,idx1_for_plot], emax[0,idx1_for_plot]))
    plt.show()
    '''
    return a, b, tA, tB

def nearest_x(x_query, x_sampled):
    
    # NOTE: This version assumes equally spaced x points along an axis.
    
    N = x_sampled.shape[1]
    
    # Assumed to be the same in every dimension.
    n_steps_dim = np.round((x_sampled.shape[0])**(1/N))
        
    delta_x = np.zeros(N)
    
    for i in range(N):
        
        delta_x[i] = (x_sampled[-1,i] - x_sampled[0,i])/(n_steps_dim-1)
        
    nearest_x_steps = ((x_query - x_sampled[0,:]) / delta_x).round(0).astype(int)
    
    # If the dimension is constant.
    nearest_x_steps[:, delta_x == 0] = 0
    
    x_nearest = (nearest_x_steps * delta_x) + x_sampled[0,:]
    
    # NOTE: This works only for "C-contiguous style" with the last index 
    # varying the fastest. 
    idx = np.sum([(nearest_x_steps[:,i] * (n_steps_dim**(N-1-i))) for i in 
                   range(N)], axis = 0).astype(int)
    
    if np.any(idx > x_sampled.shape[0]) or np.any(idx < 0):
        
        raise Exception("Index calculation did not succeed.")
    
    if np.allclose(x_sampled[idx,:], x_nearest, rtol = 10e-3) is False:
        
        raise Exception("Acquisition array indexing or step size is not as expected.")
        
        
    '''
    # This version works for non-equally spaced x points.
    
    # Note: I tested this is faster than  np.linalg.norm + 
    # scipy.optimize.linear_sum_assignment. I sthere other ways to make this
    # faster?
    
    idx = np.zeros(x_query.shape[0], ) -1
    
    # TO DO: Vectorize.
    for i in range(x_query.shape[0]):
        
        idx[i] = np.argmin(np.sum((x_query[i,:] - x_sampled)**2, axis = 1))
    
        
    '''
    return idx

def get_acq_values(x, acq_object, acq_params):

    if isinstance(acq_object, dict) is True:
        
        # Assume dictionary contains two numpy arrays that are the x and y
        # values of the acquisition function along the search space.
        
        x_acq = acq_object['x']
        idx_x_steps = nearest_x(x, x_acq).astype(int)
        acq_values = acq_object['y'][idx_x_steps]
   
    else:
        
        # Assume the object can predict acquisition values directly.
        acq_values = acq_object(acq_params, x)
        
    return acq_values
    
def integrate_over_acqf(p, K, t_start, t_stop, n_points, acq_object, 
                        acq_max = True, acq_params = None):
    """
    Integrate over the acquisition function along the line that spans points
    P and P+K.

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    t_start : TYPE
        DESCRIPTION.
    t_stop : TYPE
        DESCRIPTION.
    n_points : TYPE
        DESCRIPTION.
    BO_object : TYPE
        DESCRIPTION.
    acq_max : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    I : TYPE
        DESCRIPTION.

    """
    t_steps = np.linspace(t_start, t_stop, n_points, axis = 0) # Works for arrays and single points.
    delta_t = np.abs(t_steps[1] - t_steps[0])
    
    x_steps = np.tile(p.T, n_points).T + (np.reshape(t_steps, (n_points, 1)) * K)
    
    acq_values = get_acq_values(x_steps, acq_object = acq_object, 
                                acq_params = acq_params)
    
    # In GPyOpt, the acquisition values are negative but could be otherwise
    # in other packages.
    
    # Caused issues because you cannot know the true range of vals and you might
    # end up having a different scaling btw points:
    # Normalize the acquisition values to [0,1] to make it easier to swap BO
    # packages.
    
    if acq_max is False:
        acq_values = -acq_values
    
    #if np.max(acq_values) != np.min(acq_values):
    #    
    #    acq_values_norm = (acq_values - np.min(acq_values))/(np.max(acq_values) - np.min(acq_values))
    #    
    #else:
    #    
    #    # Cannot normalize if there is no knowledge of the range.
    #    acq_values_norm = acq_values
        
    # Calculate the integral
    I = np.sum(acq_values) * delta_t #* K should in principle be here but K magnitude is always 1.
    
    return I

def integrate_all_K_over_acqf(p, K_cand, t_starts, t_stops, n_points, acq_object, 
                        acq_max = True, acq_params = None):
    """
    Integrate over the acquisition function along the line that spans points
    P and P+K.

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    t_start : TYPE
        DESCRIPTION.
    t_stop : TYPE
        DESCRIPTION.
    n_points : TYPE
        DESCRIPTION.
    BO_object : TYPE
        DESCRIPTION.
    acq_max : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    I : TYPE
        DESCRIPTION.

    """
    
    acq_values_all = np.zeros((K_cand.shape[0], n_points))
    delta_t_all = np.zeros((K_cand.shape[0], ))
    for i in range(K_cand.shape[0]):
        
        t_steps = np.linspace(t_starts[i], t_stops[i], n_points, axis = 0) # Works for arrays and single points.
        delta_t = np.abs(t_steps[1] - t_steps[0])
        delta_t_all[i] = delta_t
        x_steps = np.tile(p.T, n_points).T + (np.reshape(t_steps, (n_points, 1)) * K_cand[[i],:])
        
        acq_values = get_acq_values(x_steps, acq_object = acq_object, 
                                    acq_params = acq_params)
        acq_values_all[[i],:] = acq_values.T
    
    acq_min_val = np.min(acq_values_all)
    acq_max_val = np.max(acq_values_all)
    # In GPyOpt, the acquisition values are negative but could be otherwise
    # in other packages.
    
    # Caused issues because you cannot know the true range of vals and you might
    # end up having a different scaling btw points:
    # Normalize the acquisition values to [0,1] to make it easier to swap BO
    # packages.
    
    if acq_max is False:
        acq_values_all = -acq_values_all
    
    if acq_max_val != acq_min_val:
       
        acq_values_norm = (acq_values_all - acq_min_val)
        acq_values_norm = acq_values_norm/np.max(acq_values_norm)
       
    else:
        
        # Cannot normalize if there is no knowledge of the range.
        acq_values_norm = acq_values_all
        
    # Calculate the integral
    I = np.sum(acq_values_norm, axis = 1) * delta_t_all #* K should in principle be here but K magnitude is always 1.
    
    return I

def choose_K(BO_object, p, K_cand, emax = 1, emin = 0, M = 2, acq_max = True,
             selection_method = 'integrate_acq_line', constrain_sum_x = False,
             plotting = True):
    """
    Note that the selection method 'integrate_acq_line' is straightforward
    integration here, so it results in the preference toward longer lines
    rather than short ones (even if their acquisition function values would be 
    the same). This is good for regularization, as short lines provide less
    information.

    Parameters
    ----------
    BO_bject : TYPE
        DESCRIPTION.
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
        DESCRIPTION. The default is 'integrate_acq_line'.

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
                                       emin = emin, M = M,
                                       constrain_sum_x = constrain_sum_x,
                                       plotting = plotting)
    #print('K candidates:\n', K)
    #print('A candidates:\n', A)
    
    if selection_method == 'integrate_acq_line':
        
        I_all = np.empty((K_cand.shape[0],1)) 
        
        #for i in range(K_cand.shape[0]):
        #    
        #    I_all[i] = integrate_over_acqf(p, K_cand[[i],:], tA[i,:], tB[i,:], 
        #                                   500, acq_object = acq_from_BO_object, 
        #                                   acq_max = acq_max, acq_params = BO_object)
        
        I_all = integrate_all_K_over_acqf(p, K_cand, tA, tB, 500, 
                                  acq_object = acq_from_BO_object, 
                                  acq_max = acq_max, acq_params = BO_object)
        
        idx = np.argmax(I_all, axis = 0)
        print('Mean value of integrals along lines: ', np.mean(I_all), 
              '\nChosen value of the integral and its index: ', I_all[idx], idx,
              '\nAll values of the integrals that were compared: ', I_all)
        
    elif selection_method == 'random_line':
        
        idx = np.random.randint(0, A.shape[0])
        
    else:
        
        raise Exception("Not implemented.")
        
    A_sel = A[idx, :]
    B_sel = B[idx, :]
    tA_sel = tA[idx]
    tB_sel = tB[idx]
    K_sel = K_cand[idx, :]
    
    return A_sel, B_sel, tA_sel, tB_sel, K_sel

def test_constraint(x, upper_lim = 1, lower_lim = 0.995):
    
    xsum = np.sum(x, axis = 1)
    
    if (xsum >= lower_lim).all() and (xsum <= upper_lim).all():
    
        result = True
    
    else:
    
        result = False
        
    return result

def pick_random_init_data(n_init, N, emax, emin, M, K_cand,
                          constrain_sum_x = False, plotting = True):
    
    # Randomly selected init points P.
    p = np.random.rand(n_init, N)
    
    if constrain_sum_x == True:
        
        # Init points need to fulfill the constraint.
        for i in range(n_init):
            
            while test_constraint(p[[i],:]) == False:
                
                p[i,:] = np.random.rand(1, N)
                
    # Acquisition function is uniform at this stage so let's pick the line randomly.
    A_sel = np.empty(p.shape)
    B_sel = np.empty(p.shape)
    tA_sel = np.empty((p.shape[0],))
    tB_sel = np.empty((p.shape[0],))
    K_sel = np.empty(p.shape)
    
    for i in range(n_init):
        
        if M is not None:
        
            A_sel[i,:], B_sel[i,:], tA_sel[i], tB_sel[i], K_sel[i,:] = choose_K(
                None, p[[i], :], K_cand, emax = emax, emin = emin, M = M,
                selection_method = 'random_line', 
                constrain_sum_x = constrain_sum_x, plotting = plotting)
            
        else:
            
            A_sel[i,:] = np.nan
            B_sel[i,:] = np.nan
            tA_sel[i] = np.nan
            tB_sel[i] = np.nan
            K_sel[i,:] = np.nan
        
    return p, A_sel, B_sel, tA_sel, tB_sel, K_sel

def compute_x_coords_along_lines(idx_to_compute, N, n_droplets, tA_sel, tB_sel, 
                               p_sel, K_sel):
    
    # idx_to_compute is a list of indices.
    x_steps = np.empty((len(idx_to_compute), n_droplets, N))
    
    for k in range(len(idx_to_compute)):
        
        if ((tA_sel[idx_to_compute[k]] == tB_sel[idx_to_compute[k]]) or 
            (n_droplets == 1)):
            
            # Option 1: Point P is at the corner of the search space and the 
            # only point of the candidate line that lies within the search
            # space is P.
            # Option 2: Only one droplet is being printed in which case the
            # method should converge into single-point BO.
            x = p_sel[idx_to_compute]
        
        else:
            
            #print('tA_sel:\n', tA_sel[idx_to_compute[k]])
            # Steps along the line gradient (in parameterized form of the line).
            t_steps = np.reshape(np.linspace(tA_sel[idx_to_compute[k]], 
                                         tB_sel[idx_to_compute[k]], n_droplets, 
                                         axis = 0), (n_droplets, 1)) # Works for arrays and single points.
        
            # Same in cartesian coordinates.
            x_steps[k,:,:] = p_sel[idx_to_compute[k],:] + t_steps * K_sel[[idx_to_compute[k]],:]
            
            # Reshape for the computation purposes.
            x = np.reshape(x_steps, (x_steps.shape[0]*x_steps.shape[1], x_steps.shape[2]))
    
    return x

def append_bo_data(x, y, x_round, y_round, x_all = None, y_all = None):
    
    x_round.append(x)
    y_round.append(y)
    
    # Append to the cumulative x and y variables (that will go into BO).
    
    if x_all is None:
        
        # Round 0
        x_all = x.copy()
        y_all = y.copy()
        
    else:
        
        x_all = np.append(x_all, x, axis = 0)
        y_all = np.append(y_all, y, axis = 0)
    
    return x_round, y_round, x_all, y_all

def extract_opt_value(x_all, y_all, task_max):
    
    if task_max is True:
        
        idx_opt = np.argmax(y_all, axis = 0)
        
    else:
        
        idx_opt = np.argmin(y_all, axis = 0)
        
    y = y_all[idx_opt, :].copy()
    x = x_all[idx_opt, :].copy()
    
    return x, y

def extract_model_opt_value(BO_object, x_all, task_max):
    
    y_all = predict_from_BO_object(BO_object, x_all)
    
    if task_max is True:
        
        idx_opt = np.argmax(y_all, axis = 0)
        
    else:
        
        idx_opt = np.argmin(y_all, axis = 0)
        
    y = y_all[idx_opt, :].copy()
    x = x_all[idx_opt, :].copy()
    
    return x, y

def calc_regret(x_opt, y_opt, x_true, y_true = 0):
    
    regret_x = np.sum((x_opt - x_true)**2, axis = 1)
    regret_y = np.sum((y_opt - y_true)**2, axis = 1)
    
    return regret_x, regret_y


#def create_2d_grid(range_min=0, range_max=1, interval=0.01):
#   
#    a = np.arange(range_min, range_max, interval)
#    xt, yt = np.meshgrid(a,a, sparse=False)
#    points = np.transpose([xt.ravel(), yt.ravel()])
#    ## The x, y, z coordinates need to sum up to 1 in a ternary grid.
#    #points = points[abs(np.sum(points, axis=1)-1) < interval]
#    
#    return points

def create_Nd_grid(N, range_min=0, range_max=1, interval=0.01, 
                   constrain_sum_x = False):
    
    a = np.arange(range_min, range_max, interval)
    b = [a] * N
    grid = np.meshgrid(*b, sparse=False)
    points = np.reshape(np.array(grid).transpose(), (a.shape[0]**N, N))
    
    if constrain_sum_x == True:
        
        ## The x_i coordinates need to sum up to 1.
        # Note! This parts needs to be done with an external constraint 
        # checking function if you want to create flexible constraints.
        points = points[abs(np.sum(points, axis=1)-1) < interval]
    
    return points

def calc_ternary(x):
    
    x0_tern = 0.5 * ( 2.*x[:,0]+x[:,1] ) / (np.sum(x, axis = 1))
    x1_tern = 0.5*np.sqrt(3) * x[:,1] / (np.sum(x, axis = 1))
    
    return x0_tern, x1_tern

def calc_3d_from_ternary(x):
    
    x1 = 2/np.sqrt(3)*x[:,1]
    x0 = 2*x[:,0] - 2/np.sqrt(3)*x[:,1]
    x2 = 1 - x0 - x1
    x3d = np.stack((x0, x1, x2), axis = 1)
    
    return x3d

def plot_ternary(x, p_sel, K_sel, idx_to_compute):
    
    plt.figure()
    
    if x is not None:
        
        x0_tern, x1_tern = calc_ternary(x)
        plt.scatter(x0_tern, x1_tern, marker = '.', label = 'Samples')
    
    plt.plot([0.5,1,0,0.5], [0.866,0,0,0.866], 'k--', linewidth = 0.5,
             label = 'Search space') # TO DO: generalize
    x0_tern, x1_tern = calc_ternary(p_sel[idx_to_compute, :])
    plt.scatter(x0_tern, x1_tern, c='b', 
                marker = 's', label = 'BO-suggested point P')
    # K
    x0_tern, x1_tern = calc_ternary(K_sel[idx_to_compute, :] + p_sel[idx_to_compute, :])
    plt.scatter(x0_tern, x1_tern, c='b', 
                marker = 'x', label = 'Point K that defines the line direction')
    plt.legend()
    plt.show()

def plot_BO_progress(BO_object, x_plot, x, y, p_sel, K_sel, idx_to_compute, emin, 
                     emax, j, N, task_max, x_opt_true = None, y_opt_true = None,
                     x_opt_uncertainty = None):
    
    if N == 2:
        BO_object.plot_acquisition()
    
    if N == 3:
        
        surf_points = create_Nd_grid(3, range_min=0, range_max=1, interval=0.01, 
                           constrain_sum_x = True)
        
        surf_data = predict_from_BO_object(BO_object, surf_points)
        
        triangleplot(surf_points, surf_data, 
                     norm = mpl.colors.Normalize(vmin=np.min(surf_data), 
                                                 vmax=np.max(surf_data)), 
                     surf_axis_scale = 1, 
                     cmap = 'RdBu_r',
                     cbar_label = '$\mu(\overrightarrow{x})$', saveas = None, 
                     #surf_levels = np.arange(np.min(surf_data)-0.01, np.max(surf_data)+0.01, 
                     #                    (np.max(surf_data)-np.min(surf_data))/20),#[(-2+i*0.1) for i in range(31)],
                     scatter_points = x, scatter_color = np.ravel(y), 
                     cbar_spacing = None, cbar_ticks = None, #np.round(np.arange(np.min(surf_data), np.max(surf_data), 
                     #                    (np.max(surf_data)-np.min(surf_data))/5),
                     #decimals = 1), 
                     show = False)
        
        if x_opt_true is not None:
            
            # Plot optimum location.
            plt.scatter(calc_ternary(x_opt_true)[0],
                    calc_ternary(x_opt_true)[1],
                    marker = '*', c = 'm', zorder = 3)
            # Plot a circle representing the experimental uncertainty.
            if x_opt_uncertainty is not None:
                
                circle = plt.Circle((calc_ternary(x_opt_true)[0], 
                        calc_ternary(x_opt_true)[1]), x_opt_uncertainty, 
                       edgecolor='m', facecolor = None, linewidth = 0.5, fill = False, linestyle = '--')
                plt.gcf().gca().add_patch(circle)
                
        #plt.scatter(calc_ternary(np.array([[0.5, 0.5, 0]]))[0],
        #            calc_ternary(np.array([[0.5, 0.5, 0]]))[1],
        #            marker = '+', c = 'y')
        #plt.scatter(calc_ternary(np.array([[0, 0.5, 0.5]]))[0],
        #            calc_ternary(np.array([[0, 0.5, 0.5]]))[1],
        #            marker = 'x', c = 'y')
        #plt.scatter(calc_ternary(np.array([[0.5, 0, 0.5]]))[0],
        #            calc_ternary(np.array([[0.5, 0, 0.5]]))[1],
        #            marker = 's', c = 'y')
        
        n_opt_to_plot = 5
        colors_temp = (np.zeros((n_opt_to_plot,3)).T + np.array([i*0.5/n_opt_to_plot for i in range(n_opt_to_plot)]).T).T
        
        if task_max is True:
            
            # This line has not been tested.
            plt.scatter(calc_ternary(surf_points[surf_data[:,0].argsort()[(-1-n_opt_to_plot):(-1)],:])[0],
                    calc_ternary(surf_points[surf_data[:,0].argsort()[(-1-n_opt_to_plot):(-1)],:])[1],
                    marker = '+', c = colors_temp, zorder = 3)
        
        else:
            
            plt.scatter(calc_ternary(surf_points[surf_data[:,0].argsort()[:n_opt_to_plot],:])[0],
                    calc_ternary(surf_points[surf_data[:,0].argsort()[:n_opt_to_plot],:])[1],
                    marker = '+', c = colors_temp, zorder = 3)
        
        n_opt_to_plot = 5
        colors_temp = ((np.zeros((n_opt_to_plot,3)) + np.array([[0,0.5,0]])).T + np.array([i*0.5/n_opt_to_plot for i in range(n_opt_to_plot)]).T).T
        
        if task_max is True:
            
            # This line has not been tested.
            plt.scatter(calc_ternary(x[y[:,0].argsort()[(-1-n_opt_to_plot):(-1)],:])[0],
                    calc_ternary(x[y[:,0].argsort()[(-1-n_opt_to_plot):(-1)],:])[1],
                    marker = 'x', c = colors_temp, zorder = 3)
        
        else:
            
            plt.scatter(calc_ternary(x[y[:,0].argsort()[:n_opt_to_plot],:])[0],
                    calc_ternary(x[y[:,0].argsort()[:n_opt_to_plot],:])[1],
                    marker = 'x', c = colors_temp, zorder = 3)
        
        plt.show()
        
        #######
        
        [_, surf_data] = predict_from_BO_object(BO_object, surf_points, 
                                                return_std = True)
        
        triangleplot(surf_points, surf_data, 
                     norm = mpl.colors.Normalize(vmin=np.min(surf_data), 
                                                 vmax=np.max(surf_data)), 
                     surf_axis_scale = 1, 
                     cmap = 'RdBu_r',
                     cbar_label = '$\sigma(\overrightarrow{x})$', saveas = None, 
                     #surf_levels = np.arange(np.min(surf_data)-0.01, np.max(surf_data)+0.01, 
                     #                    (np.max(surf_data)-np.min(surf_data))/20),#[(-2+i*0.1) for i in range(31)],
                     scatter_points = x, scatter_color = 'k', 
                     cbar_spacing = None, cbar_ticks = None, #np.round(np.arange(np.min(surf_data), np.max(surf_data), 
                     #                    (np.max(surf_data)-np.min(surf_data))/5),
                     #decimals = 1), 
                     show = False)
        
        if x_opt_true is not None:
            
            # Plot optimum location.
            plt.scatter(calc_ternary(x_opt_true)[0],
                    calc_ternary(x_opt_true)[1],
                    marker = '*', c = 'm', zorder = 3)
            # Plot a circle representing the experimental uncertainty.
            if x_opt_uncertainty is not None:
                
                circle = plt.Circle((calc_ternary(x_opt_true)[0], 
                        calc_ternary(x_opt_true)[1]), x_opt_uncertainty, 
                       edgecolor='m', facecolor = None, linewidth = 0.5, fill = False, linestyle = '--')
                plt.gcf().gca().add_patch(circle)
                
        #plt.scatter(calc_ternary(np.array([[0.5, 0.5, 0]]))[0],
        #            calc_ternary(np.array([[0.5, 0.5, 0]]))[1],
        #            marker = '+', c = 'y')
        #plt.scatter(calc_ternary(np.array([[0, 0.5, 0.5]]))[0],
        #            calc_ternary(np.array([[0, 0.5, 0.5]]))[1],
        #            marker = 'x', c = 'y')
        #plt.scatter(calc_ternary(np.array([[0.5, 0, 0.5]]))[0],
        #            calc_ternary(np.array([[0.5, 0, 0.5]]))[1],
        #            marker = 's', c = 'y')
        
        n_opt_to_plot = 5
        colors_temp = (np.zeros((n_opt_to_plot,3)).T + np.array([i*0.5/n_opt_to_plot for i in range(n_opt_to_plot)]).T).T
        
        if task_max is True:
            
            # This line has not been tested.
            plt.scatter(calc_ternary(surf_points[surf_data[:,0].argsort()[(-1-n_opt_to_plot):(-1)],:])[0],
                    calc_ternary(surf_points[surf_data[:,0].argsort()[(-1-n_opt_to_plot):(-1)],:])[1],
                    marker = '+', c = colors_temp, zorder = 3)
        
        else:
            
            plt.scatter(calc_ternary(surf_points[surf_data[:,0].argsort()[:n_opt_to_plot],:])[0],
                    calc_ternary(surf_points[surf_data[:,0].argsort()[:n_opt_to_plot],:])[1],
                    marker = '+', c = colors_temp, zorder = 3)
        
        n_opt_to_plot = 5
        colors_temp = ((np.zeros((n_opt_to_plot,3)) + np.array([[0,0.5,0]])).T + np.array([i*0.5/n_opt_to_plot for i in range(n_opt_to_plot)]).T).T
        
        if task_max is True:
            
            # This line has not been tested.
            plt.scatter(calc_ternary(x[y[:,0].argsort()[(-1-n_opt_to_plot):(-1)],:])[0],
                    calc_ternary(x[y[:,0].argsort()[(-1-n_opt_to_plot):(-1)],:])[1],
                    marker = 'x', c = colors_temp, zorder = 3)
        
        else:
            
            plt.scatter(calc_ternary(x[y[:,0].argsort()[:n_opt_to_plot],:])[0],
                    calc_ternary(x[y[:,0].argsort()[:n_opt_to_plot],:])[1],
                    marker = 'x', c = colors_temp, zorder = 3)
        
        plt.show()
        
        #######
        
        
        # Acquisition function looks funny because there are so many zero outputs. To do: why?
        surf_points = create_Nd_grid(3, range_min=0, range_max=1, interval=0.02, 
                           constrain_sum_x = True)
        
        surf_data = get_acq_values(surf_points, acq_object = acq_from_BO_object, 
                                    acq_params = BO_object)
        
        surf_levels = np.arange(np.min(surf_data)-0.01, np.max(surf_data)+0.01, 
                            (np.max(surf_data)-np.min(surf_data))/20)
        triangleplot(surf_points, surf_data, 
                     norm = mpl.colors.Normalize(vmin=-2, vmax=1), 
                     surf_axis_scale = 1, 
                     cmap = 'RdBu_r',
                     cbar_label = '$A(\overrightarrow{x})$', saveas = None, 
                     surf_levels = surf_levels,
                     scatter_points = x, scatter_color = 'k', 
                     cbar_spacing = None, cbar_ticks = surf_levels, #np.round(np.arange(np.min(surf_data), np.max(surf_data), 
                                         #(np.max(surf_data)-np.min(surf_data))/5),
                     #decimals = 1), 
                     show = False)
        
        if x_opt_true is not None:
            
            plt.scatter(calc_ternary(x_opt_true)[0],
                    calc_ternary(x_opt_true)[1],
                    marker = '*', c = 'm', zorder = 3)
        plt.show()
        
        plt.figure()
        plt.scatter(calc_ternary(surf_points)[0],
                calc_ternary(surf_points)[1],
                marker = '.', c = np.ravel(surf_data))
        cbar = plt.colorbar()
        cbar.set_label('$A(\overrightarrow{x})$')
        plt.axis('off')
        plt.show()
    
        x0_tern, x1_tern = calc_ternary(x)
        
        plt.figure()
        plt.scatter(x0_tern, x1_tern, marker = '.', label = 'Samples')
        plt.plot([0.5,1,0,0.5], [0.866,0,0,0.866], 'k--', linewidth = 0.5,
                 label = 'Search space') # TO DO: generalize
        x0_tern, x1_tern = calc_ternary(p_sel[[idx_to_compute], :])
        plt.scatter(x0_tern, x1_tern, c='b', 
                    marker = 's', label = 'BO-suggested point P')
        # K
        x2_tern, x3_tern = calc_ternary(K_sel[[idx_to_compute], :] + p_sel[[idx_to_compute], :])
        plt.scatter(x2_tern, x3_tern, c='b', 
                    marker = 'x', label = 'Point K that defines the line direction')
        plt.plot([x0_tern, x2_tern], [x1_tern, x3_tern], c = 'b', linewidth = 0.5, linestyle = '--')
        plt.scatter(calc_ternary(x_opt_true)[0], calc_ternary(x_opt_true)[1],
                marker = '*', c = 'm', label = 'True optimum')
        plt.legend()
        plt.show()
    
    plt.figure()
    # Search space
    if j > 0:
        y_plot_bo = predict_from_BO_object(BO_object, x_plot)
        plt.scatter(x_plot[:,[0]], x_plot[:,[1]], marker = '.', c = y_plot_bo) # To do: change to contour
        cbar = plt.colorbar()
        cbar.set_label('$\mu(\overrightarrow{x})$')
    else:
        plt.plot([0,0,1,1,0], [0,1,1,0,0], 'k--', linewidth = 0.5,
                 label = 'Search space') # TO DO: generalize
    # Line
    plt.scatter(x[:,0], x[:,1], marker = '.', label = 'Samples')
    # P
    plt.scatter(p_sel[idx_to_compute, 0], p_sel[idx_to_compute, 1], c='b', 
                marker = 's', label = 'BO-suggested point P')
    # K
    plt.scatter(K_sel[idx_to_compute, 0] + p_sel[idx_to_compute, 0], 
                K_sel[idx_to_compute, 1] + p_sel[idx_to_compute, 1], c='b', 
                marker = 'x', label = 'Point K that defines the line direction')
    plt.xlim((emin[0,0]-1,emax[0,0]+1))
    plt.ylim((emin[0,1]-1,emax[0,1]+1))
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Sampled line(s) on round ' + str(j))
    plt.legend()#['$\mu(x)$', 'Samples', 'BO-suggested point P', 'Point K that defines the line direction'])
    plt.show()
    
    plt.figure()
    # Search space
    if j > 0:

        y_plot_bo = get_acq_values(x_plot, acq_object = acq_from_BO_object, 
                                    acq_params = BO_object) #acq_from_BO_object(BO_object, x_plot)
        
        plt.scatter(x_plot[:,[0]], x_plot[:,[1]], marker = '.', c = y_plot_bo) # To do: change to contour
        cbar = plt.colorbar()
        cbar.set_label('$A(\overrightarrow{x})$')
    else:
        plt.plot([0,0,1,1,0], [0,1,1,0,0], 'k--', linewidth = 0.5,
                 label = 'Search space') # TO DO: generalize
    # Line
    plt.scatter(x[:,0], x[:,1], marker = '.', label = 'Samples')
    # P
    plt.scatter(p_sel[idx_to_compute, 0], p_sel[idx_to_compute, 1], c='b', 
                marker = 's', label = 'BO-suggested point P')
    # K
    plt.scatter(K_sel[idx_to_compute, 0] + p_sel[idx_to_compute, 0], 
                K_sel[idx_to_compute, 1] + p_sel[idx_to_compute, 1], c='b', 
                marker = 'x', label = 'Point K that defines the line direction')
    plt.xlim((emin[0,0]-1,emax[0,0]+1))
    plt.ylim((emin[0,1]-1,emax[0,1]+1))
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Sampled line(s) on round ' + str(j))
    plt.legend()#['$\mu(x)$', 'Samples', 'BO-suggested point P', 'Point K that defines the line direction'])
    plt.show()
    
def plot_BO_main_results(x_opt, y_opt, y_round, n_rounds, title = ''):
    
    plt.figure()
    plt.plot(y_opt)
    plt.ylabel('$y_{opt}$')
    plt.xlabel('Round')
    plt.title(title)
    plt.show()    
    
    plt.figure()
    plt.plot(x_opt)
    plt.ylabel('$x_{i, opt}$')
    plt.xlabel('Round')
    plt.title(title)
    plt.legend([('i=' + str(i)) for i in range(x_opt.shape[1])], 
               ncol = int(np.ceil(x_opt.shape[1]/5)), loc = 'upper_right')
    plt.show()    

    plt.figure()
    for j in range(n_rounds):
        
        plt.scatter(np.zeros((y_round[j].shape)) + j, y_round[j], c='k')
    
    plt.ylabel('$y$')
    plt.xlabel('Round')
    plt.title(title)
    plt.show()    

def plot_landscapes(BO_object, x_plot, target_funs, target_fun_idx,
                    idx0 = 0, idx1 = 1):

    y_plot_gt = sample_y(x_plot, target_fun_idx, target_funs)
    
    y_plot_bo = predict_from_BO_object(BO_object, x_plot)
    
    plt.figure()
    plt.title('Ground truth')
    plt.scatter(x_plot[:,[idx0]], x_plot[:,[idx1]], c = y_plot_gt)
    plt.xlabel('$x_' + str(idx0) + '$')
    plt.ylabel('$x_' + str(idx1) + '$')
    cbar = plt.colorbar()
    cbar.set_label('$f(\overrightarrow{x})$')
    plt.show()
    
    plt.figure()
    plt.title('BO result')
    plt.scatter(x_plot[:,[idx0]], x_plot[:,[idx1]], c = y_plot_bo)
    plt.xlabel('$x_' + str(idx0) + '$')
    plt.ylabel('$x_' + str(idx1) + '$')
    cbar = plt.colorbar()
    cbar.set_label('$\mu(\overrightarrow{x})$')
    plt.show()

def calculate_gt(target_funs, target_fun_idx, task_max, points = None,
                 constrain_sum_x = None):
    
    if points is not None:
        
        y_gt = sample_y(points, target_fun_idx, target_funs)
        
    else:
        
        points_temp = create_Nd_grid(N, interval = 0.001*N, 
                                constrain_sum_x = constrain_sum_x)
        y_gt = sample_y(points_temp, target_fun_idx, target_funs)
    
    # Ground truth optimum value
    if task_max is True:
        
        idx_opt_gt = np.argmax(y_gt, axis = 0)
        
    else:
        
        idx_opt_gt = np.argmin(y_gt, axis = 0)
        
    y_opt_gt = y_gt[idx_opt_gt, :].copy()
    x_opt_gt = points[idx_opt_gt, :].copy()
    
    return x_opt_gt, y_opt_gt, y_gt

def read_init_data(file, x_columns_start_idx = 1, y_column_idx = 7,
                   average_duplicates = False, scale_x = False,
                   square_y = False):
    
    if isinstance(file, str) is True:
        
        # Only one file to read.
        files = [file]
        
    else:
        
        # A list of files.
        files = file
        
    x_all = np.empty((0,y_column_idx-x_columns_start_idx))
    y_all = np.empty((0,1))
    
    for f in files:
        
        data = pd.read_csv(f, index_col=0)
        print(f)
        x = data.iloc[:,x_columns_start_idx:y_column_idx].values
        y = data.iloc[:, y_column_idx].values.reshape((x.shape[0],1))
        
        if average_duplicates is True:
            
            unique_x, idx = np.unique(x, return_inverse=True, axis = 0)
            
            unique_y = np.zeros((unique_x.shape[0],1))
            unique_y[:,:] = np.nan
            
            for i in range(unique_x.shape[0]):
                
                idx_to_average = np.where(idx == i)
                
                unique_y[i, :] = np.mean(y[idx_to_average,0])
                
            x = unique_x
            y = unique_y
            
        x_all = np.concatenate((x_all, x), axis=0)
        y_all = np.concatenate((y_all, y), axis = 0)
        
    if scale_x is True:
        
        x_all = x_all/(np.reshape(np.sum(x_all, axis = 1), (x_all.shape[0],1)))
        
    if square_y is True:
        
        y_all = y_all**2
    
    return x_all, y_all
    

def simulated_batch_BO(n_rounds, n_init, n_droplets, N, target_fun_idx, target_funs, 
                       noise_level, emin, emax, task_max, plotting, 
                       file_init_data = None):
    
    if n_init > 0:
        
        n_init_data = n_init * n_droplets
        init_data_x, _, _, _, _, _ = pick_random_init_data(
            n_init_data, N, emax, emin, M = None, K_cand = None,
            plotting = plotting)
        
    else:
        
        init_data_x, init_data_y = read_init_data(file_init_data, 
                                                  y_column_idx = N+1)
        n_init_data = init_data_x.shape[0]
        
    # P is the point suggested by BO (information for debugging).
    p_sel = np.empty((n_init_data + n_rounds*n_droplets, N))
    p_sel[:,:] = np.nan
    p_sel[0:n_init_data, :] = init_data_x
    
    # Initialize variables for storing BO result data.
    
    # Data collected during the round in question. 
    x_round = []
    y_round = []
    
    # Optimum value vs round.
    x_opt = np.empty((n_rounds, N))
    y_opt = np.empty((n_rounds, 1))
    x_opt[:,:] = np.nan
    y_opt[:,:] = np.nan
    
    # Initialize the work variables that will be appended during the BO loop.
    x_all = None
    y_all = None
    
    # Run simulated BO.
    for j in range(n_rounds):
        
        print("Round " + str(j))
        
        # Choose the index (indices) of point p in the result data variables.
        if j == 0:
            idx_to_compute = [l for l in range(n_init_data)]
        
        else:
            idx_to_compute = [*range(n_init_data + (j - 1)*n_droplets, 
                                     n_init_data + j*n_droplets)]
        
        x = p_sel[idx_to_compute,:]
        
        # "Do the measurement"
        
        if ((j == 0) & (n_init == 0)):
            
            # Use data read from the init data file.
            y = init_data_y
            
        else:
            
            # Use simulated data.
            y = sample_y(x, target_fun_idx, target_funs, noise_level = noise_level)
    
        # Append to the BO data variables.
        x_round, y_round, x_all, y_all = append_bo_data(x, y, x_round, y_round, 
                                                        x_all, y_all)
        
        # Define BO object with data x_all, y_all.
        BO_object = define_bo(x_all, y_all, emin, emax, N, task_max = task_max,
                              batch_size = n_droplets,
                              constrain_sum_x = constrain_sum_x)
        
        # Suggest the p point for the next round.
        p_next = suggest_next_x(BO_object)
        p_sel[range((n_init + j)*n_droplets, (n_init + j + 1)*n_droplets), :] = p_next
        print('p_next: ', p_next)
        
        # Track optimum value
        x_opt[j, :], y_opt[j, :] = extract_opt_value(x_all, y_all, task_max)
        
        if plotting == True:
            
            plot_BO_progress(BO_object, x_plot, x_all, y_all, p_sel, np.zeros(p_sel.shape), 
                             idx_to_compute, emin, emax, j, N, task_max)
        
    return x_round, y_round, x_all, y_all, x_opt, y_opt
    
def init_plotting_grids(N, target_fun_idx, constrain_sum_x = False):

    # Initialize help variables required for plotting
    if N > 1:
        
        # Grid along the whole search space.
        if N <= 5:
            
            points = create_Nd_grid(N, interval = 0.01 * N, 
                                    constrain_sum_x = constrain_sum_x)
        
        else:
            
            points = create_Nd_grid(N, interval = 0.1 * N, 
                                    constrain_sum_x = constrain_sum_x)
            
        # Create a grid across the first two dimensions.
        points_temp = create_Nd_grid(2, constrain_sum_x = constrain_sum_x)
        
        # Ackley
        if target_fun_idx == 1:
            
            # In our shifted Ackley, optimum is at xi = 1/N, so the slice taken 
            # for plots looks the best if xi = 1/N, i>1.
            slice_loc = 1/N
        
        else:
            
            # Slice taken at xi=0, i>1.
            slice_loc = 0
            
        x_plot = np.concatenate((points_temp, 
                                     np.zeros((points_temp.shape[0], N-2)) 
                                     + slice_loc), axis = 1)
        
    else:
        
        raise Exception("Not implemented.")
        
    return points, x_plot

if __name__ == "__main__":
    
    poisson_model = joblib.load(os.getcwd()+'/../data/poisson_RF_trained.pkl')
    
    ###############################################################################
    # BO SETTINGS
    
    # Number of input dimensions.
    N = 3
    # Names of input dimensions (used only for saving the results):
    x_columns = ['Neonblue', 'Red', 'Yellow']#['Red', 'Pink', 'Blue', 'Yellow', 'Neonblue', 'White', 'Water'] # Leave pink, blue or white out if you need to reduce dims
    
    # Number of space angles to be tested for each dimension during BO (for
    # comparing which line the print).
    M = 21
    
    # Number of BO rounds.
    n_rounds = 1
    
    # Number of randomly sampled initial points at round 0 of BO. Will be read
    # from a csv file using read_init_data() if set to 0. The file will not be
    # used if n_init > 0.
    n_init = 0
    
    file_init_data = ['./Source_data/3D-6/rewards_sid_averaged_3D-6-S0.csv',
                      './Source_data/3D-6/rewards_sid_averaged_3D-6-S1.csv',
                      #'./Source_data/3D-6/rewards_sid_averaged_3D-6-S2.csv',
                      './Source_data/3D-6/rewards_sid_averaged_3D-6-S2-LCB.csv',
                      './Source_data/3D-6/rewards_sid_averaged_3D-6-S3.csv',
                      './Source_data/3D-6/rewards_sid_averaged_3D-6-S4.csv',
                      './Source_data/3D-6/rewards_sid_averaged_3D-6-S5.csv',
                      #'./Source_data/3D-6/rewards_sid_averaged_3D-6-S5-2.csv',
                      './Source_data/3D-6/rewards_sid_averaged_3D-6-S6.csv',
                      './Source_data/3D-6/rewards_sid_averaged_3D-6-S7.csv'
    ]

    scale_sum_init_x_to_one = True
    
    # Number of droplets printed when printing a line gradient (only for simulations).
    n_droplets = 1000
    
    # Lower bounderies of the search space along each dimension.
    emin = np.tile([[0]], N)
    # Upper boundaries of the search space along each dimension.
    emax = np.tile([[1]], N)
    
    # Is the BO task maximization (vs. minimization)?
    task_max = False
    # Is the acquisition function optimum at the maximum with the BO package you
    # use? Should be 'False' for GPyOpt.
    acq_max = False
    
    # Target fun, choose index among options.
    target_funs = {1: 'ackley', 2: 'poisson', 3: 'zombi', 4: 'experimental'}
    target_fun_idx = 4
    
    # True optimum location and value (and uncertainty of location). Set to None if unknown.
    y_opt_true = np.array([[0]])
    x_opt_true = np.array([[0.27, 0.459, 0.27]])
    x_opt_uncertainty = (0.05/4)*2 + 0.01 # 5 syringe pulls / 4 ml vial, each has +- 0.01 ml uncertainty, vials A and B. Plus 1% for catching the gradient in the hyperspectral camera correctly.
    
    
    plotting = True
    
    
    # Choose True if you want to pick random candidate points along the search 
    # space, choose False if points with evenly spaced space angles.
    random_K_cand = False
    
    selection_method = 'integrate_acq_line' # 'batch' # 'random_line' # 
    
    # Proportion of max value for Ackley, absolute st. dev. for Poisson.
    noise_level = 0.2
    
    # Set to True if you want to constrain the sum of x variable values to one
    # (i.e., proportional search). No other types of constrainst have been
    # implemented.
    constrain_sum_x = True
    
    # Set to True if you want to save all the results.
    save_results = True
    
    ###############################################################################
    # INITIALIZE VARIABLES
    
    # The first variable spans the whole search space as a uniform grid; the
    # second one plots a 2D slice (either with the rest of the variables set to
    # zero or, for Ackley, with them set to 0.3 that is the optimum value).
    points, x_plot = init_plotting_grids(N, target_fun_idx, 
                                         constrain_sum_x = constrain_sum_x)
    
    if selection_method == 'batch':
        
        # Run standard batch BO with batch size of n_droplets.
        x_round, y_round, x_all, y_all, x_opt, y_opt = simulated_batch_BO(
            n_rounds, n_init, n_droplets, N, target_fun_idx, target_funs, 
            noise_level, emin, emax, task_max, plotting, constrain_sum_x, 
            file_init_data = file_init_data) # TO DO
    
    else:
        
        # Run line BO.
        
        # Initialize arrays for logging BO progress.
        
        # P is the point suggested by BO (information for debugging).
        p_sel = np.empty((n_init + n_rounds, N))
        p_sel[:,:] = np.nan
        # K_sel is the selected candidate point (line "P K_sel" defines the direction of
        # the selected line gradient; information for debugging).
        K_sel = np.empty((n_init + n_rounds, N))
        K_sel[:,:] = np.nan
        # A_sel and B_sel are the inlet and outlet points of the selected line "P K_sel"
        # at the boundaries of the search space, respectively (for printing the line 
        # gradient).
        A_sel = np.empty((n_init + n_rounds, N))
        B_sel = np.empty((n_init + n_rounds, N))
        A_sel[:,:] = np.nan
        B_sel[:,:] = np.nan
        # tA_sel and tB_sel are the corresponding points in the parameterized
        # representation of the line; information for debugging). 
        tA_sel = np.zeros((n_init + n_rounds,))
        tB_sel = np.zeros((n_init + n_rounds,))
        tA_sel[:] = np.nan
        tB_sel[:] = np.nan
        
        # Initialize variables for storing BO result data.
        
        # Data collected during the round in question. 
        x_round = []
        y_round = []
        
        # Optimum value vs round.
        x_opt = np.empty((n_rounds, N))
        y_opt = np.empty((n_rounds, 1))
        y_opt_model = np.empty((n_rounds, 1))
        x_opt_model = np.empty((n_rounds, N))
        regret_x_opt = np.empty((n_rounds, 1))
        regret_y_opt = np.empty((n_rounds, 1))
        regret_x_opt_model = np.empty((n_rounds, 1))
        regret_y_opt_model = np.empty((n_rounds, 1))
        x_opt[:,:] = np.nan
        y_opt[:,:] = np.nan
        x_opt_model[:,:] = np.nan
        y_opt_model[:,:] = np.nan
        regret_x_opt[:,:] = np.nan
        regret_y_opt[:,:] = np.nan
        regret_x_opt_model[:,:] = np.nan
        regret_y_opt_model[:,:] = np.nan
        
        # Initialize the work variables that will be appended during the BO loop.
        x_all = None
        y_all = None
        
        # Initialize K candidate points that define the angle of the line together
        # with BO-suggested point P.
        K_cand = calc_K(N, M, constrain_sum_x)
        
        ###############################################################################
        # RUN BO
        
        # Pick the initial data points.
        
        if n_init > 0:
            
            # Pick the initial data points randomly.
            p_sel[0:n_init, :], A_sel[0:n_init, :], B_sel[0:n_init, :], tA_sel[0:n_init], tB_sel[0:n_init], K_sel[0:n_init, :] = pick_random_init_data(
                n_init, N, emax, emin, M, K_cand, 
                constrain_sum_x = constrain_sum_x, plotting = plotting) # TO DO
            
            plot_ternary(None, p_sel, K_sel, range(n_init))
            
        else:
            
            # Read the initial data points from file.
            init_data_x, init_data_y = read_init_data(file_init_data, 
                                                      y_column_idx = N+1,
                                                      scale_x = scale_sum_init_x_to_one)
            n_init_data = init_data_x.shape[0]
            
        # Run simulated BO.
        for j in range(n_rounds):
            
            print("Round " + str(j))
            #print('p_sel:\n', p_sel)
            
            # "Do the measurement"
            
            if (j == 0) & (n_init == 0):
                
                # Use data read from the result data file.
                x = init_data_x
                y = init_data_y
                idx_to_compute = 0
                
            else:
                
                # Simulate data.
                
                # Choose the index (indices) of point p in the result data variables.
                if (j == 0) & (n_init > 0):
                    idx_to_compute = [l for l in range(n_init)]
                    
                else:
                    idx_to_compute = [n_init + j - 1]
                
                # Compute the equally spaced x values along the selected line (or lines if 
                # there are multiple init points).
                x = compute_x_coords_along_lines(idx_to_compute, N, n_droplets, 
                                               tA_sel, tB_sel, p_sel, K_sel)
                
                y = sample_y(x, target_fun_idx, target_funs, noise_level = noise_level)
            
            # Append to the BO data variables.
            x_round, y_round, x_all, y_all = append_bo_data(x, y, x_round, y_round, 
                                                            x_all, y_all)
            
            # Define BO object with data x_all, y_all.
            BO_object = define_bo(x_all, y_all, emin, emax, N, 
                                  task_max = task_max, 
                                  constrain_sum_x = constrain_sum_x) # TO DO
            
            # Suggest the p point for the next round.
            p_next = suggest_next_x(BO_object)
            p_sel[n_init + j, :] = p_next
            print('p_next: ', p_next)
            
            # The inlet and outlet points of the line for the next round.
            
            # Candidate points K_cand change in every round and are chosen randomly
            # across the search space.
            if random_K_cand == True:
                # TO DO: Scale with emin and emax
                K_cand = (np.random.rand(K_cand.shape[0], K_cand.shape[1]) - 
                          p_sel[n_init + j, :]) # TO DO ensure the constraint
                
                if plotting == True:
                    
                    idx0_for_plot = 0
                    idx1_for_plot = N-1
                    
                    plt.figure()
                    plt.scatter(K_cand[:,idx0_for_plot], K_cand[:,idx1_for_plot])
                    plt.xlabel('$x_' + str(idx0_for_plot) + '$')
                    plt.ylabel('$x_' + str(idx1_for_plot) + '$')
                    plt.title('All the points K')
                    plt.show()
            
            A_sel[n_init + j, :], B_sel[n_init + j, :], tA_sel[n_init + j], tB_sel[n_init + j], K_sel[n_init + j, :] = choose_K(
                BO_object, p_next, K_cand, emax = emax, emin = emin, M = M, 
                acq_max = acq_max, selection_method = selection_method,
                constrain_sum_x = constrain_sum_x, plotting = plotting) # TO DO
            
            print('A next: ', A_sel[n_init + j, :], ', sums up to ', 
                  np.sum(A_sel[n_init + j, :]))
            print('B next: ', B_sel[n_init + j, :], ', sums up to ', 
                  np.sum(B_sel[n_init + j, :]))
            
            # Track optimum values.
            
            x_opt[j, :], y_opt[j, :] = extract_opt_value(x_all, y_all, task_max)
            print('y_opt sampled this far: ', y_opt[j,:], '\n')
            print('x_opt sampled this far: ', x_opt[j,:], '\n')
            
            x_opt_model[j, :], y_opt_model[j,:] = extract_model_opt_value(
                BO_object, points, task_max)
            print('y_opt predicted by model this far: ', y_opt_model[j,:], '\n')
            print('x_opt predicted by model this far: ', x_opt_model[j,:], '\n')
            
            if y_opt_true is not None:
                
                regret_x_opt[j, :], regret_y_opt[j,:] = calc_regret(
                    x_opt[j, :], y_opt[j, :], x_true = x_opt_true, 
                    y_true = y_opt_true)
                regret_x_opt_model[j, :], regret_y_opt_model[j,:] = calc_regret(
                    x_opt_model[j, :], y_opt_model[j, :], x_true = x_opt_true, 
                    y_true = y_opt_true)
                print('regret_x_opt (sampled) this far: ', regret_x_opt[j,:], 
                      '\n')
                print('regret_x_opt_model (predicted) this far: ', 
                      regret_x_opt_model[j,:], '\n')
                
            if plotting == True:
                
                plot_BO_progress(BO_object, x_plot, x_all, y_all, p_sel, K_sel, idx_to_compute, emin, 
                             emax, j, N, task_max, x_opt_true = x_opt_true,
                             y_opt_true = y_opt_true, 
                             x_opt_uncertainty = x_opt_uncertainty)
        

    
    ###############################################################################
    
    if n_rounds > 0:
        
        # Plot convergence and sampled y values.
        if plotting == True:
            
            plot_BO_main_results(x_opt, y_opt, y_round, n_rounds, title = 'random_K_cand = ' + 
                                 str(random_K_cand) + ', selection = ' + selection_method)
        
        # Plot ground truth and final surrogate model mean prediction along two first 
        # axes.
            
        #x_opt_gt, y_opt_gt, y_gt = calculate_gt(target_funs, target_fun_idx, 
        #                                              task_max, points = points,
        #                                              constrain_sum_x = constrain_sum_x)
        
        print('\nBO results:\ny_opt = ', y_opt[-1,0], 'found at\nx = ', x_opt[-1,:],
              ',\ny range during BO is ', [np.min(y_all), np.max(y_all)], 
        #      '.\n\nGround truth values:\ny_opt_gt = ', y_opt_gt[0], 
         #     ' at\nx_opt_gt = ', x_opt_gt[0,:], ',\n y range ',
         #     [np.min(y_gt), np.max(y_gt)], '.\n'
         )
        
        if plotting == True:
            
            if N > 1:
                
                plot_landscapes(BO_object, x_plot, target_funs, target_fun_idx,
                                    idx0 = 0, idx1 = 1)
                
    y_opt_gp_round_by_round = np.array([])
    y_opt_obs_round_by_round = np.array([])
    plt.plot(range(5), )
            
    # TO DO:
    # Think about adding human
    # Calculate real life time and compare to the national lab
    # Compare to single point BO, BO with random line alignment, and batch BO. 
    # Start another file and GIT.
    # The code needs to recognize +-0 if point p is on the edge.
    # Look at the latest run and also random cand = False run - Is x sometimes outside the search space?
    # Poisson sujuu tähän mennessä parhaiten LCB:llä (lambda=10) ja millä tahansa K-pisteiden valinnan ja viivan valinnan kriteereillä, kunhan se ei ole satunnainen + satunnainen. Noin 25 kierrosta riitti. 
    # Seuraavaksi: testaa zombi-hop, sitten jatka tämän kanssa D=10:llä. Konvergoituuko nyt?
    '''
    points = create_2d_grid()
    
    acq_for_plot = BO_object.acquisition.acquisition_function(points)
    if acq_max is False:
        acq_for_plot = -acq_for_plot
    acq_for_plot_norm = (acq_for_plot - np.min(acq_for_plot))/(
        np.max(acq_for_plot) - np.min(acq_for_plot)
        )
    
    plt.figure()
    plt.scatter(points[:,[0]], points[:,[1]], 
                c = acq_for_plot_norm)
    plt.colorbar()
    plt.show()
    '''
    
    if save_results is True:
        
        time_now = datetime.now().strftime("%y%m%d%H%M")
        session_filename = ('./Results/globalsave' + time_now + '.pkl')
        dill.dump_session(session_filename)
        # and to load the session again:
        #    dill.load_session(filename)
        
        if n_rounds > 0:
            
            idx_next_samples = [-1]
            
        else:
            
            idx_next_samples = range(n_init)
            p_next = np.array([[-1 for i in range(N)]])
            
        next_samples_df = pd.DataFrame(data = np.concatenate((p_next, 
                                                              A_sel[idx_next_samples, :],
                                                              B_sel[idx_next_samples, :]),
                                                             axis = 0), 
                                       columns = x_columns, 
                                       index = ['P_next'] + 
                                       ['A_next' for i in idx_next_samples] +
                                       ['B_next' for i in idx_next_samples])
        next_samples_df_4ml = next_samples_df.copy()*4
        next_samples_df_4ml.columns = next_samples_df_4ml.columns + '_4ml'
        next_samples_df = pd.concat((next_samples_df, next_samples_df_4ml), axis = 1)
        next_samples_df.to_csv('./Results/next_samples_' + time_now + '.csv')
        
        
        