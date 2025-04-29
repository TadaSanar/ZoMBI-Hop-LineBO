#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:39:26 2023

@author: atiihone
"""

import GPy
import GPyOpt
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import dill
from datetime import datetime

from linebo_util import create_Nd_grid, read_init_data, append_bo_data, extract_opt_value, calc_regret
from linebo_wrappers import define_bo, suggest_next_x, extract_model_opt_value
from linebo_fun import pick_random_init_data, calc_K, compute_x_coords_along_lines, choose_K
from linebo_plots import plot_BO_main_results, plot_BO_progress, plot_landscapes, init_plotting_grids, plot_ternary

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

def calculate_gt(target_funs, target_fun_idx, task_max, points = None,
                 constrain_sum_x = None, N = None):
    
    if points is not None:
        
        # Calculate ground truth for the given points. 
        y_gt = sample_y(points, target_fun_idx, target_funs)
        
    else:
        
        # Create a grid and calculate ground truth for these points.
        if N is not None:
            
            
            points_temp = create_Nd_grid(N, interval = 0.001*N, 
                                         constrain_sum_x = constrain_sum_x)
            
        else:
            
            Exception('Give N, i.e., the dimensions of the search space for calculating the ground truth values.')
            
        y_gt = sample_y(points_temp, target_fun_idx, target_funs)
    
    # Ground truth optimum value
    if task_max is True:
        
        idx_opt_gt = np.argmax(y_gt, axis = 0)
        
    else:
        
        idx_opt_gt = np.argmin(y_gt, axis = 0)
        
    y_opt_gt = y_gt[idx_opt_gt, :].copy()
    x_opt_gt = points[idx_opt_gt, :].copy()
    
    return x_opt_gt, y_opt_gt, y_gt


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

if __name__ == "__main__":
    
    poisson_model = './data/poisson_RF_trained.pkl'
    
    ###############################################################################
    # BO SETTINGS
    
    # Number of input dimensions.
    N = 3
    # Names of input dimensions (used only for saving the results):
    x_columns = ['Neonblue', 'Red', 'Yellow']#['Red', 'Pink', 'Blue', 'Yellow', 'Neonblue', 'White', 'Water'] # Leave pink, blue or white out if you need to reduce dims
    
    # Lower bounderies of the search space along each dimension.
    emin = np.tile([[0]], N)
    # Upper boundaries of the search space along each dimension.
    emax = np.tile([[1]], N)
    # Set scale_sum_init_x_to_one = True if you want to scale the input "x"
    # data so that their sum is one. This is handy if you are actually
    # interested in proportions (like 27% x0, 73% x1) but you synthesize a
    # larger or variable volume (like 1.1 ml of x0, 3.0 ml of x1).
    # Note: Set emin = np.tile([[0]], N) and emax = np.tile([[1]], N) if you
    # use this option, otherwise the points would be allocated to wrong
    # locations!
    scale_sum_init_x_to_one = True
    
    # Number of space angles to be tested for each dimension during BO (for
    # comparing which line should be selected for to be printed). More is
    # better, unless the run gets too slow.
    M = 10
    # Number of droplets printed when printing a line gradient (this variable
    # is used only if running simulations).
    n_droplets = 30
    
    # Number of BO rounds.
    # Set any positive integer if you run simulations with a target function.
    # Set n_rounds = 0 if you want to get only the randomly selected initial
    # samples (which are also lines).
    # Set n_rounds = 1 if you run experiments and want to get only the next
    # round suggestions.
    n_rounds = 1
    
    # Number of randomly sampled initial samples (lines) at round 0 of BO.
    # Set n_init = 0 if you want to read the initial data from csv files
    # with file paths listed in 'file_init_data'.
    # Set n_init to a positive integer if you want to use randomly sampled
    # initial samples (lines) instead. In this case the csv files with paths
    # in 'file_init_data' are not used at all.
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
    
    # Set task_max = False if the BO task is minimization.
    # Set task_max = True if the BO task is maximization.
    task_max = False
    # Is the acquisition function 'optimum' its maximum value with the BO
    # package you use? Should be 'False' for GPyOpt.
    acq_max = False
    
    # Target fun, choose the index among options.
    target_funs = {1: 'ackley', 2: 'poisson', 3: 'zombi', 4: 'experimental'}
    target_fun_idx = 4
    
    # True optimum location and value (and uncertainty of location).
    # Set to None if unknown.
    y_opt_true = np.array([[0]])
    x_opt_true = np.array([[0.27, 0.459, 0.27]])
    x_opt_uncertainty = (0.05/4)*2 + 0.01 # Logic: 5 syringe pulls / 4 ml vial,
    # each has +- 0.01 ml uncertainty, vials A and B. Plus 1% for catching the
    # gradient start location in the hyperspectral camera inaccurately.
    
    # Plot figures?
    plotting = False
    
    # Candidate points K define together with BO-suggested point P the
    # line candidates PK (another way to think about this is that K together
    # with P define the space angle candidates for the line).
    # Set random_K_cand = True if you want to pick random candidate locations K.
    # Set random_K_cand = False if you want points with evenly spaced space angles.
    random_K_cand = False
    
    # Method for selecting the line among the candidates PK.
    # Set selection_method = 'integrate_acq_line' if you want to integrate over
    # the acquisition function value over the search space (constraint included).
    # Set selection_method = 'random_line' if you want to choose randomly among
    # the candidates.
    # Set selection_method = 'batch' if you want to run just standard batch BO
    # with no line-BO.
    selection_method = 'integrate_acq_line' # 'batch' # 'random_line' # 
    
    # Proportion of max value for Ackley, absolute st. dev. for Poisson.
    noise_level = 0.2
    
    # Set to True if you want to constrain the sum of x variable values to one
    # (i.e., proportional search). No other types of constraints have been
    # implemented.
    # Set to False if you want to run BO across the whole unconstrained hypercube.
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
                                  constrain_sum_x = constrain_sum_x,
                                  implementWithBOPackage = 'GpyOpt')
            
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
                constrain_sum_x = constrain_sum_x, plotting = plotting, acq_params = None) # TO DO
            
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
                             emax, j, N, task_max,  x_opt_true = x_opt_true,
                             y_opt_true = y_opt_true, 
                             x_opt_uncertainty = x_opt_uncertainty,
                             acq_params = None)
        

    
    ###############################################################################
    
    if n_rounds > 0:
        
        # Plot convergence and sampled y values.
        if plotting == True:
            
            plot_BO_main_results(x_opt, y_opt, y_round, n_rounds, title = 'random_K_cand = ' + 
                                 str(random_K_cand) + ', selection = ' + selection_method)
        
        
        # Report the results.
        print('\nBO results:\ny_opt = ', y_opt[-1,0], 'found at\nx = ', x_opt[-1,:],
              ',\ny range during BO is ', [np.min(y_all), np.max(y_all)])
        
        # Calculate ground truth optimum from the target functions
        # (keep commented out unless runnig a simulation).
        #x_opt_gt, y_opt_gt, y_gt = calculate_gt(target_funs, target_fun_idx, 
        #                                              task_max, points = points,
        #                                              constrain_sum_x = constrain_sum_x)
        #
        #print('.\n\nGround truth values:\ny_opt_gt = ', y_opt_gt[0], 
        #     ' at\nx_opt_gt = ', x_opt_gt[0,:], ',\n y range ',
        #     [np.min(y_gt), np.max(y_gt)], '.\n')
        
        # Plot ground truth (if known) and the final surrogate model mean 
        # prediction along the two first axes.
        
        if plotting == True:
            
            if N > 1:
                
                y_plot_gt = sample_y(x_plot, target_fun_idx, target_funs)
                
                plot_landscapes(BO_object, x_plot, y_plot_gt,
                                    idx0 = 0, idx1 = 1)
                
    y_opt_gp_round_by_round = np.array([])
    y_opt_obs_round_by_round = np.array([])
    plt.plot(range(5), )
            
    # TO DO:
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
        
        
        