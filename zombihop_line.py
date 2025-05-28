import utils
from acquisitions import *
import numpy as np
import pandas as pd
import random

from zombihop import *
#import diversipy
import numpy as np
import os
import joblib
import pandas as pd

import matplotlib.pyplot as plt

from sampler import line_bo_sampler
import sys

from linebo_plots import calc_3d_from_ternary


def run_zombi_main():

    seed = np.random.seed(1) # seed for repeatability

    ####################
    # MAIN SETTINGS - EDIT THESE
    D = 10 # Number of dimensions in th search space (and in Ackley function if you use that)
    # n_init = 5 # Number of initial data points.
    # model_type = 'ackley' # Choose one of these: 'ackley', 'poisson', 'dye'
    plot = True # Enable the main result plots (does not affect Line-BO plots)
    resolution = 3
    n_droplets = 30 # number of droplets to predict
    ####################

    # Get random initial compositions, the elements of x sum to 1
    X_init = np.random.dirichlet(np.ones(D), size=n_droplets)
    # if you want 3-decimals:
    X_init = np.round(X_init, 3)
    # and fix the last column
    X_init[:, -1] = 1.0 - X_init[:, :-1].sum(axis=1)
    
    # TO DO: Generate one initial line. Partially implemented here, not ready.
    ## Get one random initial composition, the elements of x sum to 1
    #X_init = np.random.dirichlet(np.ones(D), size=1)
    ## if you want 3-decimals:
    #X_init = np.round(X_init, 3)
    ## and fix the last column
    #X_init[:, -1] = 1.0 - X_init[:, :-1].sum(axis=1)
    ## Pick one line spanning the point above randomly from the search space with
    ## the unit sphere logic (one can also use random_generation_type='cartesian';
    ## in that case provide also p=X_init).
    #K_init = calc_K(D, M=2, constrain_sum_x=True, plotting=True, 
    #                generate_randomly=True, max_candidates=1,
    #                random_generation_type='spherical')#,
    #                #p=X_init)
    # TO DO: Use function pick_random_init_data() in linebo_fun.py to extract
    # the actual lines. Armi did not have time to check that the function still
    # works. Also TO DO: The dirichlet generation and calc_K could be moved to
    # inside that function.
    
    X_init = pd.DataFrame(X_init)
    
    # DUMMY Y EXPERIMENTAL FOR X_INIT
    Y_init = pd.DataFrame(np.random.uniform(1,4.5, size=X_init.shape[0]))





    # Run ZoMBI
    zombi = ZombiHop(seed = seed,                       # A random seed for model reproducibility, IMPORTANT: *change every time a new independent trial is run*
                    X_init = X_init,                   # X data
                    Y_init = Y_init,                   # fX data
                    Y_experimental = None,   # model to predict f(X) from X
                    Gammas = 10,                        # Number of hops to other needles
                    alphas = 10,#10,                       #(X/N)# Number of ZoMBI zoom-ins for each hop, zooming repeatedly in
                    n_draws_per_activation = 10,#10,       #X / 10 # Number of samples drawn for each zoom-in
                    acquisition_type = LCB_ada,        # acquisition function options: LCB, EI, LCB_ada, EI_abrupt
                    tolerance = 0.9,                  # Error tolerance of GP prediction, used to end a ZoMBI zoom-in and move to the next needle
                    penalty_width = 0.3,               # Width of penalty region about needle => inhibits BO searching from areas surrounding previously found needles
                    m = 5,                             # Top m-number of data points used to zoom in bounds
                    k = 5,                            # Top k-number of data points to keep
                    lower_bound = np.zeros(D),
                    upper_bound = np.ones(D),#np.array([1,np.sqrt(3)/2]),
                    resolution = resolution,                   # 20Number for the resolution of the mesh search space, e.g., resolution=10
                    sampler = line_bo_sampler)#

    X_all, Y_all, needle_locs, needles = zombi.run_experimental(n_droplets=n_droplets, n_vectors=11, verbose = True, plot = False)


    if plot == True:
        
        plt.figure(figsize=(10,4))
        for n in range(needles.shape[0]):
            plt.axhline(needles.iloc[n,0], c='r', ls='-', lw=0.5, alpha=1)
        plt.scatter(Y_all.index, Y_all, c='k')
        plt.ylabel('Target value')
        plt.xlabel('Number of Experiments')
        plt.grid()
        plt.legend(['Discovered Needles'])
        plt.show()
        
        
        plt.figure(figsize=(10,4))
        for n in range(needles.shape[0]):
            plt.axhline(needles.iloc[n,0], c='r', ls='-', lw=0.5, alpha=1)
        plt.plot(Y_all, c='k')
        plt.ylabel('Target value')
        plt.xlabel('Number of Experiments')
        plt.grid()
        plt.legend(['Discovered Needles'])
        plt.show()
        