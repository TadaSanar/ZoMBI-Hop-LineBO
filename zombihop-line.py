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

def ackley(x, b=0.5, a=20, c=2*np.pi, limit=15, invert = False, 
           shift_opt_from_origo = 0.3):
    """
    Evaluate Ackley or inverted Ackley function value at desired points.

    Parameters
    ----------
    x : Numpy array (n_points, n_dims)
        Function evaluation locations. 
    b : float, optional
        Ackley function parameter. The default is 0.5.
    a : float, optional
        Ackley function parameter. The default 
        is 20.
    c : float, optional
        Ackley function parameter. The default is 2*np.pi.
    limit : float, optional
        Ackley function parameter. The default is 15.
    invert : boolean, optional
        Invert Ackley. Set to 'True' if you want the Ackley peak to be a
        maximum, set to 'False' if you want the peak to be a minimum. The 
        default is True.
    shift_opt_from_origo : float, optional
        Shift optimum to this point from origo that is the default optimum
        location for Ackley. The default is True.
        

    Returns
    -------
    f : Numpy Array (n_points, 1)
        Ackley function values at points x.

    """
    
    x = (x - shift_opt_from_origo).T * limit
    d = len(x) # dimension of input vector x
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x*x) / d))
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    
    f = a + np.exp(1) + sum_sq_term + cos_term
    
    if invert is True:
        
        f = -f
        
    return f

def dye(x, assume_ternary = False):
    
    
    if dye_model is not None:
        
        if isinstance(x, pd.DataFrame):
            
            if assume_ternary is True:
                
                x3d = calc_3d_from_ternary(x.values)
                
            else:
                
                x3d = x.values
            
        else:
            
            if assume_ternary is True:
                
                x3d = calc_3d_from_ternary(x)
            
            else:
                
                x3d = x
            
        y = dye_model.predict(x3d)[0]
    
    else:
        
        # Use a dummy model.
        y = np.zeros((x.shape[0],1))
        y = np.ravel(y)
    
    return y
    

####################
# MAIN SETTINGS - EDIT THESE

N = 3 # Number of dimensions in th search space (and in Ackley function if you use that)
n_init = 2 # Number of initial data points.
model_type = 'ackley' # Choose one of these: 'ackley', 'poisson', 'dye'
plot = True # Enable the main result plots (does not affect Line-BO plots)

####################




X_init = pd.DataFrame(data = np.random.rand(n_init, N)) #np.diversipy.polytope.sample(n_points=100000, lower=lower, upper=upper, thin=0)

if model_type == 'poisson':
    # TO DO: Loading Poisson RF model or Poisson dataset from ZoMBI repo does not work
    # due to pickle version issues. Train a new RF model directly from the Poisson data.
    poisson_model_rf = joblib.load('./data/poisson_RF_trained.pkl')#os.getcwd()+'/../HPER/data/poisson_RF_trained.pkl')
    poisson_model = poisson_model_rf.predict # call the prediction function
    Y_init = pd.DataFrame(data = poisson_model(X_init), columns = ['y']) # fX dataset
    Y_experimental = poisson_model

elif model_type == 'ackley':

    Y_init = pd.DataFrame(data = (ackley(X_init.values)), columns = ['y']) # fX dataset
    Y_experimental = ackley
    
    x_grid, y_grid = np.meshgrid(np.arange(0,1,0.05), np.arange(0,1,0.05))
    plt.figure()
    plt.scatter(x_grid, y_grid, c = ackley(np.transpose([x_grid, y_grid])))
    plt.colorbar()
    plt.title('2-dimensional Ackley function')
    plt.show()

elif model_type == 'dye':
    
    if 'GPy' in sys.modules:
        
        dye_model = joblib.load('./data/3D-6-final-GP-model')
        
    else:
        
        dye_model = None
        print("Dummy model used instead of the GPy dye model.")
        
    Y_init = pd.DataFrame(data = dye(X_init), columns = ['y']) # fX dataset
    Y_experimental = dye
    
#Y_init = pd.DataFrame(poisson_model(np.array(X_init)), columns = Y.columns.values)


seed = np.random.seed(1) # seed for repeatability

# Run ZoMBI
zombi = ZombiHop(seed = seed,                       # A random seed for model reproducibility, IMPORTANT: *change every time a new independent trial is run*
                 X_init = X_init,                   # X data
                 Y_init = Y_init,                   # fX data
                 Y_experimental = Y_experimental,   # model to predict f(X) from X
                 Gammas = 10,                        # Number of hops to other needles
                 alphas = 10,#10,                       #(X/N)# Number of ZoMBI zoom-ins for each hop, zooming repeatedly in
                 n_draws_per_activation = 10,#10,       #X / 10 # Number of samples drawn for each zoom-in
                 acquisition_type = LCB_ada,        # acquisition function options: LCB, EI, LCB_ada, EI_abrupt
                 tolerance = 5,                  # Error tolerance of GP prediction, used to end a ZoMBI zoom-in and move to the next needle
                 penalty_width = 0.2,               # Width of penalty region about needle => inhibits BO searching from areas surrounding previously found needles
                 m = 5,                             # Top m-number of data points used to zoom in bounds
                 k = 5,                            # Top k-number of data points to keep
                 lower_bound = np.zeros(N),
                 upper_bound = np.ones(N),#np.array([1,np.sqrt(3)/2]),
                 resolution = 10,                   # 20Number for the resolution of the mesh search space, e.g., resolution=10
                 sampler = line_bo_sampler)#

X_all, Y_all, needle_locs, needles = zombi.run_virtual(verbose = False, plot = plot)


if model_type == 'dye':
    
    # A "3D" case constrained into a 2D triangle (ternary search space) can
    # also be defined as a 2D case (done above in the dye model definition step).
    # We need to transform the ternary values back into 3D in that case.
    
    X_all = calc_3d_from_ternary(X_all.values)
    needle_locs = calc_3d_from_ternary(needle_locs.values)


print('Needle locations: ', needle_locs, '\n')

print('Needle values: ', needles, '\n')

print('For a default Ackley test function, the optimum is at [0.3, 0.3, 0.3] with value of approx. 0. I think there is something wrong with my ZoMBI-Hop settings in this file because we are not seeing the optimum with the current settings.')

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
    
    if model_type == 'poisson':
        
        # Zoomed-in y axes in the plots-
        
        plt.figure(figsize=(10,4))
        for n in range(needles.shape[0]):
            plt.axhline(needles.iloc[n,0], c='r', ls='-', lw=0.5, alpha=1)
        plt.scatter(Y_all.index, Y_all, c='k')
        plt.ylim(-3,1)
        plt.ylabel('Target value')
        plt.xlabel('Number of Experiments')
        plt.grid()
        plt.legend(['Discovered Needles'])
        plt.show()
        
        plt.figure(figsize=(10,4))
        for n in range(needles.shape[0]):
            plt.axhline(needles.iloc[n,0], c='r', ls='-', lw=0.5, alpha=1)
        plt.plot(Y_all, c='k')
        plt.ylim(-3,1)
        plt.ylabel('Target Value')
        plt.xlabel('Number of Experiments')
        plt.grid()
        plt.legend(['Discovered Needles'])
        plt.show()
