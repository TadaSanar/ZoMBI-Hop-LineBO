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
#sys.path.append('./../Line-BO/HPER')

from bo_gpy_dyes import calc_3d_from_ternary

# Build analytical model
def ackley(x, b=0.5, a=20, c=2*np.pi, limit=15):
    """
    x: vector of input values
    """
    x = (x-0.3).T * limit
    d = len(x) # dimension of input vector x
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x*x) / d))
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

def dye(x, assume_ternary = False):
    
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
    y = np.ravel(y)
    
    return y
    

####################



####################



# Construct dataset from model
N = 3 # 5D Ackley function
n_init = 2
model_type = 'dye'
plot = True

X_init = pd.DataFrame(data = np.random.rand(n_init, N))#np.diversipy.polytope.sample(n_points=100000, lower=lower, upper=upper, thin=0)

if model_type == 'poisson':
    
    poisson_model = joblib.load(os.getcwd()+'/../HPER/data/poisson_RF_trained.pkl')
    poisson_model = poisson_model.predict # call the prediction function
    Y_init = pd.DataFrame(data = poisson_model(X_init), columns = ['y']) # fX dataset
    Y_experimental = poisson_model

elif model_type == 'ackley':

    Y_init = pd.DataFrame(data = (-1 * ackley(X_init.values)), columns = ['y']) # fX dataset
    Y_experimental = ackley

elif model_type == 'dye':
    
    dye_model = joblib.load(os.getcwd()+'/./data/3D-6-final-GP-model')
    Y_init = pd.DataFrame(data = dye(X_init), columns = ['y']) # fX dataset
    Y_experimental = dye
    
#Y_init = pd.DataFrame(poisson_model(np.array(X_init)), columns = Y.columns.values)


seed = np.random.seed(1) # seed for repeatability

# Run ZoMBI
zombi = ZombiHop(seed = seed,                       # A random seed for model reproducibility, IMPORTANT: *change every time a new independent trial is run*
                 X_init = X_init,                   # X data
                 Y_init = Y_init,                   # fX data
                 Y_experimental = Y_experimental,   # model to predict f(X) from X
                 Gammas = 3,                        # Number of hops to other needles
                 alphas = 10,                       #(X/N)# Number of ZoMBI zoom-ins for each hop, zooming repeatedly in
                 n_draws_per_activation = 10,       #X / 10 # Number of samples drawn for each zoom-in
                 acquisition_type = LCB_ada,        # acquisition function options: LCB, EI, LCB_ada, EI_abrupt
                 tolerance = 0.05,                  # Increase tolerance! # Error tolerance of GP prediction, used to end a ZoMBI zoom-in and move to the next needle
                 penalty_width = 0.15,               # Width of penalty region about needle => inhibits BO searching from areas surrounding previously found needles
                 m = 10,                             # Top m-number of data points used to zoom in bounds
                 k = 10,                            # Top k-number of data points to keep
                 lower_bound = np.zeros(N),
                 upper_bound = np.ones(N),#np.array([1,np.sqrt(3)/2]),
                 resolution = 20,                   # Number for the resolution of the mesh search space, e.g., resolution=10
                 sampler = line_bo_sampler)#

X_all, Y_all, needle_locs, needles = zombi.run_virtual(verbose = False, plot = plot)

X_all_3d = calc_3d_from_ternary(X_all.values)
needle_locs_3d = calc_3d_from_ternary(needle_locs.values)

if plot == True:
    
    plt.figure(figsize=(10,4))
    for n in range(needles.shape[0]):
        plt.axhline(needles.iloc[n,0], c='r', ls='-', lw=0.5, alpha=1, label='Discovered Needles')
    plt.scatter(Y_all.index, Y_all, c='k')
    plt.ylabel('Poission\'s Ratio')
    plt.xlabel('Number of Experiments')
    plt.grid()
    plt.legend(['Discovered Needles'])
    plt.show()
    
    
    plt.figure(figsize=(10,4))
    for n in range(needles.shape[0]):
        plt.axhline(needles.iloc[n,0], c='r', ls='-', lw=0.5, alpha=1, label='Discovered Needles')
    plt.plot(Y_all, c='k')
    plt.ylabel('Poission\'s Ratio')
    plt.xlabel('Number of Experiments')
    plt.grid()
    plt.legend(['Discovered Needles'])
    plt.show()
    
    plt.figure(figsize=(10,4))
    for n in range(needles.shape[0]):
        plt.axhline(needles.iloc[n,0], c='r', ls='-', lw=0.5, alpha=1, label='Discovered Needles')
    plt.scatter(Y_all.index, Y_all, c='k')
    plt.ylim(-3,1)
    plt.ylabel('Poission\'s Ratio')
    plt.xlabel('Number of Experiments')
    plt.grid()
    plt.legend(['Discovered Needles'])
    plt.show()
    
    plt.figure(figsize=(10,4))
    for n in range(needles.shape[0]):
        plt.axhline(needles.iloc[n,0], c='r', ls='-', lw=0.5, alpha=1, label='Discovered Needles')
    plt.plot(Y_all, c='k')
    plt.ylim(-3,1)
    plt.ylabel('Poission\'s Ratio')
    plt.xlabel('Number of Experiments')
    plt.grid()
    plt.legend(['Discovered Needles'])
    plt.show()
