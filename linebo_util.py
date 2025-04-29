#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:25:42 2024

@author: atiihone
"""

import numpy as np
import pandas as pd

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


def calc_regret(x_opt, y_opt, x_true, y_true = 0):
    
    regret_x = np.sum((x_opt - x_true)**2, axis = 1)
    regret_y = np.sum((y_opt - y_true)**2, axis = 1)
    
    return regret_x, regret_y


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
