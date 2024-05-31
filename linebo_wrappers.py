#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:13:57 2024

@author: atiihone
"""

import numpy as np
import GPy
import GPyOpt
#import zombihop


from linebo_util import nearest_x

def define_bo(x_all, y_all, emin, emax, N, task_max = True, batch_size = 1,
              constrain_sum_x = False, implementWithBOPackage = 'GpyOpt'):
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
    if implementWithBOPackage == 'GpyOpt':
        
        # Set up BO with GPyOpt BO package.
        
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
        
    else:
        
        Exception('define_bo() has not been implemented for BO package ' + 
                  implementWithBOPackage + '. Request another BO package.')
    
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
    
    if type(BO_object) is GPyOpt.methods.BayesianOptimization:
        
        p_next = BO_object.suggest_next_locations()
        
    else:
        
        p_next = None
        Exception('This function has not been implemented for this type of BO.')
    
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
    
    if type(BO_object) is GPyOpt.methods.BayesianOptimization:
        
        gpmodel = BO_object.model.model
        
        if type(gpmodel) is GPy.models.gp_regression.GPRegression:
            
            # Prediction output is mean, variance.
            y, var = gpmodel.predict(x)
            std = np.sqrt(var)
            
        elif type(gpmodel) is GPyOpt.models.gpmodel.GPModel:
            
            # Prediction output is mean, standard deviation.
            y, std = gpmodel.predict(x)
            var = (std)**2
        
        y_train_unscaled = BO_object.Y
        
    else:
        
        Exception('This function has not been implemented for this type of BO.')
    
    if y_train_unscaled.shape[0] > 0:
        
        posterior_mean_true_units = y * np.std(y_train_unscaled) + \
            np.mean(y_train_unscaled)
        posterior_std_true_units = (np.std(y_train_unscaled)) * (std)
        
    if return_std is False:
        
        result = posterior_mean_true_units
        
    else:
        
        result = [posterior_mean_true_units, posterior_std_true_units]
    
    return result

def define_acq_object(BO_object, acq_params = None):#, x):
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
        The acquisition function value of x datapoint(s) predicted with BO_object.

    """
    if isinstance(BO_object, GPyOpt.methods.BayesianOptimization):
        
        # TO DO: Is there difference among the two options?
        #a = BO_object.acquisition._compute_acq(x)
        #a = BO_object.acquisition.acquisition_function(x)
        acq_object = BO_object.acquisition.acquisition_function
        
    elif (type(acq_params['acq_dictionary']) is dict): # && (type(BO_object) is [type of ZoMBI-HOP's BO object])
        
        acq_object = acq_params['acq_dictionary']
    
    else:#if (BO_object == acq_fun_zombihop):
        
        acq_object = BO_object # TO DO: Korjaa näytteistysfunktioksi! #A
    
    #else:
    #    
    #    acq_object = None
    #    Exception('This function has not been implemented for this type of BO.')
    
    return acq_object

def get_acq(x, acq_object, acq_params):

    if isinstance(acq_object, dict) is True:
        
        # Assume dictionary contains two numpy arrays that are the x and y
        # values of the acquisition function along the search space.
        
        # This is required for the default implementation of ZoMBI-Hop.
        x_acq = acq_object['x']
        idx_x_steps = nearest_x(x, x_acq).astype(int)
        acq_values = acq_object['y'][idx_x_steps]
   
    else:
        
        # Assume the object can predict acquisition values directly.
        acq_fun = acq_object
        if acq_params is not None:
            
            acq_values = acq_fun(x, acq_params)
        
        else:
            
            acq_values = acq_fun(x)
        
        
        
    return acq_values
    


def extract_model_opt_value(BO_object, x_all, task_max):
    
    y_all = predict_from_BO_object(BO_object, x_all)
    
    if task_max is True:
        
        idx_opt = np.argmax(y_all, axis = 0)
        
    else:
        
        idx_opt = np.argmin(y_all, axis = 0)
        
    y = y_all[idx_opt, :].copy()
    x = x_all[idx_opt, :].copy()
    
    return x, y

#def create_2d_grid(range_min=0, range_max=1, interval=0.01):
    
# OLD VERSION THAT IS BEING STORED FOR A WHILE.
#   
#    a = np.arange(range_min, range_max, interval)
#    xt, yt = np.meshgrid(a,a, sparse=False)
#    points = np.transpose([xt.ravel(), yt.ravel()])
#    ## The x, y, z coordinates need to sum up to 1 in a ternary grid.
#    #points = points[abs(np.sum(points, axis=1)-1) < interval]
#    
#    return points


###############################################################################
# Helper functions needed only within linebo_wrappers.py:

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

'''
def indicator_constraints_gpyopt(x, constraints):
    """
    DELETE? FUNCTION NOT BEING USED ANYWHERE?
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
'''

    
