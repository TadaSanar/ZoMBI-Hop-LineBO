#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:29:30 2024

@author: atiihone
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from plotting_v2 import triangleplot

#from linebo_fun import get_acq_values

from linebo_util import create_Nd_grid

#import linebo_main.create_Nd_grid
from linebo_wrappers import predict_from_BO_object, define_acq_object, get_acq
#import linebo_main.acq_from_BO_object
#import linebo_main.sample_y

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
                     x_opt_uncertainty = None, acq_params = None):
    
    if N == 2:
        BO_object.plot_acquisition()
    
    if N == 3:
        
        acq_object = define_acq_object(BO_object, acq_params = acq_params)
        
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
        
        surf_data = get_acq(surf_points, acq_object = acq_object, 
                                    acq_params = acq_params)
        
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

        y_plot_bo = get_acq(x_plot, acq_object = acq_object, 
                                    acq_params = acq_params) #acq_from_BO_object(BO_object, x_plot)
        
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

def plot_landscapes(BO_object, x_plot, y_plot_gt = None,
                    idx0 = 0, idx1 = 1):

    y_plot_bo = predict_from_BO_object(BO_object, x_plot)
    
    if y_plot_gt is not None:
        
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
