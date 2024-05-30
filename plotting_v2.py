#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
MIT Photovoltaics Laboratory
"""

import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
import datetime

#script_dir = os.path.dirname(__file__)
#results_dir = os.path.join(script_dir, 'Results/')
#
#if not os.path.isdir(results_dir):
#    os.makedirs(results_dir)


def triangleplot(surf_points, surf_data, norm, surf_axis_scale = 1, cmap = 'RdBu_r',
                 cbar_label = '', saveas = None, surf_levels = None,
                 scatter_points=None, scatter_color = None, cbar_spacing = None,
                 cbar_ticks = None):


    mpl.rcParams.update({'font.size': 8})
    mpl.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})
    
    
    b=surf_points[:,0]
    c=surf_points[:,1]
    a=surf_points[:,2]
    # values stored in the last column
    v = np.squeeze(surf_data)/surf_axis_scale#[:,-1]/surf_axis_scale
    # translate the data to cartesian
    x = 0.5 * ( 2.*b+c ) / ( a+b+c )
    y = 0.5*np.sqrt(3) * c / (a+b+c)
    # create a triangulation
    T = tri.Triangulation(x,y)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(3.6/2.54)
    fig.set_figwidth(5/2.54)
    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.14, top=0.91)

    if scatter_points is not None:
        #Triangulation for the points suggested by BO.
        b_p = scatter_points[:,0]
        c_p = scatter_points[:,1]
        a_p = scatter_points[:,2]
        x_p = 0.5 * ( 2.*b_p+c_p ) / ( a_p+b_p+c_p)
        y_p = 0.5*np.sqrt(3) * c_p / (a_p+b_p+c_p)
        
        im3 = ax.scatter(x_p, y_p, s=8, c=scatter_color, cmap=cmap, edgecolors='black', linewidths=.5, alpha=1, zorder=2, norm=norm)
    # plot the contour
    if surf_levels is None:
        
        nlevels = 40
        minvalue = 0
        
        if norm.vmin < minvalue:
            minvalue = norm.vmin-(np.abs(norm.vmin)/100)
            
        if norm.vmin > (minvalue + ((norm.vmax-minvalue)/nlevels)):
            minvalue = norm.vmin-(np.abs(norm.vmin)/100)
            
        surf_levels = np.arange(minvalue, norm.vmax+(np.abs(norm.vmax)/100), (norm.vmax-minvalue)/nlevels)
        if norm.vmax != 0:
            surf_levels = np.round(surf_levels, -int(np.floor(np.log10(norm.vmax))-1))#2))
            
        else:
            surf_levels = np.round(surf_levels, 1)
            
        surf_levels = np.unique(np.append(surf_levels, 2*surf_levels[-1] - surf_levels[-2]))
        
        if surf_levels[0] > minvalue:
            surf_levels = np.insert(surf_levels, 0, surf_levels[0] - (surf_levels[1] - surf_levels[0]))
    
    if np.std(surf_levels) == 0: # The above automatic solution does not work for <1 values, resulting in constant surf levels. In these cases, it is best to fall back to automatic.
        surf_levels = np.arange(norm.vmin, norm.vmax, (norm.vmax-norm.vmin)/nlevels)
        
    im=ax.tricontourf(x,y,T.triangles,v, cmap=cmap, levels=surf_levels)
    
    myformatter=matplotlib.ticker.ScalarFormatter()
    myformatter.set_powerlimits((0,2))
    if cbar_spacing is not None:
        cbar=plt.colorbar(im, ax=ax, spacing=cbar_spacing, ticks=cbar_ticks)
    else:
        cbar=plt.colorbar(im, ax=ax, format=myformatter, spacing=surf_levels, ticks=surf_levels[1::4])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_label, fontsize=8)#, labelpad = -0.5)
    plt.axis('off')
# =============================================================================
#     Values used in C2a:
#     plt.text(0.35,-0.1,'Cs (%)')
#     plt.text(0.10,0.54,'FA (%)', rotation=61)
#     plt.text(0.71,0.51,'MA (%)', rotation=-61)
#     plt.text(-0.0, -0.1, '0')
#     plt.text(0.87, -0.1, '100')
#     plt.text(-0.07, 0.13, '100', rotation=61)
#     plt.text(0.39, 0.83, '0', rotation=61)
#     plt.text(0.96, 0.05, '0', rotation=-61)
#     plt.text(0.52, 0.82, '100', rotation=-61)
#     
# =============================================================================
    plt.text(0.35,-0.1,'Cs (%)')
    plt.text(0.09,0.34,'FA (%)', rotation=61)
    plt.text(0.71,0.31,'MA (%)', rotation=-61)
    plt.text(-0.0, -0.1, '0')
    plt.text(0.87, -0.1, '100')
    plt.text(-0.08, 0.05, '100', rotation=61)
    plt.text(0.38, 0.83, '0', rotation=61)
    plt.text(0.96, 0.05, '0', rotation=-61)
    plt.text(0.52, 0.75, '100', rotation=-61)
    
    # create the grid
    corners = np.array([[0, 0], [1, 0], [0.5,  np.sqrt(3)*0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    # creating the grid
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=0)
    #plotting the mesh
    im2=ax.triplot(trimesh,'k-', linewidth=0.5)
    
    plt.tight_layout()
    
    if saveas:
        fig.savefig(saveas + '.pdf', transparent = True)
        #fig.savefig(saveas + '.svg', transparent = True)
        #fig.savefig(saveas + '.png', dpi=300)
    plt.show()
    return fig, ax

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    ## TO DO: This is a modified version of an open access function. Deal with ack.
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), #128, endpoint=False), 
        np.linspace(midpoint, 0.3, 80, endpoint=False),
        np.linspace(0.3, 1.0, 49, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def create_ternary_grid(range_min=0, range_max=1, interval=0.005):
    
    a = np.arange(range_min, range_max, interval)
    xt, yt, zt = np.meshgrid(a,a,a, sparse=False)
    points = np.transpose([xt.ravel(), yt.ravel(), zt.ravel()])
    # The x, y, z coordinates need to sum up to 1 in a ternary grid.
    points = points[abs(np.sum(points, axis=1)-1) < interval]
    
    return points    

def init_plots(rounds, limit_file_number, time_str, results_folder):
    
    # This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    points = create_ternary_grid()
    
    mean = [None for k in range(rounds)]
    std = [None for k in range(rounds)]
    acq = [None for k in range(rounds)]
    
    #original_folder = os.getcwd()
    #os.chdir(original_folder)
    if time_str is None:
        time_now = '{date:%Y%m%d%H%M}'.format(date=datetime.datetime.now())
    else:
        time_now = time_str
        
    results_dir = results_folder + 'Triangles/'+ time_now + '/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if (limit_file_number == True) and (rounds > 10):
        
        # Limit the number of plotting files to 5/BO cycle/plot type.
        # Additionally, only the most essential plots will be created.
        
        rounds_to_plot = [0, 2, 10, int(np.floor(rounds/2)), (rounds-1)]
    
    else:
        
        rounds_to_plot = range(rounds)
    
    return points, mean, std, acq, time_now, results_dir, rounds_to_plot

def fill_ternary_grid(mean, std, GP_model, points, y_train_data = None):
    
    # : Here the posterior mean and std_dv+acquisition function are calculated.
    mean, posterior_var = GP_model.predict(points) # MAKING THE PREDICTION, GPy GPregression model assumed
    
    if y_train_data is not None:
        
        # Scaling the normalized data back to the original units (GPyOpt BO scales the data before training the model).
        mean = mean*y_train_data.std()+y_train_data.mean()
        std = np.sqrt(posterior_var)*y_train_data.std()
    
    else:
        
        std = np.sqrt(posterior_var)
        
    return mean, std

def fill_ternary_grids(mean, std, acq, rounds, BO_batch, points, y_train_data = None):
    
    for i in range(rounds):
        
        if y_train_data is None:
            y_t = None
        else:
            y_t = y_train_data[i]
        
        # : Here the posterior mean and std_dv+acquisition function are calculated.
        mean[i], std[i] = fill_ternary_grid(mean[i], std[i], BO_batch[i].model.model, points, y_train_data = y_t)
        
        acq_i=BO_batch[i].acquisition.acquisition_function(points)
        # Scaling the acquisition function to btw 0 and 1.
        acq[i] = (-acq_i - min(-acq_i))/(max(-acq_i - min(-acq_i)))

    return mean, std, acq    
    
    
def save_round_to_csv_files(mean, std, acq, rounds, materials, points,
                            limit_file_number, results_dir, time_now,
                            x_data, y_data, next_suggestions = None,
                            mean_name = 'Ic (px*min)',
                            std_name = 'Std of Ic (px*min)', acq_name = 'EIC'):

    for i in range(rounds):

        if (limit_file_number == False) and (next_suggestions is not None):        
            next_suggestions[i].to_csv(results_dir + 'Bayesian_suggestion_round_'+str(i) + time_now + '.csv', float_format='%.3f')

        inputs = x_data[i]
        inputs[mean_name] = y_data[i]['Merit']
        inputs=inputs.sort_values(mean_name)
        inputs=inputs.drop(columns=['Unnamed: 0'], errors='ignore')

        if (limit_file_number == False):
            inputs.to_csv(results_dir + 'Model_inputs_round_'+str(i)+ time_now + '.csv', float_format='%.3f')
        
        inputs2 = pd.DataFrame(points, columns=materials)
        inputs2[mean_name]=mean[i]
        inputs2[std_name]=std[i]
        inputs2[acq_name]=acq[i]

        if (limit_file_number == False):
            inputs2.to_csv(results_dir + 'Model_round_' + str(i) + time_now + '.csv', float_format='%.3f')

def define_norm_for_surf_plot(target, color_lims = None):
    
    if color_lims is not None:
        lims = color_lims
    else:
        lims = [np.min(target), np.max(target)]
        
    if (lims[0] == 0) and (lims[1] == 0): # There's only one datapoint that is zero.
            
        lims[0] = -0.5
        lims[1] = 0.5
        
    if lims[0] == lims[1]: # There's only one datapoint that is nonzero.
            
        lims[0] = lims[0] - np.abs(lims[0]/2)
        lims[1] = lims[1] + np.abs(lims[1]/2)
    
    norm = matplotlib.colors.Normalize(vmin=lims[0], vmax=lims[1])
    
    return norm
    
def plot_surf_with_lims_and_name(points, target, color_lims, cmap, cbar_label,
                                     saveas):
    
    norm = define_norm_for_surf_plot(target, color_lims = color_lims)
    
    triangleplot(points, target, norm, cmap = cmap,
                 cbar_label = cbar_label, saveas = saveas)

    
def plot_mean_only(points, mean, color_lims = None, cmap = 'RdBu_r',
                   cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h)', saveas = None):
                
    plot_surf_with_lims_and_name(points, mean,
                                     color_lims = color_lims, cmap = cmap,
                                     cbar_label = cbar_label, saveas = saveas)

def plot_std_only(points, std, color_lims = None, cmap = 'RdBu_r',
                   cbar_label = r'Std $I_{c}(\theta)$ (px$\cdot$h)', saveas = None):
                
    plot_surf_with_lims_and_name(points, std,
                                     color_lims = color_lims, cmap = cmap,
                                     cbar_label = cbar_label, saveas = saveas)

def init_acq_plot(points, acq, cmap_base):

    if cmap_base == 'RdBu':
    	
    	orig_cmap = mpl.cm.RdBu
    	# Shift the colormap (in order to remove the red background that appears with the standard version of the colormap).
    	shifted_cmap = shiftedColorMap(orig_cmap, start=0, midpoint=0.000005, stop=1, name='shifted')
    	cmap_name = 'shifted'
    else:
    	
    	# Use user defined colormap and do nothing else.
    	cmap_name = cmap_base
    	
    if acq is None:
    	
    	# Plot the starting point of the acquisitions i.e. an even distribution.
        acq_to_plot = np.ones((points.shape[0], 1))*0.3
    
    else:
        acq_to_plot = acq
        
        ## Old version:
        #if i == 0:
        #    # In the first round there is an even distribution.
        #    test_data = np.concatenate((points, acq_normalized[i]), axis=1)
        #    test_data = test_data[test_data[:,3] != 0]
        #    test_data[:,3] = np.ones(test_data[:,3].shape)*0.3
        #else:
        #    test_data = np.concatenate((points, acq_normalized[i-1]), axis=1)
        #    test_data = test_data[test_data[:,3] != 0]
    	
    # Acquisition function may have a lot of zero values. The plot looks
    # pretty weird unless they are removed.
    all_data = np.concatenate((points, acq_to_plot), axis = 1)
    all_data = all_data[all_data[:,3] != 0]
    acq_to_plot = all_data[:,3]
    points_to_plot = all_data[:,0:3] 
    
    plot_params = {
        'surf_axis_scale': 1.0,
        'cmap_name': cmap_name,
        'surf_levels': (0,0.009,0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1),
        'cbar_spacing': 'proportional',
        'cbar_ticks': (0,0.2,0.4,0.6,0.8,1)
        }
    
    return points_to_plot, acq_to_plot, plot_params
    
def plot_acq_only(points, acq, color_lims = None, cmap_base = 'RdBu',
                   cbar_label = r'Acq(\theta)', saveas = None):
    
    # Acq. colorbar label in C2a:
    # r'$EIC (\theta, \beta_{DFT})$'
    
    # Acq.fun. plot looks weird unless zeros are removed and colormap shifted. 
    points_to_plot, acq_to_plot, plot_params = init_acq_plot(points, acq, cmap_base)
    
    # Let's not use the convenience function 'plot_surf_with_lims_and_name'
    # because we change so many parts. Repeat its functionality, instead.
    
    # Note that color lims does not include zeros (since they are removed at
    # two lines above this point) unless the user feeds in color_lims with zero
    # included.
    norm = define_norm_for_surf_plot(acq_to_plot, color_lims = color_lims)
    
    triangleplot(points_to_plot, acq_to_plot, norm,
                 surf_axis_scale = plot_params['surf_axis_scale'],
                 cmap = plot_params['cmap_name'],
                 cbar_label = cbar_label,
                 saveas = saveas, 
                 surf_levels = plot_params['surf_levels'],
                 cbar_spacing = plot_params['cbar_spacing'],
                 cbar_ticks = plot_params['cbar_ticks'])
    
def plot_acq_and_data(points, acq, data, color_lims = None, cmap_base = 'RdBu',
                   cbar_label = r'Acq(\theta)', saveas = None):
    
    
    # Acq.fun. plot looks weird unless zeros are removed and colormap shifted. 
    points_to_plot, acq_to_plot, plot_params = init_acq_plot(points, acq, cmap_base)
    
    # Let's not use the convenience function 'plot_surf_with_lims_and_name'
    # because we change so many parts. Code again, instead.
    
    # Note that color lims does not include zeros (since they are removed at
    # two lines above this point) unless the user feeds in color_lims with zero
    # included.
    norm = define_norm_for_surf_plot(acq_to_plot, color_lims = color_lims)
    
    # Colors of the samples.     
    # Old version in C2a:
    #newPal = {} #0:'k', 1:'k', 2:'k', 3:'k', 4: 'k'}#{0:'#8b0000', 1:'#7fcdbb', 2:'#9acd32', 3:'#ff4500', 4: 'k'} #A14
    #for i in rounds_to_plot:
    #    newPal[i] = 'k'
    
    triangleplot(points_to_plot, acq_to_plot, norm,
                 surf_axis_scale = plot_params['surf_axis_scale'],
                 cmap = plot_params['cmap_name'],
                 cbar_label = cbar_label,
                 saveas = saveas, 
                 surf_levels = plot_params['surf_levels'],
                 scatter_points=data, scatter_color = 'k',
                 cbar_spacing = plot_params['cbar_spacing'],
                 cbar_ticks = plot_params['cbar_ticks'])
    
def plot_mean_and_data(points, mean, data_x, data_y, color_lims = None, cmap = 'RdBu_r',
                   cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h)', saveas = None):
                
    # Let's not use the convenience function 'plot_surf_with_lims_and_name'
    # because we change so many parts. Repeat its functionality, instead.
    
    norm = define_norm_for_surf_plot(mean, color_lims = color_lims)
    
    triangleplot(points, mean, norm, cmap = cmap,
                 cbar_label = cbar_label, saveas = saveas,
                 scatter_points = data_x, scatter_color = np.ravel(data_y),
                 cbar_spacing = None, cbar_ticks = None)

def plotBO(rounds, suggestion_df, compositions_input, degradation_input,
           BO_batch, materials, X_rounds, x_next, Y_step, X_step,
           limit_file_number = True, time_str = None, results_folder = './Results/'):
    
    # Create a ternary grid 'points', the necessary folder structure, and 
    # file name templates. Initialize the posterior mean and st.dev., and
    # acquisition function lists.
    points, posterior_mean, posterior_std, acq_normalized, time_now, results_dir, rounds_to_plot = init_plots(
        rounds, limit_file_number, time_str, results_folder)
    
    # Fill in the lists with surfaces to plot.
    posterior_mean, posterior_std, acq_normalized = fill_ternary_grids(posterior_mean, 
                                                                       posterior_std, 
                                                                       acq_normalized, 
                                                                       rounds, 
                                                                       BO_batch, 
                                                                       points, 
                                                                       y_train_data = Y_step)
    
    ###############################################################################
    # SAVE RESULTS TO CSV FILES
    ###############################################################################
    
    # Save csvs only if limit_file_number is False.
    save_round_to_csv_files(posterior_mean, posterior_std, acq_normalized,
                            rounds, materials, points,
                                limit_file_number, results_dir, time_now,
                                x_data = compositions_input,
                                y_data = degradation_input,
                                next_suggestions = suggestion_df)
            
    ###############################################################################
    # PLOT
    ###############################################################################
    # Let's plot the resulting suggestions. This plot works for 3 materials only.
    
    # Min and max values for each contour plot are determined and normalization
    # of the color range is calculated.
    axis_scale = 60 # Units from px*min to px*hour.
    lims_p = [np.min(posterior_mean)/axis_scale, np.max(posterior_mean)/axis_scale]
    lims_s = [np.min(posterior_std)/axis_scale, np.max(posterior_std)/axis_scale]
    lims_a = [0,1]#[np.min(acq_normalized), np.max(acq_normalized)]
    
    for i in rounds_to_plot:
        
        # Plot acquisition function at the beginning of the round with samples
        # suggested for the round.
        
        if i == 0:
            acq_to_plot = None # At the beginning, we have an even distribution.
        else:
            acq_to_plot = acq_normalized[i-1]
            # Note i-1, this is because acq is saved at the end of each round
            # and here we plot acq at the beginning of each round.
        
        plot_acq_and_data(points, acq_to_plot, X_rounds[i], color_lims = lims_a,
                          cbar_label = r'$Acq(\theta)$ in round ' + str(i),
                          saveas = results_dir +
                          'Acq-with-single-round-samples-round'+str(i) + 
                          '-' + time_now)
        
        # Plot posterior mean with samples acquired by the end of the round.
        plot_mean_and_data(points, posterior_mean[i]/axis_scale,
                           X_step[i], Y_step[i]/axis_scale,
                           color_lims = lims_p,
                           cbar_label =
                           r'$I_{c}(\theta)$ (px$\cdot$h) in round ' + str(i),
                           saveas = results_dir +
                           'Modelled-Ic-with-samples-round' + str(i) + 
                           '-' + time_now)
        
        
        if (limit_file_number == False):        
            
            plot_mean_only(points, posterior_mean[i]/axis_scale,
                           color_lims = lims_p,
                           saveas = results_dir + 'Modelled-Ic-no-grid-round' +
                           str(i) + '-' + time_now)
            
            plot_std_only(points, posterior_std[i]/axis_scale,
                          color_lims = lims_s,
                          saveas = results_dir + 'St-Dev-of-modelled-Ic-round' +
                          str(i) + '-' + time_now)
            
            plot_acq_only(points, acq_to_plot,
                                      color_lims = lims_a,
                                      saveas = results_dir + 'Acq-round'+str(i) + 
                                     '-' + time_now)
            
    plt.close('all')
    
    '''
    norm = matplotlib.colors.Normalize(vmin=lims[2][0], vmax=lims[2][1])
    # Shift the colormap (removes the red background that appears with the std colormap)
    orig_cmap = mpl.cm.RdBu
    shifted_cmap = shiftedColorMap(orig_cmap, start=0, midpoint=0.000005, stop=1, name='shifted')
    # Colors of the samples in each round.     
    newPal = {}#0:'k', 1:'k', 2:'k', 3:'k', 4: 'k'}#{0:'#8b0000', 1:'#7fcdbb', 2:'#9acd32', 3:'#ff4500', 4: 'k'} #A14
    for i in rounds_to_plot:
        newPal[i] = 'k'
        print('Plotting round ', i) #A14
        if i==0:
            # In the first round there is an even distribution.
            test_data = np.concatenate((points, acq_normalized[i]), axis=1)
            test_data = test_data[test_data[:,3] != 0]
            test_data[:,3] = np.ones(test_data[:,3].shape)*0.3
        else:
            test_data = np.concatenate((points, acq_normalized[i-1]), axis=1)
            test_data = test_data[test_data[:,3] != 0]
        triangleplot(test_data[:,0:3], test_data[:,3], norm,
                     surf_axis_scale = 1.0, cmap = 'shifted',
                     cbar_label = r'$EIC(\theta, \beta_{DFT})$ in round ' + str(i), #A14
                     saveas = results_dir + 'EIC-with-single-round-samples-round'+str(i) + 
                     '-' + time_now, #A14
                     surf_levels = (0,0.009,0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8,1),
                     scatter_points=X_rounds[i], scatter_color = newPal[i],
                     cbar_spacing = 'proportional',
                     cbar_ticks = (0,0.2,0.4,0.6,0.8,1))
        
        # Plot acquisition function without samples only if file number does not have to be limited.
    if (limit_file_number == False):        
        for i in rounds_to_plot:
            #print('Round: ', i) #A14
            if i==0:
                # In the first round there is a even distribution.
                test_data = np.concatenate((points, acq_normalized[i]), axis=1)
                test_data = test_data[test_data[:,3] != 0]
                test_data[:,3] = np.ones(test_data[:,3].shape)*0.3
            else:
                test_data = np.concatenate((points, acq_normalized[i-1]), axis=1)
                test_data = test_data[test_data[:,3] != 0]
            triangleplot(test_data[:,0:3], test_data[:,3], norm,
                         surf_axis_scale = 1.0, cmap = 'shifted',
                         cbar_label = r'$EIC(\theta, \beta_{DFT})$', #A14
                         saveas = results_dir + 'EIC-round'+str(i) + 
                         '-' + time_now, #A14
                         surf_levels = (0,0.009,0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8,1),
                         cbar_spacing = 'proportional',
                         cbar_ticks = (0,0.2,0.4,0.6,0.8,1))
    
    #A14
    norm = matplotlib.colors.Normalize(vmin=lims[0][0], vmax=lims[0][1])
    sample_points = np.empty((0,3))
    for i in rounds_to_plot:
        triangleplot(points, posterior_mean[i]/axis_scale, norm,
                     cmap = 'RdBu_r',
                     cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h) in round ' + str(i), #A14
                     saveas = results_dir + 'Modelled-Ic-with-samples-round'+str(i) + 
                     '-' + time_now, #A14
                     scatter_points=X_step[i],
                     scatter_color = np.ravel(Y_step[i]/axis_scale),
                     cbar_spacing = None, cbar_ticks = None)
        '''