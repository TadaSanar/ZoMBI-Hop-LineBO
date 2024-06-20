#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:56:25 2024

@author: atiihone
"""


import numpy as np
import n_sphere
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import lstsq #, solve
import warnings

from linebo_wrappers import get_acq, define_acq_object

def test_constraint(x, upper_lim = 1, lower_lim = 0.995):
    
    xsum = np.sum(x, axis = 1)
    
    if (xsum >= lower_lim).all() and (xsum <= upper_lim).all():
    
        result = True
    
    else:
    
        result = False
        
    return result

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
        
        # Invert A and solve.
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
                   plot_triangle = True, lims = [-2,2], show_p_coord = False,
                   show_a_coord = False):
    
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=60)
    
    
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('x2')
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)
    
    # Plot all points A (or B):
    for i in range(a.shape[0]):
        
        # Plot a line between point P and (P+A). 
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
        
        ax.plot(np.ravel([p[0,0], 0]), 
                np.ravel([p[0,1], 0]),
                np.ravel([p[0,2], -2]), c='k',
                linewidth = 0.5, linestyle = '--')
    label = 'P'
    if show_p_coord is True:
        label = label + ' ' + str(p)
    ax.scatter(p[:,0], p[:,1], p[:,2], c = 'k', label=label)
    
    label = first_point_label
    if show_a_coord is True:
        label = label + ' ' + str(p+a)
    ax.scatter(p[:,0] + a[:,0], p[:, 1] + a[:,1], p[:,2] + a[:,2], c = 'b', 
               label = label)
    #ax.scatter(a2[:,0], a2[:,1], a2[:,2], c = 'r')
    
    
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
    in the parameterized form of the line: x = p + k * tA (or x = p + k * tB, 
    respectively).

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
    
    # Solve t_min values for each point K_cand. Ordering: Point 0 dim 0, 
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
    if (np.isnan(tAcands).all() is True) or (np.isnan(tBcands).all() is True):
        
        raise Exception('Something wrong with defining A or B candidates!')
        
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
            
    '''
    if (N == 3) and (plotting == True):
        
        if constrain_sum_x == False:
            
            plot_triangle = False
            
        else:
            
            plot_triangle = True
            
            lims = [-2,2]
            
        
        plot_K_P_in_3D(K_cand, p, 
                       'Real 3D space\n(K_candidates outside\nthe search space are ok)', 
                       first_point_label = 'K_cand', 
                       plot_triangle = plot_triangle)
        plot_K_P_in_3D(K_cand-p, p-p, 
                       'Coordinate transfrom\nso that P in origo', 
                       first_point_label = 'K_cand', plot_triangle = False)
        plot_K_P_in_3D(a-p, p, 
                       'Real 3D space\n(A candidates outside\nthe search space are ok)', 
                       first_point_label = 'A candidates',
                       plot_triangle = plot_triangle, 
                       lims = [emin[0,0], emax[0,1]])
        plot_K_P_in_3D(b-p, p, 
                       'Real 3D space\n(B candidates outside\nthe search space are ok)', 
                       first_point_label = 'B candidates',
                       plot_triangle = plot_triangle,
                       lims = [emin[0,0], emax[0,1]])
        #print('P: ', p)
        #print('K: ', K_cand-p)
    '''    
    
    # If any of the points are outside the search space at this point, it means
    # P is on the boundary of the search space and, additionally, there is no t
    # value that could make the line candidate get inside the search space. I.e.,
    # P is in these cases always in the corner, I think.
    # Let's set t values for those points to zero (per the algo above, "the other
    # point A or B" is already zero, so in practice the line length goes to zero).
    if (np.any(np.any((a > emax), axis = 1)) or np.any(np.any((a < emin), axis = 1)) or 
        np.any(np.any((b > emax), axis = 1)) or np.any(np.any((b < emin), axis = 1))):
        
        tA[np.any((a > emax), axis = 1)] = 0
        tA[np.any((a < emin), axis = 1)] = 0
        tB[np.any((b > emax), axis = 1)] = 0
        tB[np.any((b < emin), axis = 1)] = 0
        
        # Actual points A and B for each point K.
        a = p + tA * K_cand
        b = p + tB * K_cand
    
    if (N == 3) and (plotting == True):
        
        if constrain_sum_x == False:
            
            plot_triangle = False
            
        else:
            
            plot_triangle = True
            
            lims = [-2,2]
            
        
        plot_K_P_in_3D(K_cand, p, 
                       'Real 3D space\n(K_candidates outside\nthe search space are ok)', 
                       first_point_label = 'K_cand', 
                       plot_triangle = plot_triangle)
        plot_K_P_in_3D(K_cand-p, p-p, 
                       'Coordinate transfrom\nso that P in origo', 
                       first_point_label = 'K_cand', plot_triangle = False)
        plot_K_P_in_3D(a-p, p, 
                       'Real 3D space\n(A candidates outside\nthe search space not ok)', 
                       first_point_label = 'A candidates',
                       plot_triangle = plot_triangle, 
                       lims = [emin[0,0], emax[0,1]])
        plot_K_P_in_3D(b-p, p, 
                       'Real 3D space\n(B candidates outside\nthe search space not ok)', 
                       first_point_label = 'B candidates',
                       plot_triangle = plot_triangle,
                       lims = [emin[0,0], emax[0,1]])

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
    
    acq_values = get_acq(x_steps, acq_object = acq_object, 
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
        
        acq_values = get_acq(x_steps, acq_object = acq_object, 
                                    acq_params = acq_params)
        acq_values_all[[i],:] = acq_values.T
    
    # In GPyOpt, the acquisition values are negative but could be otherwise
    # in other packages.
    
    # Caused issues because you cannot know the true range of vals and you might
    # end up having a different scaling btw points:
    # Normalize the acquisition values to [0,1] to make it easier to swap BO
    # packages.
    
    if acq_max is False:
        acq_values_all = -acq_values_all
    
    acq_min_val = np.min(acq_values_all)
    acq_max_val = np.max(acq_values_all)

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
             plotting = 'plot_all', acq_params = None):
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
    acq_max : boolean, optional
        Set True if acquisition function is defined so that its maximum is the
        optimum. The default is True.
    selection_method : str, optional
        Line selection method. 'integrate_acq_line' integrates over the
        acquisition function values. 'random_line' picks the line randomly
        among the candidates defined py points K_cand and point p. The default 
        is 'integrate_acq_line'.
    
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
    if plotting == 'plot_all':
        
        plotting_cands = True
        
    else:
        
        plotting_cands = False
        
    A, B, tA, tB = extract_inlet_outlet_points(p, K_cand = K_cand, emax = emax,
                                       emin = emin, M = M,
                                       constrain_sum_x = constrain_sum_x,
                                       plotting = plotting_cands)
    #print('K candidates:\n', K)
    #print('A candidates:\n', A)
    
    if selection_method == 'integrate_acq_line':
        
        I_all = np.empty((K_cand.shape[0],1)) 
        
        #for i in range(K_cand.shape[0]):
        #    
        #    I_all[i] = integrate_over_acqf(p, K_cand[[i],:], tA[i,:], tB[i,:], 
        #                                   500, acq_object = acq_function, 
        #                                   acq_max = acq_max, acq_params = BO_object)
        
        acq_object = define_acq_object(BO_object, acq_params = acq_params)
        
        I_all = integrate_all_K_over_acqf(p, K_cand, tA, tB, 500, 
                                  acq_object = acq_object, 
                                  acq_max = acq_max, acq_params = acq_params)
        
        if np.sum(np.abs(I_all)) == 0:
            
            raise Exception("Something wrong with the acquisition function values or A/B candidates!")
        
        idx = np.argmax(I_all, axis = 0)
        print('Mean value of integrals along lines: ', np.mean(I_all), ' +- ', np.std(I_all), 
              '\nChosen value of the integral and its index: ', I_all[idx], idx,
              #'\nAll values of the integrals that were compared: ', I_all
              )
        
    elif selection_method == 'random_line':
        
        # Note: This method can pick also lines of length zero (i.e., A=P and
        # B=P). Think if this is ok or not.
        idx = np.random.randint(0, A.shape[0])
        
    else:
        
        raise Exception("Not implemented.")
        
    A_sel = A[idx, :]
    B_sel = B[idx, :]
    tA_sel = tA[idx]
    tB_sel = tB[idx]
    K_sel = K_cand[idx, :]
    
    if ((A_sel < emin).any() or (A_sel > emax).any() or 
        (B_sel < emin).any() or (B_sel > emax).any()) == True:
        
        message = ("The selected A or B seem to be outside the current search" +
                   " space boundary. The most common for this to happen is that " +
                   "the BO-suggested point P is outside the search space " +
                   "boundaries, which then propagates to A or B. The current " +
                   "points and values are:\n" +
                   "- emin: " + str(emin) + "\n" +
                   "- emax: " + str(emax) + "\n" +
                   "- P: " + str(p) + "\n" +
                   "- A: " + str(A_sel) + "\n" +
                   "- B: " + str(B_sel) + "\n")
        
        raise Exception(message)
    
    # Plot the selected A and B if the dimensionality of the search space is three.
    if (p.shape[1] == 3) and (plotting != 'plot_none'):
    
        plot_K_P_in_3D(A_sel-p, p, 'Real 3D space\n(A selected outside\nthe search space not ok)', 
                       first_point_label = 'A selected',
                       plot_triangle = constrain_sum_x, 
                       lims = [emin[0,0], emax[0,1]],
                       show_a_coord = True,
                       show_p_coord = True)
        plot_K_P_in_3D(B_sel-p, p, 'Real 3D space\n(B selected outside\nthe search space not ok)',
                       first_point_label = 'B selected',
                   plot_triangle = constrain_sum_x, 
                   lims = [emin[0,0], emax[0,1]],
                   show_a_coord = True,
                   show_p_coord = True)
    
    
    return A_sel, B_sel, tA_sel, tB_sel, K_sel

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
                constrain_sum_x = constrain_sum_x, plotting = plotting,
                acq_object = None)
            
        else:
            
            A_sel[i,:] = np.nan
            B_sel[i,:] = np.nan
            tA_sel[i] = np.nan
            tB_sel[i] = np.nan
            K_sel[i,:] = np.nan
        
    return p, A_sel, B_sel, tA_sel, tB_sel, K_sel

def compute_x_coords_along_lines(idx_to_compute, N, n_droplets, tA_sel, tB_sel, 
                               p_sel, K_sel):
    
    # NOTE: Armi edited the shapes of these funs in 6/19/2024. Confirm the
    # edits did not affect Line-BO repo.
    
    # idx_to_compute is a list of indices.
    x_steps = np.zeros((len(idx_to_compute), n_droplets, N))
    
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
            x_steps[k,:,:] = p_sel[idx_to_compute[k],:] + t_steps * K_sel[#[idx_to_compute[k]],
                                                                          :]
            
            # Reshape for the computation purposes.
            x = np.reshape(x_steps, (x_steps.shape[0]*x_steps.shape[1], x_steps.shape[2]))
    
    return x