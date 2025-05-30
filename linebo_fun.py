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
from numpy.linalg import lstsq  # , solve
import warnings

from linebo_wrappers import get_acq, define_acq_object


def test_constraint(x, upper_lim=1, lower_lim=0.995):

    xsum = np.sum(x, axis=1)

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
    eigenvecs_base_idx = 0  # np.random.randint(0,N)
    eigenvecs_others_idx = list(range(N))
    eigenvecs_others_idx.remove(eigenvecs_base_idx)
    eigenvecs_subspace_simple = (eigenvecs[eigenvecs_others_idx, :] -
                                 eigenvecs[eigenvecs_base_idx, :])

    # So let's pick new eigenvecs that have pi/2 angles.
    eigenvecs_base_idx = 0  # np.random.randint(0,N)
    eigenvecs_others_idx = list(range(N-1))
    eigenvecs_others_idx.remove(eigenvecs_base_idx)
    eigenvecs_subspace_new = (-0.5*eigenvecs_subspace_simple[eigenvecs_base_idx, :] +
                              eigenvecs_subspace_simple)
    # And let's make them to have the same lengths.
    eigenvecs_subspace_new[eigenvecs_base_idx, :] = 4 * \
        eigenvecs_subspace_new[eigenvecs_base_idx, :]
    eigenvecs_subspace_new = (eigenvecs_subspace_new / (np.sqrt(np.sum(
        eigenvecs_subspace_new**2, axis=1))).reshape((N-1, 1)))

    dot_prod = np.dot(eigenvecs_subspace_new[0, :], eigenvecs_subspace_new.T)
    if ((not np.isclose(dot_prod[eigenvecs_base_idx], 1, atol=0.02)) or
            (np.any(~np.isclose(dot_prod[eigenvecs_others_idx], 0, atol=0.02)))):

        raise Exception('Eigenvectors are not orthogonal or scaling is wrong!')

    # Cartesian coordinates expressed in the true search space. Note that the
    # points are now "on the right plane" but do not necessarily fulfill the
    # conditions of sum(x_i)=1.
    cart_coords = np.matmul(cart_coords_low_dim, eigenvecs_subspace_new)

    #print('K candidates after transform to actual dimensionality: ', cart_coords)

    if np.isclose(np.sqrt(np.sum(cart_coords**2)), 1, atol=0.02) is False:

        raise Exception('K_cand are not scaled to one!')

    # Scaling of the values so that the points are actually within the subspace
    # boundaries is done just for clarity - for the purposes of this code the
    # previous step would already have sufficed.
    # cart_coords = cart_coords / np.reshape(np.sum(cart_coords, axis = 1), (N-1,1))

    return cart_coords

def calc_K(dimensionality, M, constrain_sum_x=False, plotting=False,
           generate_randomly=False, max_candidates=np.inf,
           random_generation_type='spherical', p=None):
    """
    Optimized candidate‐direction generator.
    If M**(N-1) <= max_candidates, fall back to your original grid.
    Otherwise, sample max_candidates random directions on the unit (N-1)-sphere.

    In spherical coordinates:
    rows: points
    columns: coordinates r, angles

    max_candidates should be by default inf and set only intentionally because
    this avoids user mistakes - otherwise people might for example benchmark
    different K without no effect.
    Randomly generated K candidates option used to be provided outside
    the function to save compute time (because fixed K need to be
    generated only once but the random ones should be generated at every
    iteration), however, it's not expensive and is likely more logical to
    provide it here. One needs both max_candidates and generate_randomly
    because benchmarking requires the ability to generate the same number of
    K candidates with both methods.

    """
    
    # effective spherical dimension
    # The dimensionality of the serach is effectively reduced by one if the
    # search is constrained to sum(X_i) = 1.
    N = dimensionality - 1 if constrain_sum_x else dimensionality

    # how many grid points WOULD we have?
    # Number of angles to be tested in total.
    K = M**(max(1, N-1))
    if (K <= max_candidates) and (generate_randomly is False):
        # — original grid approach —

        # Filling in the spherical coordinates of K equally distributed points around
        # the unit circle centered to point P.
        sph = np.zeros((K, N))
        # Unit circle, r=1. Radius is the first dimensions in n_sphere representation.
        sph[:, 0] = 1.

        # The angle step will be multiplied with these numbers for each point K in each
        # dimension.
        km = np.array(list(product(range(M), repeat=N-1)))

        # Assign the spherical coordinate angles. There are N-1 angles.
        # N-2 first ones have a range [0,pi], the last one [0,2*pi]. We look at
        # only the first half of the hypersphere (because the lines will span
        # both halves symmetrically anyway).

        # TO DO check: [Already dealt with.] Which logic is the correct one?
        # ranges here are [0, pi/2] and [0,pi], respectively.
        # OR
        # ranges here are [0,pi] and [0,pi], respectively (halving one dimension
        # already cuts out half of the hypersphere points, right?).
        # Dimensions 1, ..., N-2

        if N > 1:
            # Radius of 1 so 1*M in denominator
            sph[:, 1:-1] = km[:, :-1] * np.pi / M
            # Using pi instead of 2pi on purpose (see above)
            sph[:, -1] = km[:, -1] * np.pi / M
        cart_coords = n_sphere.convert_rectangular(sph)

    else:
        # — random sampling instead —

        if K > max_candidates:
            K = max_candidates

        if random_generation_type == 'spherical':

            # Use Muller-Marsaglia normalised Gaussians. Here, the K candidates
            # are distributed randomly on a unit sphere (so they are truly
            # random).
            rand = np.random.normal(size=(K, N))
            cart_coords = rand / np.linalg.norm(rand, axis=1, keepdims=True)

        if random_generation_type == 'cartesian':

            if p is None:
                raise Exception(
                    'The selected point P is required for generating random K candidates with a cartesian method! Provide P as argument p.')

            else:

                # Generate random K candidates that lie in the cartesian search
                # space. The difference to the random sampling on a unit sphere
                # is when P is located close to the corners of the search
                # space - in those cases, the lines are here more likely to be
                # long and span the interior of the search space instead of
                # crossing the corner. This behavior has performed better than
                # (the same number of) equally spaced K candidates in high-
                # dimensional benchmarks (the sperichal random sampling to be
                # tested).
                cart_coords = np.random.rand(K, dimensionality)
                cart_coords = cart_coords - p

    # Remove duplicates (e.g., spherical coordinates (1, 0, theta) will result
    # in cartesian coordinates (1, 0, 0) with all theta angles). Note that
    # np.unique also sorts the array at the same time.
    cart_coords = np.unique(cart_coords, axis=0)

    if plotting and cart_coords.shape[1] >= 2:

        import matplotlib.pyplot as plt
        idx0_for_plot = 0
        idx1_for_plot = N-1

        plt.figure()
        plt.scatter(cart_coords[:, idx0_for_plot],
                    cart_coords[:, idx1_for_plot])
        plt.xlabel('$x_' + str(idx0_for_plot) + '$')
        plt.ylabel('$x_' + str(idx1_for_plot) + '$')
        plt.title('All the points K')
        if generate_randomly:
            plt.text(0, 0, 'Generation: ' +
                     random_generation_type + ', p: ' + str(p))
        plt.show()
        
    if constrain_sum_x and ((generate_randomly is False) or 
                            (random_generation_type != 'cartesian')):
        # Return back to the true dimensionality of the search space.
        cart_coords = transform_to_Nplusone_dim(cart_coords)


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
    idcs = np.ravel(np.where((diag_A == 0)))  # & np.ravel((b != 0)))[0]

    diag_A_temp = np.delete(diag_A, idcs, axis=0)
    b_temp = np.delete(b, idcs, axis=0)

    # Confirm that the matrix is invertible. We assume that after the deletions
    # above, A is invertible (something is wrong if it is not). Normally, you
    # would check invertibility with a determinant but np.det() rounds to 0
    # with large matrices. So let's utilize diagonal matrix determinant rule
    # instead (A not invertible if det(A) = 0, det(diag) = d_11*d_22*...*d_NN).
    if np.any(diag_A_temp == 0) == True:

        # Not invertible, solve with least squares.
        A_temp = np.diag(diag_A_temp)
        t = lstsq(A_temp, b_temp, rcond=None)[0]
        print('Used least-squares solver (why?).')
        print('det(A)=', np.linalg.det(A_temp), 'N_unique_eigenvalues=',
              np.unique(np.linalg.eig(A_temp)[0]).shape[0])
        print('Shape A_temp: ', A_temp.shape)

    else:

        # Invert A and solve.
        diag_Ainv_temp = np.reshape(1/diag_A_temp, (-1, 1))
        t = diag_Ainv_temp * b_temp

        # Ainv_temp = np.diag(diag_Ainv_temp)
        # t = np.matmul(Ainv_temp, b_temp)

        # Invertible matrix, can be solved exactly.
        # t = solve(A_temp, b_temp)

    # Fix the dimensionality of the variable.
    for i in idcs:
        t = np.insert(t, i, np.nan, axis=0)

    t = np.reshape(t, (K, N))

    return t


def plot_K_P_in_3D(a, p, title=None, first_point_label='K_cand',
                   plot_triangle=True, lims=[-2, 2], show_p_coord=False,
                   show_a_coord=False):

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
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
        ax.plot(np.ravel([p[:, 0], p[:, 0] + a[i, 0]]),
                np.ravel([p[:, 1], p[:, 1] + a[i, 1]]),
                np.ravel([p[:, 2], p[:, 2] + a[i, 2]]), c='b',
                linewidth=0.5, linestyle='--')

    if plot_triangle is True:

        ax.plot([1, 1], [0, 0], [lims[0], 0], c='k', linewidth=0.5,
                linestyle='--')
        ax.plot([0, 0], [1, 1], [lims[0], 0], c='k', linewidth=0.5,
                linestyle='--')
        ax.plot([0, 0], [0, 0], [lims[0], 1], c='k', linewidth=0.5,
                linestyle='--')
        ax.plot([1, 0, 0, 1], [0, 1, 0, 0], [lims[0], lims[0], lims[0], lims[0]],
                c='k', linewidth=0.5, linestyle='--')
        ax.plot([1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], c='k',
                label='Constraint')

    else:

        ax.plot(np.ravel([p[0, 0], 0]),
                np.ravel([p[0, 1], 0]),
                np.ravel([p[0, 2], -2]), c='k',
                linewidth=0.5, linestyle='--')
    label = 'P'
    if show_p_coord is True:
        label = label + ' ' + str(p)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='k', label=label)

    label = first_point_label
    if show_a_coord is True:
        label = label + ' ' + str(p+a)
    ax.scatter(p[:, 0] + a[:, 0], p[:, 1] + a[:, 1], p[:, 2] + a[:, 2], c='b',
               label=label)
    # ax.scatter(a2[:,0], a2[:,1], a2[:,2], c = 'r')

    # ax.set_proj_type('ortho')
    plt.legend()
    plt.title(title)
    plt.show()

'''
def extract_inlet_outlet_points(p, K_cand, emax, emin, M,
                                constrain_sum_x=False, plotting=False):
    """
    Vectorized boundary‐intersection: compute tA and tB for each K in one shot.
    """
    
    # ensure shapes
    
    pr = p.reshape(1, -1)               # (1, D)
    K = K_cand.shape[0]                # number of candidates
    D = K_cand.shape[1]                # dimensionality

    # broadcast p to (K,D)
    p_rep = np.repeat(pr, K, axis=0)    # (K, D)
    Kmat = K_cand                     # (K, D)

    # avoid divide-by-zero headaches
    with np.errstate(divide='ignore', invalid='ignore'):
        t_low = (emin - p_rep) / Kmat  # may be +/–inf or nan where Kmat=0
        t_high = (emax - p_rep) / Kmat

    # stack both into shape (K, 2*D)
    t_stack = np.concatenate([t_low, t_high], axis=1)

    # compute tA = max negative crossing, tB = min positive crossing
    tA = np.nanmax(np.where(t_stack <= 0, t_stack, np.nan),
                   axis=1, keepdims=True)
    tB = np.nanmin(np.where(t_stack >= 0, t_stack, np.nan),
                   axis=1, keepdims=True)

    # any NaNs mean boundary itself is the only valid point
    tA = np.nan_to_num(tA, 0.0)
    tB = np.nan_to_num(tB, 0.0)

    # compute A and B
    A = p_rep + tA * Kmat
    B = p_rep + tB * Kmat

    if plotting and D == 3:
        plot_K_P_in_3D(Kmat, p, "candidates", plot_triangle=constrain_sum_x)
        plot_K_P_in_3D(K_cand, p, 'Actual 3D space', first_point_label = 'K_cand')
        plot_K_P_in_3D(K_cand-p, p-p, 'Transformed 3D space (P in origo)', first_point_label = 'K_cand', 
                       plot_triangle = False)
        plot_K_P_in_3D(A-p, p, 'Actual 3D space', first_point_label = 'A', lims = [0,1])
        plot_K_P_in_3D(B-p, p, 'Actual 3D space', first_point_label = 'B', lims = [0,1])
    
    print('New!')
    return A, B, tA, tB

'''
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
    
    # Solve t_max values for each point k.
    tmax = solve_t_matrix_eq(p, K_cand, emax)

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
                
                print("Constraint not fulfilled for A!\n")
                #raise Exception("Constraint not fulfilled! a = ", a, ", b = ",
                #                b, ", p = ", p, ", K_cand = ", K_cand, ", tA = ",
                #                tA, ", tB = ", tB)
            
            if test_constraint(b[[i],:]) == False:
                
                print("Constraint not fulfilled for B!\n")
                #raise Exception("Constraint not fulfilled! a = ", a, ", b = ",
                #                b, ", p = ", p, ", K_cand = ", K_cand, ", tA = ",
                #                tA, ", tB = ", tB)
        
    # If any of the points are significantly outside the search space at this point, it means
    # P is on the boundary of the search space and, additionally, there is no t
    # value that could make the line candidate get inside the search space. I.e.,
    # P is in these cases always in the corner, I think.
    # Let's set t values for those points to zero (per the algo above, "the other
    # point A or B" is already zero, so in practice the line length goes to zero).
    a[np.isclose(a,0, atol= 1e-10)] = 0
    b[np.isclose(b,0, atol= 1e-10)] = 0
    
    if np.any((a > emax)) or np.any((a < emin)):
        tA[np.any((a > emax), axis = 1)] = 0
        tA[np.any((a < emin), axis = 1)] = 0
        a = p + tA * K_cand
    
    if np.any((b > emax)) or np.any((b < emin)):
        
        tB[np.any((b > emax), axis = 1)] = 0
        tB[np.any((b < emin), axis = 1)] = 0
        b = p + tB * K_cand
    
    if (N == 3) and (plotting == True):
    
        #bp()
        plot_K_P_in_3D(K_cand, p, 'Actual 3D space', first_point_label = 'K_cand')
        plot_K_P_in_3D(K_cand-p, p-p, 'Transformed 3D space (P in origo)', first_point_label = 'K_cand', 
                       plot_triangle = False)
        plot_K_P_in_3D(a-p, p, 'Actual 3D space', first_point_label = 'A', lims = [0,1])
        plot_K_P_in_3D(b-p, p, 'Actual 3D space', first_point_label = 'B', lims = [0,1])
        #print('P: ', p)
        #print('K: ', K_cand-p)
        

    return a, b, tA, tB



def integrate_over_acqf(p, K, t_start, t_stop, n_points, acq_object,
                        acq_max=True, acq_params=None,
                        normalize_line_length=False):
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
    normalize_line_length : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    I : TYPE
        DESCRIPTION.

    """
    t_steps = np.linspace(t_start, t_stop, n_points,
                          axis=0)  # Works for arrays and single points.
    delta_t = np.abs(t_steps[1] - t_steps[0])

    x_steps = np.tile(p.T, n_points).T + \
        (np.reshape(t_steps, (n_points, 1)) * K)

    acq_values = get_acq(x_steps, acq_object=acq_object,
                         acq_params=acq_params)

    # In GPyOpt, the acquisition values are negative but could be otherwise
    # in other packages and also with other acquisition functions.
    if acq_max is False:
        acq_values = -acq_values

    # Acquisition functions can have positive or negative values. This caused
    # issues in normalizing/shifting the acquisition function. Thus, let's just
    # do the normal integration.

    # if np.max(acq_values) != np.min(acq_values):
    #
    #    acq_values_norm = (acq_values - np.min(acq_values))/(np.max(acq_values) - np.min(acq_values))
    #
    # else:
    #
    #    # Cannot normalize if there is no knowledge of the range.
    #    acq_values_norm = acq_values

    # Calculate the integral
    # * K should in principle be here but K magnitude is always 1.
    I = np.sum(acq_values) * delta_t

    return I

# def integrate_all_K_over_acqf(p, K_cand, t_starts, t_stops,
#                               n_points, acq_object,
#                               acq_max=True, acq_params=None):
#     """
#     Fully vectorized integration: builds all x_steps in one big array, calls
#     get_acq() exactly once, then reshapes.
#     """
#     K = K_cand.shape[0]
#     D = K_cand.shape[1]

#     # build t_grid: shape (K, n_points)
#     ts = np.linspace(0, 1, n_points, endpoint=True)
#     # each row: t = t_start + (t_stop - t_start) * ts
#     t_grid = t_starts + (t_stops - t_starts) * ts.reshape(1, -1)  # (K, n_points)

#     # expand to (K, n_points, D)
#     p_rep = p.reshape(1, 1, D)
#     K_rep = K_cand.reshape(K, 1, D)
#     x_steps = p_rep + t_grid[:,:,None] * K_rep            # (K, n_points, D)
#     flat_x  = x_steps.reshape(-1, D)                      # (K*n_points, D)

#     # one shot acquisition
#     acq_vals = get_acq(flat_x, acq_object=acq_object, acq_params=acq_params)
#     if not acq_max:
#         acq_vals = -acq_vals
#     acq_vals = acq_vals.reshape(K, n_points)

#     # approximate integral = sum(acq) * delta_t
#     delta_t = (t_stops - t_starts).flatten() / (n_points - 1)
#     I = np.sum(acq_vals, axis=1) * delta_t

#     return I


def integrate_all_K_over_acqf(p, K_cand, t_starts, t_stops, n_points,
                              acq_fun, acq_max=True, acq_params=None):
    """
    Vectorized integration of the acquisition function along each line.
    Builds one giant X_steps matrix of shape (K*n_points, D),
    calls GP_pred/acquisition ONCE, then reshapes back to (K, n_points)
    and integrates.  This cuts GP_pred calls from O(K) down to 1.
    """
    K = K_cand.shape[0]
    D = K_cand.shape[1]

    # 1) build all the t-steps at once: shape (K, n_points)
    t_lin = np.linspace(0.0, 1.0, n_points, endpoint=True)
    t_steps = t_starts.reshape(K, 1) * (1 - t_lin) + \
        t_stops.reshape(K, 1) * t_lin
    # shape (K, n_points)

    # 2) build all x_steps: shape (K, n_points, D) → (K*n_points, D)
    P_rep = np.repeat(p.reshape(1, D), K * n_points, axis=0)
    K_rep = np.repeat(K_cand, n_points, axis=0)
    t_rep = t_steps.reshape(K * n_points, 1)
    X_all = P_rep + K_rep * t_rep  # (K*n_points, D)

    # 3) call acquisition function once on the big batch
    Acq_all = get_acq(X_all, acq_object=acq_fun, acq_params=acq_params)

    # In GPyOpt, the acquisition values are negative but could be otherwise
    # in other packages and also with other acquisition functions.
    if not acq_max:
        Acq_all = -Acq_all

    # 4) reshape and integrate
    Acq_mat = Acq_all.reshape(K, n_points)
    delta_t = np.abs(t_lin[1] - t_lin[0])

    # Acquisition functions can have positive or negative values. This caused
    # issues in normalizing/shifting the acquisition function. Thus, let's just
    # do the normal integration.
    I_all = Acq_mat.sum(axis=1) * delta_t

    return I_all


def choose_K(BO_object, p, K_cand, emax=1, emin=0, M=2,
             acq_max=True, selection_method='integrate_acq_line',
             constrain_sum_x=False, plotting=False, acq_params=None):
    """
    Wire up the new vectorized integrator & extractor. Everything else is identical.
    """
    A, B, tA, tB = extract_inlet_outlet_points(p, K_cand, emax, emin, M,
                                               constrain_sum_x=constrain_sum_x,
                                               plotting=plotting)
    if selection_method == 'integrate_acq_line':
        acq_object = define_acq_object(BO_object, acq_params=acq_params)
        I_all = integrate_all_K_over_acqf(p, K_cand, tA, tB,
                                          n_points=500,
                                          acq_fun=acq_object,
                                          acq_max=acq_max,
                                          acq_params=acq_params)
        # idx = np.argmax(I_all, axis=0)
        # — tie-break if every line is equally good —
        if np.allclose(I_all, I_all.flat[0], atol=1e-8):
            # all integrals identical: pick a random direction
            idx = np.random.randint(0, I_all.shape[0])
        else:
            idx = np.argmax(I_all, axis=0)
    elif selection_method == 'random_line':
        idx = np.random.randint(0, K_cand.shape[0])
    else:
        raise Exception("Not implemented.")

    return A[idx], B[idx], tA[idx], tB[idx], K_cand[idx]


def pick_random_init_data(n_init, N, emax, emin, M, K_cand,
                          constrain_sum_x=False, plotting=True):

    # Randomly selected init points P.
    p = np.random.rand(n_init, N)

    if constrain_sum_x == True:

        # Init points need to fulfill the constraint.
        for i in range(n_init):

            while test_constraint(p[[i], :]) == False:

                p[i, :] = np.random.rand(1, N)

    # Acquisition function is uniform at this stage so let's pick the line randomly.
    A_sel = np.empty(p.shape)
    B_sel = np.empty(p.shape)
    tA_sel = np.empty((p.shape[0],))
    tB_sel = np.empty((p.shape[0],))
    K_sel = np.empty(p.shape)

    for i in range(n_init):

        if M is not None:

            A_sel[i, :], B_sel[i, :], tA_sel[i], tB_sel[i], K_sel[i, :] = choose_K(
                None, p[[i], :], K_cand, emax=emax, emin=emin, M=M,
                selection_method='random_line',
                constrain_sum_x=constrain_sum_x, plotting=plotting,
                acq_object=None)

        else:

            A_sel[i, :] = np.nan
            B_sel[i, :] = np.nan
            tA_sel[i] = np.nan
            tB_sel[i] = np.nan
            K_sel[i, :] = np.nan

    return p, A_sel, B_sel, tA_sel, tB_sel, K_sel


def compute_x_coords_along_lines(idx_to_compute, N, n_droplets,
                                 tA_sel, tB_sel, p_sel, K_sel):
    """
    Always returns an (n_droplets * len(idx_to_compute), N) array,
    even if len(idx_to_compute)==1 or n_droplets==1.
    """
    import numpy as np

    # Coerce everything to numpy arrays and ensure 2D shapes
    p_arr = np.asarray(p_sel)
    K_arr = np.asarray(K_sel)
    tA_arr = np.asarray(tA_sel)
    tB_arr = np.asarray(tB_sel)

    if p_arr.ndim == 1:
        p_arr = p_arr.reshape(1, -1)          # (1, N)
    if K_arr.ndim == 1:
        K_arr = K_arr.reshape(1, -1)          # (1, N)
    tA_arr = tA_arr.reshape(-1, 1)            # (K_cand, 1)
    tB_arr = tB_arr.reshape(-1, 1)

    # Number of tasks
    K_idxs = len(idx_to_compute)
    D = N
    M = n_droplets

    # Select the subset
    p_sub = p_arr[idx_to_compute, :]         # (K_idxs, N)
    K_sub = K_arr[idx_to_compute, :]         # (K_idxs, N)
    tA_sub = tA_arr[idx_to_compute, :]        # (K_idxs, 1)
    tB_sub = tB_arr[idx_to_compute, :]        # (K_idxs, 1)

    # Build a (K_idxs, M, 1) grid of t values in [tA, tB]
    t_lin = np.linspace(0.0, 1.0, M).reshape(1, M, 1)  # (1, M, 1)
    t_start = tA_sub.reshape(K_idxs, 1, 1)               # (K_idxs, 1, 1)
    t_stop = tB_sub.reshape(K_idxs, 1, 1)
    t_steps = t_start + (t_stop - t_start) * t_lin       # (K_idxs, M, 1)

    # Now lift P and K into the same shape
    P_rep = p_sub.reshape(K_idxs, 1, D)                 # (K_idxs, 1, D)
    K_rep = K_sub.reshape(K_idxs, 1, D)                 # (K_idxs, 1, D)

    # Compute all the droplet positions
    x_steps = P_rep + t_steps * K_rep                    # (K_idxs, M, D)

    # Flatten into (K_idxs*M, D)
    return x_steps.reshape(K_idxs * M, D)
