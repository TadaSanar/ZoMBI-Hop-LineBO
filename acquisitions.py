import utils
import numpy as np
from scipy.stats import norm

def EI(X, GP_model, fX_best, xi, ftype, n=None, ratio=None, decay=None):
    '''
    Expected Improvement acquisition function.
    :return: acquisition value of the next point to sample in the search space
    '''
    mean, std = utils.GP_pred(X, GP_model, ftype)
    fX_best = np.array(fX_best.astype(ftype)).reshape(1,-1)
    z = (fX_best - mean - xi) / std
    return (fX_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)

# def LCB_ada(X, GP_model, ratio, decay, n, ftype, fX_best=None, xi=None):
#     '''
#     Lower Confidence Bound Adaptive acquisition function.
#     :return: acquisition value of the next point to sample in the search space
#     '''
#     mean, std = utils.GP_pred(X, GP_model, ftype)
#     return - mean + ratio * std * np.power(decay, n)

def LCB_ada(X, GP_model, n, fX_best, ratio, decay, xi, ftype):
    """
    Adaptive Lower Confidence Bound:
      LCB(x) = - mean(x) + ratio * std(x) * decay^n
    """
    import numpy as _np

    # 1) get mean and std from GP_pred
    mean, std = utils.GP_pred(X, GP_model, ftype)

    # 2) force both to 1D arrays so they broadcast elementwise, not into NxN
    mean = _np.asarray(mean).reshape(-1)
    std  = _np.asarray(std ).reshape(-1)

    # 3) compute LCB safely
    return -mean + ratio * std * (decay ** n)