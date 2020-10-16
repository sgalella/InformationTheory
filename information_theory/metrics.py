import information_theory.config as config

from collections import Counter

import numpy as np

np.seterr(divide='ignore', invalid='ignore')

def entropy(P):
    P_nan = P.copy()
    P_nan[P_nan == 0] = np.nan
    return np.nansum(np.multiply(P_nan, np.log2(1 / P_nan)))

def joint_entropy(P):
    P_nan = P.copy() 
    P_nan[P_nan == 0] = np.nan 
    return np.nansum(np.multiply(P_nan, np.log2(1 / P_nan)))

def conditional_entropy(P):
    P_nan = P.copy() 
    P_nan[P_nan == 0] = np.nan
    
    marginals = np.nansum(P_nan, axis=1)
    P_cond = P_nan / marginals[:, None]

    return np.nansum(np.multiply(P_nan, np.log2(1 / P_cond)))

def mutual_information_from_table(P):
    P_nan = P.copy() 
    P_nan[P_nan == 0] = np.nan
    
    marginals_p1 = np.nansum(P_nan, axis=1)
    marginals_p2 = np.nansum(P_nan, axis=0)
    
    return np.nansum(np.multiply(P_nan, np.log2(P_nan / (np.tensordot(marginals_p1, marginals_p2, axes=0)))))

def mutual_information_from_data(X, Y, num_bins):
    
    N = X.size
    delta = 10e-10
    
    x_min, x_max = range=(X.min() - delta,  X.max() + delta)
    y_min, y_max = range=(Y.min() - delta,  Y.max() + delta)

    X_hist, X_bin = np.histogram(X, bins=num_bins, range=(x_min, x_max))
    Y_hist, Y_bin = np.histogram(Y, bins=num_bins, range=(y_min, y_max))

    X_states = np.digitize(X, X_bin)
    Y_states = np.digitize(Y, Y_bin)
    coords = Counter(zip(X_states, Y_states))

    joint_linear = np.zeros((config.NUM_STATES, config.NUM_STATES))
    for x, y in coords.keys():
        joint_linear[x-1, y-1] = coords[(x, y)] / N

    p_X = X_hist / N
    p_Y = Y_hist / N
    prod_XY = np.tensordot(p_X.T, p_Y, axes=0)

    div_XY = joint_linear / prod_XY
    div_XY[div_XY == 0] = np.nan

    return np.nansum(np.multiply(joint_linear, np.log2(div_XY)))

def transfer_entropy(X, Y):

    coords = Counter(zip(Y[1:], X[:-1], Y[:-1]))

    p_dist = np.zeros((config.NUM_STATES, config.NUM_STATES, config.NUM_STATES))
    for y_f, x_p, y_p in coords.keys():
        p_dist[y_p, y_f, x_p] = coords[(y_f, x_p, y_p)] / (len(X) - 1)

    p_yp = p_dist.sum(axis=2).sum(axis=1)
    p_joint_cond_yp = p_dist / p_yp[:, None, None]
    p_yf_cond_yp = p_dist.sum(axis=2) / p_yp[:, None]
    p_xp_cond_yp = p_dist.sum(axis=1) / p_yp[:, None]

    denominator = np.multiply(p_yf_cond_yp, p_xp_cond_yp)
    denominator[denominator == 0] = np.nan

    division = np.divide(p_joint_cond_yp, denominator[:, :, None])
    division[division == 0] = np.nan

    log = np.log2(division)

    return np.nansum(np.multiply(p_dist, log))