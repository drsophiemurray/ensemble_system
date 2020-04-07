"""
Ensemble code used for the paper, 'Modeling Ensemble Forecasting of Major Solar Flares',
under review Journal Space Weather Space Climate (2020).

Originally created by Jordan Guerra, and tidied up and tested by Sophie Murray.

"""

import pickle
import random
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from scipy.stats import kendalltau

# Define some options
METRIC = 'brier'
FLARE_CLASS = 'M'
MEAN_CLIMATOLOGY = True
UNCONSTRAINED = True
SPLIT_DATASET = False
DATA_SOURCE = "/Users/sophie/Dropbox/Ensemble_ii/software/input_180829.p"


def main():
    """
    Load forecasts and flare event information for multiple methods,
    and create an ensemble based on whatever chosen defined metric.

    Options above include
    - choice of metric,
    - what flare class data to choose from (in this data set M or X),
    - include a mean climatology or not,
    - create a constrained or unconstrained ensemble,
    - split the dataset to provide validation data or not

    Code outputs the resulting ensemble probability
    (and maybe that validation data set if Jordan confirms that's what it actually is!)
    """
    # Load input data
    input_forecasts = load_data(DATA_SOURCE)
    methods = list(input_forecasts.keys())[:-1]

    # Split the forecasts from the events
    # First take out whatever the chosen flare class is
    forecasts = [input_forecasts[j][FLARE_CLASS] for j in list(input_forecasts.keys())[:-1]]
    # Then grab associated events for chosen forecasts
    events = input_forecasts['EVENTS'][FLARE_CLASS]

    # Get mean of all events and add to members if requested
    if MEAN_CLIMATOLOGY is True:
        ebar = np.mean(events)
        forecasts.append(np.array([ebar for i in range(len(events))]))
        methods.append('Climatology')

    # Create ensemble
    ensemble_probability = create_ensemble(forecasts, events)

    return ensemble_probability


def create_ensemble(forecasts, events):
    """
    Create the ensemble by getting the weights via optimization?
    """
    # Get no. of forecasts in ensemble (minus the events)
    no_members = len(forecasts)
    no_forecasts = len(forecasts[0])
    forecast_indices = list(range(no_forecasts))

    # If you want to randomly split sample, shuffle the indices then split in half
    if SPLIT_DATASET is True:
        random.shuffle(forecast_indices)
        t_indices = forecast_indices[:no_forecasts // 2]
        v_indices = forecast_indices[(no_forecasts // 2) + 1:]
        t_forecasts = [forecasts[ii][t_indices] for ii in range(no_members)]
        v_forecasts = [forecasts[ii][v_indices] for ii in range(no_members)]
        t_events = events[t_indices]
        v_events = events[v_indices]
    else:
        t_forecasts = forecasts
        t_events = events

    # Expand bounds if unconstrained is set
    if UNCONSTRAINED is True:
        ws_ini = np.array([random.uniform(-1., 1.) for i in range(no_members)])
        bounds = tuple((-1., 1.) for ws in ws_ini)
    else:
        ws_ini = np.array([random.uniform(0., 1.) for i in range(no_members)])
        bounds = tuple((0.0, 1.0) for ws in ws_ini)

    # Set up arrays for weights
    dws = np.array([1.0 for ii in range(no_members)])  # 7 elements in an array
    weights = []

    # Calculate weights
    constraints = ({'type': 'eq',
                    'fun': lambda ws: np.array(sum([ws[ii] for ii in range(no_members)]) - 1.0),
                    'jac': lambda ws: dws})

    res = minimize(optimize_funct,
                   ws_ini,
                   constraints=constraints, bounds=bounds,
                   method='SLSQP',
                   jac=False,
                   options={'disp': False, 'maxiter': 10000, 'eps': 0.001})

    # Get ensemble probability using calculated weights
    # First mulitply the weights out, then sum them together
    combined_probability = sum([(res.x[i] * forecasts[i]) for i in list(range(no_members))])

    return combined_probability


def load_data(file):
    """
    Read pickle dump created for analysis
    'file' contains a dictionary of ndarrays (of size 1096):
    - 'MEMBER_0'...'MEMBER_5' : 'M': array and 'X': array
    - 'EVENTS': {'M': array([ 1.,  1.,  1., ...,  0.,  0.,  0.]),
                 'X': array([ 0.,  0.,  0., ...,  0.,  0.,  0.])}
    """
    return pickle.load(open(file, "rb"), encoding='latin1')


def metric_funct(metric, t, e):
    """
    Define the metrics to be used to create the ensemble
    Options are
    - brier
    - lcc
    - mae
    - nlcc_rho
    - nlcc_tau
    - rel
    """
    global funct
    # BRIER
    if metric == 'brier':
        funct = np.mean((t - e) ** 2.0)
    # LCC
    if metric == 'LCC':
        funct = np.corrcoef(t, e)[0, 1]
    # MAE
    if metric == 'MAE':
        funct = np.mean(np.abs(t - e))
    # NLCC_RHO
    if metric == 'NLCC_RHO':
        funct = spearmanr(t, e)[0]
    # NLCC_TAU
    if metric == 'NLCC_TAU':
        funct = kendalltau(t, e)[0]
    # REL
    if metric == 'REL':
        n1 = 10
        delta = 1. / float(n1)
        pgrid = (np.arange(n1) * delta)
        pvec = []
        evec = []
        numvec = []
        for i0 in range(n1 - 1):
            if i0 + 1 > n1 - 1:
                m = np.where(t >= pgrid[i0])
            else:
                m = np.where(np.logical_and(t >= pgrid[i0], t < pgrid[i0 + 1]))
            pvec.append(np.mean(t[m[0]]))
            evec.append(np.mean(e[m[0]]))
            numvec.append(len(m[0]))
        rel_vec = [nn * ((pp - ee) ** 2.0) for nn, pp, ee in zip(numvec, pvec, evec)]
        funct = np.nansum(rel_vec) / len(t)
    return funct


def optimize_funct(ws):
    """
    Optimzation procedure
    """
    global ofunct
    combination = sum([ws[i] * t_forecasts[i] for i in range(no_members)])
    ofunct = metric_funct(METRIC, combination, t_events)
    if METRIC == 'LCC':
        ofunct = -1 * ofunct
    return ofunct


if __name__ == '__main__':
    main()
