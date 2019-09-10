"""
Created on Sun Sep 25 20:32:01 2016

@author: jguerraa
"""

import np
from scipy.stats import spearmanr
from scipy.stats import kendalltau




def main(members_dictionary):
    """

    :param members_dictionary: A dictionary containing the different members M and X- class forecasts,
                                as well as corresponding flare event list
    :return:
    """
    # settings
    contstraint_option = 'unconstrained'
    metrics = ['brier', 'lcc', 'mae', 'nlcc_rho', 'nlcc_tau', 'rel']
    # load data
    number_of_members =  len(members_dictionary)-1 # for Jordan that was 6
    eqw = 1./number_of_members
    ensemble_method_names = list(members_dictionary.keys())[:-1]
    m_forecasts = [in_struc[j]['M'] for j in ensemble_method_names] #list with 6 arrays, each same size the events array
    x_forecasts = [in_struc[j]['X'] for j in ensemble_method_names]
    m_events = in_struc['EVENTS']['M']
    x_events = in_struc['EVENTS']['M']
    # run model
    ensemble_model(m_forecasts, m_events, metrics)

def calculate_metrics(metric, t, e):
    """
    Defining the metrics to be used in the creation of the ensembles
    :param metric: string, metric to be used
    :param t:
    :param e:
    :return:
    """
    global funct
    if metric == 'brier':
        funct = np.mean((t - e)**2.0)
    if metric == 'lcc':
        funct = np.corrcoef(t, e)[0, 1]
    if metric == 'mae':
        funct = np.mean(np.abs(t - e))
    if metric == 'nlcc_rho':
        funct = spearmanr(t, e)[0]
    if metric == 'nlcc_tau':
        funct = kendalltau(t, e)[0]
    if metric == 'rel':
        n1 = 10
        delta = 1./float(n1)
        pgrid = (np.arange(n1)*delta)
        pvec = []
        evec = []
        numvec = []
        for i0 in range(n1-1):
            if i0+1 > n1-1:
                m = np.where(t >= pgrid[i0])
            else:
                m = np.where(np.logical_and(t >= pgrid[i0],t < pgrid[i0+1]))
            pvec.append(np.mean(t[m[0]]))
            evec.append(np.mean(e[m[0]]))
            numvec.append(len(m[0]))
        rel_vec = [nn*((pp-ee)**2.0) for nn,pp,ee in zip(numvec,pvec,evec)]
        funct = np.nansum(rel_vec)/len(t)
    return funct

def optimize(ws):
    """

    :param ws:
    :return:
    """
    global ofunct
    combination = sum([ws[i]*t_forecasts[i] for i in range(n)])
    ofunct = calculate_metrics(metric, combination, t_events)
    if metric == 'LCC':
        ofunct = -1*ofunct
    return ofunct

def ensemble_model(forecasts, events, metrics):
    """

    :param forecasts:
    :param events:
    :param metrics:
    :return:
    """
    for metric in metrics:
        grand_average = []
        n_t = len(forecasts[0])
        indices = list(range(n_t))
        # Randomly split the sample
        for rand in range(100):





