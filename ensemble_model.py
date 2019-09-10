"""
Created on Sun Sep 25 20:32:01 2016

@author: jguerraa
"""

import np
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import random




def main(members_dictionary):
    """

    :param members_dictionary: A dictionary containing the different members M and X- class forecasts,
                                as well as corresponding flare event list
    :return:
    """
    # settings
    unconstrained = True
    metrics = ['brier', 'lcc', 'mae', 'nlcc_rho', 'nlcc_tau', 'rel']
    # load data
#    no_members =  len(members_dictionary)-1 # for Jordan that was 6
#    eqw = 1./number_of_members
    methods = list(members_dictionary.keys())[:-1]
    m_forecasts = [in_struc[j]['M'] for j in methods] #list with 6 arrays, each same size the events array
    x_forecasts = [in_struc[j]['X'] for j in methods]
    m_events = in_struc['EVENTS']['M']
    x_events = in_struc['EVENTS']['M']
    # run model
    ensemble_out = ensemble_model(m_forecasts, m_events, methods, unconstrained)


def ensemble_model(forecasts, events, methods, unconstrained):
    """

    :param forecasts:
    :param events:
    :param metrics:
    :return:
    """
    grand_average = []
    n_t = len(forecasts[0])
    indices = list(range(n_t))
    # Randomly split the sample
    for rand in range(100):
        no_members = len(forecasts)
        eqw = 1. / no_members
        random.shuffle(indices)
        t_indices = indices[:n_t // 2]
        v_indices = indices[(n_t // 2) + 1:]
        t_forecasts = [forecasts[ii][t_indices] for ii in range(no_members)]
        v_forecasts = [forecasts[ii][v_indices] for ii in range(no_members)]
        t_events = events[t_indices]
        v_events = events[v_indices]
        if unconstrained is True:
            ebar  = np.mean(t_events)
            temp = [ebar for i in range(len(t_events))]
            t_forecasts.append(np.array(temp))
            methods.append('climatology')
            no_members += 1
            eqw = 1./no_members
        dws = np.array([1.0 for ii in range(no_members)])
        weights = []
        for j in range(500):
            ws_ini = np.array([random.uniform(0.,1.) for i in range(no_members)])
            bnds = tuple((0.0,1.0) for ws in ws_ini)
            if unconstrained is True:
                ws_ini = np.array([random.uniform(-1.,1.) for i in range(n)])
                bnds = tuple((-1., 1.) for ws in ws_ini)
            cons = ({'type': 'eq',
                     'fun': lambda ws: np.array(sum([ws[ii] for ii in range(n)]) - 1.0),
                     'jac': lambda ws: dws})
            res = minimize(optimize, ws_ini,
                           constraints=cons,
                           bounds=bnds,
                           method='SLSQP',
                           jac=False,
                           options={'disp': False, 'maxiter': 10000, 'eps': 0.001})
            weights.append([ii for ii in res.x])
        weights = np.array(weights)
        w_vals = [[np.mean(weights[:, i]),
                   np.std(weights[:, i])] for i in range(n)]
        w_vals = np.array(w_vals)
        for i, j in zip(w_vals, methods):
            print((j, '%s +/- %s' % (i[0], i[1])))
        grand_average.append(w_vals)
    return grand_average

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
    combination = sum([ws[i]*forecasts[i] for i in range(n)])
    ofunct = calculate_metrics(metric, combination, events)
    if metric == 'LCC':
        ofunct = -1*ofunct
    return ofunct
















