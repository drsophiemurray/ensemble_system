# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 20:32:01 2016

@author: jguerraa
"""

import numpy as np
from scipy.optimize import minimize
import dateutil
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import pickle
import random
#
#READ INPUT DATA
in_struc = pickle.load( open( "/Users/somurray/Dropbox/Ensemble_ii/software/input_180829.p", "rb" ),encoding='latin1')
## ndarrays, e.g., in_struc['MEMBER_0']['M'] -- array([ 0. ,  0.22119922,  0.33634975, ...,  0. ])
#DEFINE SOME OPTIONS
metrics = ['brier'] #LIST OF METRICS OR 'ALL'
fclass = 'M' # LIST OR 'ALL'
if fclass == 'M':
    fclass1 = 0
if fclass == 'X':
    fclass1 = 1

#
n = len(in_struc)-1
eqw = 1./n
methods = in_struc.keys()[:-1]
option = 'Unconstrained' # Or 'Unconstrained'
forecasts = [in_struc[j][fclass] for j in methods]
events = in_struc['EVENTS'][fclass]
#
#ws_ini = np.array([0.0 for i in range(n)])
   
#DEFINITION OF METRICS
def metric_funct(metric,t,e):
    global funct
    # BRIER
    if metric == 'brier':
        funct = np.mean((t - e)**2.0)
    # LCC
    if metric == 'LCC':
        funct = np.corrcoef(t, e)[0,1]
    # MAE
    if metric == 'MAE':
        funct = np.mean( np.abs(t - e) )
    # NLCC_RHO
    if metric == 'NLCC_RHO':
        funct = spearmanr(t, e)[0]
    # NLCC_TAU
    if metric == 'NLCC_TAU':
        funct = kendalltau(t, e)[0]
    
    # REL
    if metric == 'REL':
        n1 = 10
        delta = 1./float(n1)
        pgrid = (np.arange(n1)*delta)
        pvec = []
        evec = []
        numvec = []
        for i0 in range(n1-1):
            #print pgrid[i], pgrid[i+1]
            if i0+1 > n1-1:
                m = np.where(t >= pgrid[i0])
            else:
                m = np.where(np.logical_and(t >= pgrid[i0],t < pgrid[i0+1]))
            #print m
            pvec.append(np.mean(t[m[0]]))
            evec.append(np.mean(e[m[0]]))
            numvec.append(len(m[0]))
        #print numvec
        rel_vec = [nn*((pp-ee)**2.0) for nn,pp,ee in zip(numvec,pvec,evec)]
        funct = np.nansum(rel_vec)/len(t)
    return funct
    
def optimize_funct(ws):
    global ofunct
    combination = sum([ws[i]*t_forecasts[i] for i in range(n)])
    ofunct = metric_funct(metric,combination,t_events)#np.mean((combination - events)**2.0)
    if metric == 'LCC':
        ofunct = -1*ofunct
    #    
    return ofunct

"""
# CALCULATE BRIE SCORES FOR EACH METHOD
for i in metrics:
    metric = i
    for j1,j2 in zip(methods,range(n)):
        print '%s score for %s = ' %(i,j1), metric_funct(i,t_forecasts[j2],t_events)

    #CALCULATE METRIC FOR EQUAL WEIGHTS ENSEMBLE
    peq = sum([eqw*t_forecasts[p] for p in range(n)])
    print '%s score for Equal-weights = ' %i, metric_funct(i,peq,t_events)
"""
# DEFINE JACOBIAN MATRIX FOR OPTIMIZATION (NUMERICALLY)
# DERIVATIVE OF CONSTRAIN
#dd = [ws[i] for i in range(n)]
#dd.append(-1.0)
#const_deriv = np.array(sum(dd))
# Bound weights to positive values
for i in metrics:
    #
    metric = i
    print metric
    grand_average = []
    #
    n_t = len(forecasts[0])
    indices = range(n_t)    
    #
    for rand in range(100):
        #
        # RANDOMLY SPLIT THE SAMPLE
        #
        n = len(in_struc)-1
        eqw = 1./n
        #
        random.shuffle(indices)
        t_indices = indices[:n_t/2]
        v_indices = indices[(n_t/2)+1:]
        #
        t_forecasts = [forecasts[ii][t_indices] for ii in range(n)] 
        v_forecasts = [forecasts[ii][v_indices] for ii in range(n)]
        t_events = events[t_indices]
        v_events = events[v_indices]
        #
        if option == 'Unconstrained':
            ebar  = np.mean(t_events)
            temp = [ebar for i in range(len(t_events))]
            t_forecasts.append(np.array(temp))
            methods.append('Climatology')
            n += 1 
            eqw = 1./n        
        #        
        dws = np.array([1.0 for ii in range(n)])
        #
        weights = []
        for j in range(500):

            ws_ini = np.array([random.uniform(0.,1.) for i in range(n)])
            bnds = tuple((0.0,1.0) for ws in ws_ini)
            if option == 'Unconstrained':
                ws_ini = np.array([random.uniform(-1.,1.) for i in range(n)])
                bnds = tuple((-1.,1.) for ws in ws_ini) 
            #
            cons = ({'type': 'eq', 'fun' : lambda ws: np.array(sum([ws[ii] for ii in range(n)])-1.0), 'jac' : lambda ws: dws})
        
            res = minimize(optimize_funct, ws_ini, constraints=cons, bounds=bnds, method='SLSQP', jac=False, options={'disp': False,'maxiter': 10000, 'eps': 0.001})
        
            weights.append( [ii for ii in res.x] )
        
        weights = np.array(weights)
        #plt.figure()
        """
        nplot = int(round(np.sqrt(n)))
        f, ax = plt.subplots(nplot,nplot)
        ax = np.reshape(ax,nplot*nplot)
        f.suptitle(metric)
        for plot in enumerate(ax):
            try:
                plot[1].hist(weights[:,plot[0]])
                plot[1].set_title(methods[plot[0]])
            except:
                continue
        """
        w_vals = [ [np.mean(weights[:,i]),np.std(weights[:,i])] for i in range(n) ]
        w_vals = np.array(w_vals)
        #
        for i,j in zip(w_vals,methods):
            print j, '%s +/- %s'%(i[0],i[1])#str(round(i[0],3))
            
        #
        grand_average.append(w_vals)









   
#
#for i,j in zip(res.x,methods):
#    print j,str(round(i,3))
#print res.x[0] + res.x[1] + res.x[2] + res.x[3] + res.x[4] + res.x[5] 
"""
comb_p = res.x[0]*p0 + res.x[1]*p1 + res.x[2]*p2 + res.x[3]*p3 + res.x[4]*p4 + res.x[5]*p5 + res.x[6]*np.mean(e)

for i in enumerate(comb_p):
    if i[1] < 0.:
        comb_p[i[0]] = 0.0

def perc_diff(m_scores,en_score,metric):
    diff = []
    for score in m_scores:
        diff.append(100.*((en_score-score)/score))
    #print diff
    min_metrics = ['brier','mae','rel']
    #max_metrics = ['LCC','NLCC_t','NLCC_r','ROCC','RES']
    if metric in min_metrics:
        diff = [-i for i in diff]
    return diff#[np.mean(diff),np.std(diff)]

diff1 = []
for i in ps:
    diff1.append(1.-brie_score(i))
diff1.append(1.-brie_score(peq))
diff1.append(1.-brie_score(comb_p))
print diff1

score_perc = perc_diff(em_scores,brie_score(comb_p),metric)

#SAVE DATA WITH PICKLE
brier_str = [res.x,score_perc]
pname =  metric+"_results_unc_"+fclass+".p"
pickle.dump( brier_str, open(pname, "wb" ) )
#
brier_str = [res.x,diff1]
pname =  metric+"_results_unc_"+fclass+"_metrics.p"
pickle.dump( brier_str, open(pname, "wb" ) )
#
plt.clf()
xfmt = mdates.DateFormatter('%M %YY')
fig = plt.figure(1, figsize=(10, 6))
plt.plot(time, e, label='Events', color='grey')
plt.plot(time, peq, label='Eq w probability', color='purple',lw=2)
plt.plot(time, comb_p, label='Combined probability',color='orange',lw=2)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15, rotation=25)
plt.legend()
#plt.savefig('ensembles_2015.pdf',bbox_inches='tight')

# SAVE DATA IN R FORMAT
fname = metric+"_forecast_"+fclass+"_unc.dat"
f = open(fname,'w')
f.write("obs,prob\n")
#for obs, mag4, assa, asap, noaa, equal, ensemble in zip(e,p0,p1,p2,p3,peq,comb_p):
for obs, prob in zip(e,comb_p):
    line = "%d,%0.3f" % (obs,prob)
    f.write(line+"\n")
f.close()

"""



## creating the ensemble data file
# import numpy as np
# from scipy.optimize import minimize
# import dateutil
# from matplotlib import pyplot as plt
# import matplotlib.dates as mdates
# import pickle
# #
# metric = 'brier'
# fclass = 'M'
# if fclass == 'M':
#     fclass1 = 0
# if fclass == 'X':
#     fclass1 = 1
# #
# # READING THE TIME SERIES
# # MAG4
# data = np.loadtxt('mag4_fd_probs_final_24h.txt',usecols=(2, 3), unpack=True)
# p0 = data[:][fclass1]
# p0X = data[:][1]
# # ASSA
# data = np.loadtxt('assa_fd_probs_final_24h.txt',usecols=(2, 3), unpack=True)
# p1 = data[:][fclass1]
# p1X = data[:][1]
# #p1 = movingaverage(p1,10)
# # ASAP
# data = np.loadtxt('asap_fd_probs_final_24h.txt',usecols=(2, 3), unpack=True)
# p2 = data[:][fclass1]
# p2X = data[:][1]
# #p2 = movingaverage(p2,10)
# # NOAA
# data = np.loadtxt('noaa_fd_probs_final_24h.txt',usecols=(2, 3), unpack=True)
# p3 = data[:][fclass1]
# p3X = data[:][1]
# # MOSWOC
# data = np.loadtxt('moswoc_fd_probs_final_24h.txt',usecols=(2, 3), unpack=True)
# p4 = data[:][fclass1]
# p4X = data[:][1]
# # FPS
# data = np.loadtxt('fps_fd_probs_final_24h.txt',usecols=(2, 3), unpack=True)
# p5 = data[:][fclass1]
# p5X = data[:][1]
# # EVENTS
# data = np.loadtxt('events_final_24h.txt',usecols=(2, 3), unpack=True)
# e = data[:][fclass1]
# eX = data[:][1]
# #
# struc = {'MEMBER_0':{'M':p0,'X':p0X},'MEMBER_1':{'M':p1,'X':p1X},'MEMBER_2':{'M':p2,'X':p2X},\
#           'MEMBER_3':{'M':p3,'X':p3X},'MEMBER_4':{'M':p4,'X':p4X},'MEMBER_5':{'M':p5,'X':p5X}, \
#           'EVENTS':{'M':e,'X':eX}}
#
# pname =  "input_180829.p"
# pickle.dump( struc, open(pname, "wb" ) )


