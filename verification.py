"""These functions will have come from multiple locations
- FLARECAST verification_module
- Aoife Bergin code
- Sophie's own code.

What I want:
Probabilistic
- ROC plot; area
- Relibility diagram; reliabilty, resolution, uncertainty
- Brier score
- Brier skill score
- RPS
- RPSS

Categorical:
- Truth table stuff like PC POD, POFD, FAR...
- TSS
- HSS

Other:
- Correlation coefficient
- Mean absolute error
-"""

import numpy as np
import csv
import matplotlib.pyplot as plt


def main():
    """
    Verification engine - ALL the metrics
    Read in the data and depending on type calculate various things
    and/or make plots
    """
    # Lets start with reliability and ROC plots :D


#=======================================================================================================================


def polar_plot():
    """
    Copied from Aisling Bergin's polar_skills.py
    Create a polar plot with the different skill scores calculated for the different forecasts.
    Individual axes plot the forecasts with higher scores at a larger distance from the centre of the plot.
    The largest enclosed area represents the highest scoring forecast overall.
    """
    # stuff that aisling imports
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime as dt
    from shapely import geometry
    # colours for the different areas
    c = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:olive',
         'tab:brown', 'tab:orange',  'tab:red', 'tab:purple',  'tab:pink']
    # open aisling's file
    with open('/Users/somurray/Dropbox/tcd/students/ss_astro_projects/scoreboard/Code_Upload/skill_score.csv', 'r') as f:
        reader = csv.reader(f)
        report = list(reader)
    # aisling hacked the rows and columns - will have to remove
    row_min, row_max = 1, 10  # The rows of the table; forecast by forecast, add one for np.arange effect
    col_min, col_max = 4, 12  # The columns of the table; score by score, add one for the np.arange effect
    # names of scores
    score_names = report[0][col_min:col_max]
    # names of forecasts
    forecast_names = [report[i][0] for i in np.arange(row_min, row_max, 1)]
    # score values for each forecasts (lists within list)
    score_array = [report[i][col_min:col_max] for i in np.arange(row_min, row_max, 1)]
    # get radii for each forecasts score (lists within list)
    radii = []
    for results in score_array:
        r = []
        for result in results:
            skill_array = [float(score_array[i][results.index(result)]) for i in np.arange(0, len(score_array), 1)]
            ranked_skills, ranked_names = zip(*sorted(zip(skill_array, forecast_names)))
            for string in ranked_names:
                if str(forecast_names[score_array.index(results)]) == str(string):
                    index = ranked_names.index(string)
                    r.append((float(index) + 1) / 10)
        radii.append(r)
    # next up total scores and areas overall (single lists)
    total = []
    areas = []
    for array in radii:
        elem = [array[0], array[1], array[5], array[6], array[7]]
        i = np.linspace((2. * np.pi) / float(len(elem)), (2 * np.pi), len(elem)).tolist()
        total.append(np.sum(elem))
        i.append(i[0])
        elem.append(elem[0])
        corners = []
        for j in np.arange(0, len(elem), 1):
            t = i[j]
            r = elem[j]
            corners.append([r * (np.cos(t)), r * (np.sin(t))])
        poly = geometry.Polygon(corners)
        areas.append(poly.area)
    # some sort of sorting?
    names = forecast_names
#    areas, radii, names = zip(*sorted(zip(areas, radii, forecast_names), reverse=True))
    # making the plot
    # start the plot
    ax = plt.subplot(111, projection='polar')
    for array in radii:
        # here aisling is choosing only certain scores
        # namely, ROC area, BSS, RPSS, TSS, and HSS
        elem = [array[0], array[1], array[5], array[6], array[7]]
        i = (np.linspace((2. * np.pi) / float(len(elem)), (2 * np.pi), len(elem)).tolist())
        elem.append(elem[0])
        i.append(i[0])
        plt.scatter(i, elem, label=str(names[radii.index(array)]),
                    color=c[radii.index(array)])
        plt.plot(i, elem, color=c[radii.index(array)], alpha=0.7)
        plt.fill_between(i, elem, facecolor=c[radii.index(array)], alpha=0.3)
    plt.xticks(i, [score_names[0], score_names[1], score_names[5], score_names[6], score_names[7]])
    plt.yticks( np.linspace(0, 1 , 10) , [])
    plt.ylim([0,1])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='best', bbox_to_anchor=(1.1, 0.8))
    print(forecast_names)
    print(total)
    print(areas)
    plt.show()

#=======================================================================================================================

def roc_plot_aisling():
    """
    Copied from Aisling Bergin's roc.py
    Uses following functions: contingency_table, POD, FAR, and roc_area
    """
    # stuff that aisling imports
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    # open aisling's file
    with open('/Users/somurray/Dropbox/tcd/students/ss_astro_projects/scoreboard/Code_Upload/binary_report.csv', 'r') as f:
        reader = csv.reader(f)
        report = list(reader)
        #I think 0 is the date, then it goes probability, yes/no...
    number = [1, 3, 5, 7, 9, 11, 13, 15, 17] #only the probabilities
    name = ['AMOS', 'ASAP', 'ASSA', 'BoM', 'MAG4', 'MOSWOC', 'NOAA', 'SIDC', 'SOLMON']
    areas = []
    for i in number:
        inc = 0.1 #bins
        POD_array, FAR_array = [], []
        for thresh in list(np.arange(0, 1.1, inc)):
            TP, FN, FP, TN = contingency_table_aisling(report, i, thresh)
            POD_array.append(POD(TP, FN, FP, TN))
            FAR_array.append(FAR(TP, FN, FP, TN))
            total = TP + FN + FP + TN
        # shaded area
        plt.fill_between(FAR_array, np.full(len(FAR_array), 0), POD_array, color='lightgrey')
        # points
        plt.scatter(FAR_array, POD_array)
        # line
        plt.plot(FAR_array, POD_array, label=str(name[number.index(i)]))
        # range from 0 to 1
        plt.axis([0, 1, 0, 1])
        # no skill diagonal
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=0.7)
        # titles
        plt.xlabel('False Alarm Ratio')
        plt.ylabel('Probability of Detection')
        plt.title('Relative Operating Characteristic Curve: ' + str(name[number.index(i)]))
#        plt.savefig('roc_' + str(name[number.index(i)]) + '.png')
        # calculate area
        area = roc_area(FAR_array, POD_array, str(name[number.index(i)]))
        areas.append(area)

def contingency_table_aisling(report, i, thresh):
    """

    :param report:
    :param i:
    :param thresh:
    :return:
    """
    TP, FN, FP, TN = [], [], [], []
    date, prob, obs = report[0], report[i], report[i + 1]
    for elem in date:
        if np.isfinite(float(prob[date.index(elem)])):
            if float(prob[date.index(elem)]) >= float(thresh):
                if float(obs[date.index(elem)]) == 1:
                    TP.append(elem)
                if float(obs[date.index(elem)]) == 0:
                    FP.append(elem)
            else:
                if float(obs[date.index(elem)]) == 1:
                    FN.append(elem)
                if float(obs[date.index(elem)]) == 0:
                    TN.append(elem)
    [TP, FN, FP, TN] = [float(len(elem)) for elem in [TP, FN, FP, TN]]
    return TP, FN, FP, TN


def roc_plot(forecast_time, forecast_probability, observed_yesno):
    """
    Copied from Aisling Bergin's roc.py
    Uses following functions: contingency_table, POD, FAR, and roc_area
    Dont think it needs date - to be removed
    """
    inc = 0.1 # bins
    POD_array, FAR_array = [], []
    for thresh in list(np.arange(0, 1.1, inc)):
        TP, FN, FP, TN = contingency_table(forecast_time, forecast_probability, observed_yesno, thresh)
        POD_array.append(POD(TP, FN, FP, TN))
        FAR_array.append(FAR(TP, FN, FP, TN))
        total = TP + FN + FP + TN
    # shaded area
    plt.fill_between(FAR_array, np.full(len(FAR_array), 0), POD_array, color='lightgrey')
    # points
    plt.scatter(FAR_array, POD_array)
    # line
    plt.plot(FAR_array, POD_array)
    # range from 0 to 1
    plt.axis([0, 1, 0, 1])
    # no skill diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=0.7)
    # titles
    plt.xlabel('False Alarm Ratio')
    plt.ylabel('Probability of Detection')
    plt.title('Relative Operating Characteristic Curve')
#    plt.savefig('roc_curve.png')
    # calculate area
    area = roc_area(FAR_array, POD_array)
    return area

def contingency_table(forecast_time, forecast_probability, observed_yesno, thresh):
    """
    :param date: TO BE REMOVED?
    :param prob:
    :param obs:
    :param thresh:
    :return:
    """
    TP, FN, FP, TN = [], [], [], []
    for elem in forecast_time:
        if np.isfinite(float(forecast_probability[forecast_time.index(elem)])):
            if float(forecast_probability[forecast_time.index(elem)]) >= float(thresh):
                if float(observed_yesno[forecast_time.index(elem)]) == 1:
                    TP.append(elem)
                if float(observed_yesno[forecast_time.index(elem)]) == 0:
                    FP.append(elem)
            else:
                if float(observed_yesno[forecast_time.index(elem)]) == 1:
                    FN.append(elem)
                if float(observed_yesno[forecast_time.index(elem)]) == 0:
                    TN.append(elem)
    [TP, FN, FP, TN] = [float(len(elem)) for elem in [TP, FN, FP, TN]]
    return TP, FN, FP, TN

def POD(TP, FN, FP, TN):
    """
    :param TP: no. of true positives
    :param FN: no. of false negatives
    :param FP: no. of false positives
    :param TN: no. of true negatives
    :return: probability of detection
    """
    POD = TP / (TP + FN)
    return POD


def FAR(TP, FN, FP, TN):
    """
    :param TP: no. of true positives
    :param FN: no. of false negatives
    :param FP: no. of false positives
    :param TN: no. of true negatives
    :return: false alarm ratio
    """
    FAR = FP / (FP + TN)
    return FAR


def roc_area(FAR, POD):
    """
    :param FAR: false alarm ratio
    :param POD: probability of detection
    :return:
    """
    from shapely import geometry

    corners_x, corners_y, corners = [], [], []

    for elem in FAR:
        x, y = float(elem), float(POD[FAR.index(elem)])
        corners_x.append(x)  # add point to points in shape
        corners_y.append(y)
        corners.append([x, y])

    corners_x.append(0.)  # add 0,0 to shape
    corners_y.append(0.)
    corners.append([0., 0.])

    corners_x.append(1.)  # add 1,0 to shape
    corners_y.append(0.)
    corners.append([1., 0.])

    poly = geometry.Polygon(corners)
#    x, y = poly.exterior.xy
#    plt.plot(x, y)

    return poly.area

#=======================================================================================================================

def reliability_plot_aisling():
    """Copied from Aisling Bergin's reliability.py
    Uses the following functions: binning, sample_climatology, plot_frills
    Create a reliability plot for the forecasts.
    Import the data; bin the respective probabilities and for every event aligned with a probability
    in a bin assign one to the observational frequency corresponding to the bin.
    """
    import csv, math
    import numpy as np
    import matplotlib.pyplot as plt
    from area import reliability
    from area import resolution
    import scipy.special as special
    # open aisling's file
    with open('/Users/somurray/Dropbox/tcd/students/ss_astro_projects/scoreboard/Code_Upload/binary_report.csv', 'r') as f:
        reader = csv.reader(f)
        report = list(reader)
    number = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    name = ['AMOS', 'ASAP', 'ASSA', 'BoM', 'MAG4', 'MOSWOC', 'NOAA', 'SIDC', 'SOLMON']
    inc = 0.15
    perc = list(np.arange(0, 1.1, inc))
    for i in number:
        rank_ave, obs, rank, forecasts, events = binning_aisling(report, i, perc)
        no_res, num_events, num_forecasts = sample_climatology_aisling(report, i)
        # set up plot
        a = plt.axes()
        # zone of skill, climatology line
        fig = plot_frills(no_res)
#        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        # points
        plt.scatter(rank_ave, obs)#, color='red')
        # line
        plt.plot(rank_ave, obs)#, color='red')
        # title
        plt.title('Reliability Diagram:' + str(name[number.index(i)]))
        # histogram
        a = plt.axes([.25, .65, .2, .2], facecolor='white')
        width = inc - (inc / 10)
        plt.bar(rank, forecasts, width, color = 'grey', edgecolor = 'black')
        plt.ylabel('#forecasts')

def reliability_plot(forecast_time, forecast_probability, observed_yesno):
    """Copied from Aisling Bergin's reliability.py
    Uses the following functions: binning, sample_climatology, plot_frills
    Create a reliability plot for the forecasts.
    Import the data; bin the respective probabilities and for every event aligned with a probability
    in a bin assign one to the observational frequency corresponding to the bin.
    """
    inc = 0.15
    perc = list(np.arange(0, 1.1, inc))
    rank_ave, obs, rank, forecasts, events = binning(forecast_time, forecast_probability, observed_yesno, perc)
    no_res, num_events, num_forecasts = sample_climatology(forecast_time, forecast_probability, observed_yesno)
    # set up plot
    a = plt.axes()
    # zone of skill, climatology line
    fig = plot_frills(no_res)
#    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    # points
    plt.scatter(rank_ave, obs)#, color='red')
    # line
    plt.plot(rank_ave, obs)#, color='red')
    # title
    plt.title('Reliability Diagram')
    # histogram
    a = plt.axes([.25, .65, .2, .2], facecolor='white')
    width = inc - (inc / 10)
    plt.bar(rank, forecasts, width, color = 'grey', edgecolor = 'black')
    plt.ylabel('#forecasts')

def binning(forecast_time, forecast_probability, observed_yesno, perc):
    """

    :param report:
    :param i:
    :param perc:
    :return:
    """

#    # Find the total number of forecasts in a data set without nans
#    number_forecasts = []
#    for i in prob:
#        if np.isfinite(i):
#            number_forecasts.append(i)
#    number_forecasts = float(len(number_forecasts))

    # Find the number of forecasts, number of events and number of hits in each threshold for a data set
    thresh, events, forecasts = [[] for x in range(len(perc))], [[] for x in range(len(perc))], [[] for x in range(len(perc))]

    for elem in forecast_time:
        prob = float(forecast_probability[(forecast_time.index(elem))])  # probability
        event = float(observed_yesno[(forecast_time.index(elem))])  # 1/0 event result
        if np.isfinite(prob):
            for category in perc:
                if prob >= category and prob < perc[perc.index(category) + 1]:
                    forecasts[perc.index(category)].append(prob)  # append forecasts for each one in bin
                    if event == 1.0:
                        events[perc.index(category)].append(elem)  # append events for every '1' in bin

                if category == perc[-1]:
                    break

    n_k = [float(len(elem)) for elem in forecasts]
    ranks = [float(np.mean(elem)) for elem in forecasts]
    events = [float(len(elem)) for elem in events]
    forecasts = [float(len(elem)) for elem in forecasts]

    term, o_k, rank_ave = [], [], []

    for elem in perc:
        if n_k[perc.index(elem)] == 0.:
            o = 0
            o_k.append(o)
            term.append(0)
        else:
            o = (events[perc.index(elem)]) / n_k[perc.index(elem)]
            o_k.append(o)
            term.append(n_k[perc.index(elem)] * ((elem - o) ** 2))
        rank_ave.append(ranks[perc.index(elem)])

    rank = perc

    return rank_ave, o_k, rank, forecasts, events

def binning_aisling(report, i, perc):
    """

    :param report:
    :param i:
    :param perc:
    :return:
    """
    thresh, events, forecasts = [[] for x in range(len(perc))], [[] for x in range(len(perc))], [[] for x in range(len(perc))]

    # Find the total number of forecasts in a data set
    N = []
    for elem in report[0]:
        prob = float(report[i][(report[0].index(elem))])
        if np.isfinite(prob):
            N.append(prob)
    N = float(len(N))

    # Find the number of forecasts, number of events and number of hits in each threshold for a data set

    for elem in report[0]:
        prob = float(report[i][(report[0].index(elem))])  # probability
        event = float(report[i + 1][(report[0].index(elem))])  # 1/0 event result
        if np.isfinite(prob):
            for category in perc:
                if prob >= category and prob < perc[perc.index(category) + 1]:
                    forecasts[perc.index(category)].append(prob)  # append forecasts for each one in bin
                    if event == 1.0:
                        events[perc.index(category)].append(elem)  # append events for every '1' in bin

                if category == perc[-1]:
                    break

    n_k = [float(len(elem)) for elem in forecasts]
    ranks = [float(np.mean(elem)) for elem in forecasts]
    events = [float(len(elem)) for elem in events]
    p_k = perc

    term, o_k, rank_ave = [], [], []

    for elem in perc:
        if n_k[perc.index(elem)] == 0.:
            o = 0
            o_k.append(o)
            term.append(0)
        else:
            o = (events[perc.index(elem)]) / n_k[perc.index(elem)]
            o_k.append(o)
            term.append(n_k[perc.index(elem)] * ((elem - o) ** 2))
        rank_ave.append(ranks[perc.index(elem)])

    rank = perc
    forecasts = [float(len(elem)) for elem in forecasts]

    return rank_ave, o_k, rank, forecasts, events


def sample_climatology(forecast_time, forecast_probability, observed_yesno):
    """
    counting number events, non events, etc
    :param report:
    :param i:
    :return:
    """
    num_forecasts = 0
    num_events = 0

    for elem in forecast_time:

        if np.isfinite(float(forecast_probability[(forecast_time.index(elem))])):
            num_forecasts = num_forecasts + 1

        if float(observed_yesno[(forecast_time.index(elem))]) == 1:
            num_events = num_events + 1

    no_res = float(num_events) / float(num_forecasts)

    return no_res, num_events, num_forecasts

def sample_climatology_aisling(report, i):
    """
    counting number events, non events, etc
    :param report:
    :param i:
    :return:
    """
    num_forecasts = 0
    num_events = 0

    for elem in report[0]:

        if np.isfinite(float(report[i][(report[0].index(elem))])):
            num_forecasts = num_forecasts + 1

        if float(report[i + 1][(report[0].index(elem))]) == 1:
            num_events = num_events + 1

    no_res = float(num_events) / float(num_forecasts)

    return no_res, num_events, num_forecasts


def plot_frills(no_res):
    """

    :param no_res:
    :return:
    """
    x = np.arange(0, 1.1, 0.1)

    horiz_clim = [no_res for elem in x]

    vert_clim = [no_res for elem in x]

    no_skill = [(elem + no_res) / 2 for elem in x]

    perfect = [elem for elem in x]

    plt.fill_betweenx(x, vert_clim, x, color='lightgrey')
    plt.fill_between(x, x, no_skill, color='lightgrey')

    plt.plot(x, horiz_clim, linestyle=':', color='grey', linewidth=1.)  ## climatology-horizontal
    plt.plot(vert_clim, x, linestyle=':', color='grey', linewidth=1.)  # climatology-vertical
    plt.plot(x, no_skill, linestyle='--', color='grey', linewidth=0.5)  # no skill line
    plt.plot(x, perfect, color='grey', linewidth=0.5)  # perfect reliability

    plt.axis([0, 1, 0, 1])
    plt.xlabel('Forecast Probability')
    plt.ylabel('Observed Frequency')

#=======================================================================================================================


def probabilistic_metrics():
    """

    :return:
    """
    import csv
    with open('/Users/somurray/Dropbox/tcd/students/ss_astro_projects/scoreboard/Code_Upload/binary_report.csv', 'r') as f:
        reader = csv.reader(f)
        report = list(reader)

    with open('/Users/somurray/Dropbox/tcd/students/ss_astro_projects/scoreboard/Code_Upload/climatology.csv', 'r') as f:
        reader = csv.reader(f)
        climatology = list(reader)

    number = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    perc = list(np.arange(0, 1.1, 0.1))
    for i in number:
        #  Brier score
        reliability = brier_reliability(report, i, perc)
        resolution = brier_resolution(report, i, perc)
        total, events, non_events = forecast_stats(report, i)
        uncertainty = (float(events) / float(total)) * (1. - (float(events) / float(total)))
        brier_score = reliability - resolution + uncertainty
        brier_skill_score = (resolution - reliability) / uncertainty

        # RPSS
        clim_reliability = brier_reliability(climatology, 1, perc)
        clim_resolution = brier_resolution(climatology, 1, perc)
        clim_total, clim_events, clim_non_events = forecast_stats(climatology, 1)
        clim_uncertainty = (float(clim_events) / float(clim_total)) * (1. - (float(clim_events) / float(clim_total)))
        clim_brier_score = clim_reliability - clim_resolution + clim_uncertainty
        RPSS = 1 - (float(brier_score) / float(clim_brier_score))


def brier_reliability(report, i, perc):
    thresh, events, forecasts = [[] for x in range(len(perc))],  [[] for x in range(len(perc))], [[] for x in range(len(perc))]
    # Find the total number of forecasts in a data set
    N = []
    for elem in report[0]:
        prob = float(report[i][(report[0].index(elem))])
        if np.isfinite(prob):
            N.append(prob)
    N = float(len(N))
    # Find the number of forecasts, number of events and number of hits in each threshold for a data set
    for elem in report[0]:
        prob = float(report[i][(report[0].index(elem))]) # probability
        event = float(report[i+1][(report[0].index(elem))]) # 1/0 event result
        if np.isfinite(prob):
            for category in perc:
                if prob >= category and prob < perc[perc.index(category)+1]:
                    forecasts[perc.index(category)].append(prob)  	# append forecasts for each one in bin
                    if event == 1.0:
                        events[perc.index(category)].append(elem)   # append events for every '1' in bin
                if category == perc[-1]:
                    break
    n_k, events, p_k  = [float(len(elem)) for elem in forecasts], [float(len(elem)) for elem in events],  perc
    term, o_k = [], []
    for elem in perc:
        if n_k[perc.index(elem)] == 0.:
            o = 0
            o_k.append(o)
            term.append(0)
        else:
            o = (events[perc.index(elem)])/n_k[perc.index(elem)]
            o_k.append(o)
            term.append(n_k[perc.index(elem)]*((elem - o)**2))
    reliability = (1./N)*np.sum(term)
    return reliability


def brier_resolution(report, i, perc):
    thresh, events, forecasts = [[] for x in range(len(perc))], [[] for x in range(len(perc))], [[] for x in
                                                                                                 range(len(perc))]
    # Find the total number of forecasts in a data set
    N = []
    for elem in report[0]:
        prob = float(report[i][(report[0].index(elem))])
        if np.isfinite(prob):
            N.append(prob)
    N = float(len(N))
    # Find the number of forecasts, number of events and number of hits in each threshold for a data set
    for elem in report[0]:
        prob = float(report[i][(report[0].index(elem))])  # probability
        event = float(report[i + 1][(report[0].index(elem))])  # 1/0 event result
        if np.isfinite(prob):
            for category in perc:
                if prob >= category and prob < perc[perc.index(category) + 1]:
                    forecasts[perc.index(category)].append(prob)  # append forecasts for each one in bin
                    if event == 1.0:
                        events[perc.index(category)].append(elem)  # append events for every '1' in bin
                if category == perc[-1]:
                    break

    n_k, events, p_k = [float(len(elem)) for elem in forecasts], [float(len(elem)) for elem in events], perc
    climatology = float(np.sum(events)) / float(N)
    term, o_k = [], []
    for elem in perc:
        if n_k[perc.index(elem)] == 0.:
            o = 0
            o_k.append(o)
            term.append(0)
        else:
            o = (events[perc.index(elem)]) / n_k[perc.index(elem)]
            o_k.append(o)
            term.append(n_k[perc.index(elem)] * ((o - climatology) ** 2))
    resolution = (1. / N) * np.sum(term)
    return resolution

def forecast_stats(report, i):
    # Find the total number of forecasts and events in a data set
    N = []
    events = []
    for elem in report[0]:
        prob = float(report[i][(report[0].index(elem))])
        event = float(report[i + 1][(report[0].index(elem))])  # 1/0 event result

        if np.isfinite(prob):
            N.append(prob)
            if event == 1.0:
                events.append(elem)  # append events for every '1' in bin

    N = float(len(N))
    events = float(len(events))
    return N, events, (N - events)


#=======================================================================================================================

def categorical_metrics():
    """

    :return:
    """
    # TSS (H&KSS) and HSS
    with open('/Users/somurray/Dropbox/tcd/students/ss_astro_projects/scoreboard/Code_Upload/binary_report.csv', 'r') as f:
        reader = csv.reader(f)
        report = list(reader)
    number = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    perc = list(np.arange(0, 1.1, 0.1))

    for i in number:
        TSS = []
        HSS = []
        for thresh in perc:
            TP, FN, FP, TN = contingency_table(report, i, thresh)
            if FP == 0:
                continue
            this_TSS = ((TP * TN) - (FP * FN)) / ((TP + FN) * (FP + TN))
            TSS.append(this_TSS)
            this_HSS = (2 * ((TP * TN) - (FP * FN))) / (((TP + FN) * (FN + TN)) + ((TP + FP) * (FP + TN)))
            HSS.append(this_HSS)
        TSS = max([abs(elem) for elem in TSS])
        HSS = max([abs(elem) for elem in HSS])
#    return TSS, HSS

