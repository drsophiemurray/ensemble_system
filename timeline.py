"""
This is the code Aisling used to show timelines of flare foreacsts and corresponding events.
Might come in useful at some point so just archiving as is without fixing up anything


"""

import datetime as dt
import numpy as np
import os, time, csv, pandas, math, scipy
from fun_test1 import forecast_results
import matplotlib.pyplot as plt
from flare_count import flare_count

from matplotlib import style

style.use('seaborn-whitegrid')


def forecasts():
    with open('binary_report.csv', 'r') as f:
        reader = csv.reader(f)
        report = list(reader)

    c = ['forestgreen','yellowgreen','orange','darkorange','orangered','maroon','darkslateblue','royalblue','steelblue']

    number = [1,3,5,7,9,11,13,15,17]#,21,23]
    name = ['AMOS','ASAP','ASSA','BoM','MAG4','MOSWOC','NOAA','SIDC','SOLMON']#, 'Average', 'Ensemble']

    date = [(dt.datetime.strptime(elem, '%Y-%m-%d %H:%M:%S')) for elem in report[0]]

    for elem in number:
        x, y = [], []
        print(name[number.index(elem)])
        for day in date:
            if math.isnan(float(report[elem][date.index(day)])) == False:
                x.append(day)
                y.append(elem)
        plt.scatter(x,y,color= c[number.index(elem)], s = 1.)

    #plt.title('Timeline Comparison of Forecast Data')

    #plt.gcf().autofmt_xdate()

    # Label the y-axis
    y_axis = number
    plt.yticks(y_axis, name)


def flares():
    # Label the flare count portion of y-axis
    plt.ylabel('Observed Flare Count')
    plt.xlabel('Date')
    # plt.title('NOAA Observed Solar Flare Count')

    path = '/home/aisling/Flare_Scoreboard/flare_data'

    num_m, date_m, num_x, date_x, m_class, m_times, x_class, x_times = flare_count(path)

    plt.plot(date_m, num_m, '-', label='M Flares', color='grey')
    plt.plot(date_x, num_x, '-', label='X Flares', color='red')

    print
    np.sum(num_x), 'X flares in ', len(date_x), 'days'

    print
    float(np.sum(num_m[365:]) + np.sum(num_x[365:])) / float(np.sum(num_m) + np.sum(num_x))

    # plt.xlim([dt.datetime(2014, 12, 31, 00, 00), dt.datetime(2017, 10, 15, 00, 00)])

    # plt.gcf().autofmt_xdate()

    plt.legend(loc='best')

    plt.show()






