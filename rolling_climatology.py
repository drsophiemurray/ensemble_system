"""
Create 120 day rolling climatology based on Sharpe et al 2017.

Code developed by Aisling Bergin (climatology_120.py), adapted by Sophie Murray for ensemble system.

Code currently assumes has a bunch of NOAA event list files in a folder

She only looks at M flares - fix to inlcude X aswell and output as a pandas DF

"""

import os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import csv

#path = '/home/aisling/Desktop/Code_Scoreboard/climatology_data'  		# create path to the directory of files
path = '/Users/somurray/Dropbox/met_office_folders/verification_data/swpc_event_lists/'

def main():
    """


    """
    num_m, date_m, num_x, date_x, m_class, m_times, x_class, x_times = flare_count(path)

    climatology = []
    events = []
    num = 0
    for elem in date_m[120:]:
        previous = date_m.index(elem) - 120
        to_date = date_m.index(elem)
        climatology.append(np.mean(num_m[previous:to_date]))

        if num_m[date_m.index(elem)] > 0:
            events.append(1)
            num += 1
        else:
            events.append(0)

    print(num)

    #datetime structure, probability (decimal), yes/no binary did it happen
    table_m = [date_m[120:], climatology, events]

    return table_m


def flare_count(path):
    m_class, x_class = [], []
    num_m = []
    date_m = []
    num_x = []
    date_x = []
    m_times = []
    x_times = []
    for filename in os.listdir(path):
        fullname = os.path.join(path, filename)
        lines = []
        # dates = []
        date = filename.replace('events.txt', '')
        issuedate = dt.datetime.strptime(date, '%Y%m%d')
        m_counter = 0
        x_counter = 0

        with open(fullname, 'r') as inf:
            for i in range(12):
                next(inf)
            reader = csv.reader(inf, delimiter="	")
            for item in reader:
                if len(item) != 0:
                    clean = item[0].replace("+", "")
                    cols = clean.split()
                    if 'NO' in cols:
                        break

                    lines.append(cols)

                    if 'XRA' in cols[6]:
                        if 'M' in cols[8]:
                            m_class.append(float(cols[8].replace('M', '')))
                            m_counter = m_counter + 1
                            m_times.append(dt.datetime.strptime((str(date) + str(cols[1])), '%Y%m%d%H%M'))

                        elif 'X' in cols[8]:
                            x_class.append(float(cols[8].replace('X', '')))
                            x_counter = x_counter + 1
                            x_times.append(dt.datetime.strptime((str(date) + str(cols[1])), '%Y%m%d%H%M'))

            num_m.append(m_counter)
            date_m.append(issuedate)

            num_x.append(x_counter)
            date_x.append(issuedate)

    date_m, num_m = zip(*sorted(zip(date_m, num_m)))
    date_x, num_x = zip(*sorted(zip(date_x, num_x)))

    return num_m, date_m, num_x, date_x, m_class, m_times, x_class, x_times

if __name__ == '__main__':
    main()
