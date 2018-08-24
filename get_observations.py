"""
This will grab latest list of flare events.
Take from code i wrote at met office..
-- inspired by https://github.com/drsophiemurray/flare_verification/blob/master/get_obs.py

Todays: http://services.swpc.noaa.gov/text/solar-geophysical-event-reports.txt
or: ftp://ftp.swpc.noaa.gov/pub/indices/events/events.txt
ysterday: ftp://ftp.swpc.noaa.gov/pub/indices/events/yesterday.txt
anytime ftp://ftp.swpc.noaa.gov/pub/indices/events/20180711events.txt
"""


import ftplib
import os

def main(*date):
    """
    Get flare events for a particular date
    date should be a string in format 'YYYYMMDD'
    """

    if date:
        file = date[0] + 'events.txt' #zeroth because reads in as tuple!
    else:
        file = 'yesterday.txt'
    flare_list = get_list(file)
    if flare_list:
        mx_list = extract_flares(flare_list)
        return mx_list
    else:
        print('No events found')
        return []


def get_list(file):
    """
    call ftp, download file, the extract list
    """
    #ftp
    ftp = ftplib.FTP('ftp.swpc.noaa.gov')
    ftp.login()
    ftp.cwd('pub/indices/events/')
    ftp.retrbinary('RETR '+file, open(os.getcwd() + '/' + file, "wb").write)
    ftp.close()
    #downloaded file
    flare_list = []
    with open(file, "r") as inp:
        for line in inp:
            if "XRA" in line:
                flare_list.append(line[0:80])
    os.remove(file)
    return flare_list


def extract_flares(flare_list):
    """
    split out the text to get the flare magnitudes
    - only want m and x flares
    Returns a list like
    # Start Peak End Magnitude Region
    """
    mx_list = []
    for line in flare_list:
        split_line = line.split()
        if '+' in split_line:
            if 'M' in split_line[9] or 'X' in split_line[9]:
                if len(split_line) >=15 :
                    mx = (split_line[2] + ' ' + split_line[3] + ' ' + split_line[4] + ' ' +
                          split_line[9] + ' ' + split_line[11])
                    mx_list.append(mx)
                else:
                    mx = (split_line[2] + ' ' + split_line[3] + ' ' + split_line[4] + ' ' +
                          split_line[9] + ' ' + 'NaN')
                    mx_list.append(mx)
        else:
            if 'M' in split_line[8] or 'X' in split_line[8]:
                if len(split_line) >=14 :
                    mx = (split_line[1] + ' ' + split_line[2] + ' ' + split_line[3] + ' ' +
                          split_line[8] + ' ' + split_line[10])
                    mx_list.append(mx)
                else:
                    mx = (split_line[1] + ' ' + split_line[2] + ' ' + split_line[3] + ' ' +
                          split_line[8] + ' ' + 'NaN')
                    mx_list.append(mx)
    return mx_list

if __name__ == '__main__':
    main()

