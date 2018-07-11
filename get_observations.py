"""
This will grab latest list of flare events.
Take from code i wrote at met office..

Todays: http://services.swpc.noaa.gov/text/solar-geophysical-event-reports.txt
or: ftp://ftp.swpc.noaa.gov/pub/indices/events/events.txt
ysterday: ftp://ftp.swpc.noaa.gov/pub/indices/events/yesterday.txt
anytime ftp://ftp.swpc.noaa.gov/pub/indices/events/20180711events.txt
"""


import datetime
import os

def main(date):
    """
    Get flare events for a particular date
    date should be a string in format 'YYYYMMDD'
    """
    file = date + 'events.txt'
    ftp = ftplib.FTP('ftp.swpc.noaa.gov')
    ftp.login()
    ftp.cwd('pub/indices/events/')
    ftp.retrbinary('RETR '+file, open(os.getcwd() + '/' + file, "wb").write)
    ftp.close()

    flare_list = []
    with open(file, "r") as inp:
        for line in inp:
            if "XRA" in line:
                flare_list.append(line[0:80])
    os.remove(file)



if __name__ == '__main__':
    main()

