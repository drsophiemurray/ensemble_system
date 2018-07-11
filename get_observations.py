"""
This will grab latest list of flare events.
Take from code i wrote at met office..

Todays: http://services.swpc.noaa.gov/text/solar-geophysical-event-reports.txt
or: ftp://ftp.swpc.noaa.gov/pub/indices/events/events.txt
ysterday: ftp://ftp.swpc.noaa.gov/pub/indices/events/yesterday.txt
anytime ftp://ftp.swpc.noaa.gov/pub/indices/events/20180711events.txt
"""


import ftplib
import os

def main(date):
    """
    Get flare events for a particular date
    date should be a string in format 'YYYYMMDD'
    """
    if date:
        file = date + 'events.txt'
    else:
        file = 'yesterday.txt'
    flare_list = get_list(file)
    print(flare_list)
    if flare_list:
        print('Yep')
    else:
        print('Nada')


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



if __name__ == '__main__':
    main()

