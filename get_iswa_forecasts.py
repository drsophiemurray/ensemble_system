"""
Here I plan to pull in full disk forecasts
In ideal world FLARECAST would work, but nope so will use Scoreboard
Just get whatever raw data there is for M and X flares
-most recent forecast since last run?

Steps:
- get current time
- grab forecasts for each iswa product
- put in database
- next codes: average, verify yesterday

Note:
- UFCORIN has not been included as too different to combine
- SIDC has not been included as issued at midday for 12 hours so cannot combine easily
- Below code is equivalent of something like:
    https://iswa.gsfc.nasa.gov/IswaSystemWebApp/flarescoreboard/hapi/data?id=NOAA_1_FULLDISK&time.min=2017-09-05T00:00:00.0&time.max=2017-09-10T00:00:00.0&format=json&options=fields.all
- All data fields are:
    # 0: start_window
    # 1: end_window
    # 2: issue_time
    # 3: C
    # 4: M
    # 5: CPlus
    # 6: MPlus
    # 7: X
    # 8: C_uncertainty
    # 9: M_uncertainty
    # 10: CPlus_uncertainty
    # 11: MPlus_uncertainty
    # 12: X_uncertainty
    # 13: C_value_lower
    # 14: M_value_lower
    # 15: CPlus_value_lower
    # 16: MPlus_value_lower
    # 17: X_value_lower
    # 18: C_value_higher
    # 19: M_value_higher
    # 20: CPlus_value_higher
    # 21: MPlus_value_higher
    # 22: X_value_higher
    # 23: C_level
    # 24: M_level
    # 25: CPlus_level
    # 26: MPlus_level
    # 27: X_level
"""

import requests
import datetime
import pandas as pd
import numpy as np

ISWA_FULLDISK_PRODUCTS = {"ASSA_1_FULLDISK", "ASSA_24H_1_FULLDISK",
                          "AMOS_v1_FULLDISK", "BoM_flare1_FULLDISK",
                          "MO_TOT1_FULLDISK", "NOAA_1_FULLDISK",
                          "SIDC_Operator_FULLDISK", "UFCORIN_1_FULLDISK"}
M_PLUS_FORECASTS = ["AMOS_v1_FULLDISK", "BoM_flare1_FULLDISK",
                    "SIDC_Operator_FULLDISK", "UFCORIN_1_FULLDISK"]
ISWA_DATA_LINK = "https://iswa.gsfc.nasa.gov/IswaSystemWebApp/flarescoreboard/hapi/data"

def main(*date):
    """
    Grabbing the latest forecasts currently available on ISWA and putting them in a pandas database
    For ISWA, values are in 'data' and descriptions are in 'parameters'. No data is '-1'.

    Input Parameters
    ----------------
    date: datetime object.
            Will be used as an end time for searching, with start time 24hours previous.
    Output Parameters
    -----------------
    yesterdays_forecast_data: pandas database.
                                All forecast data obtained 24hours previous.
    todays_forecast_data: pandas database.
                                All forecast data on defined date (default currrent or optional specifed)
    """
    if date:
        time_start = (date-datetime.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S.%f')
        time_end = date.strftime('%Y-%m-%dT%H:%M:%S.%f')
    else:
        time_now = datetime.datetime.utcnow()
        #TODO always take midnight? means dropping SIDC
        time_start = (time_now-datetime.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S.%f')
        time_end = time_now.strftime('%Y-%m-%dT%H:%M:%S.%f')
    # Create databases
    todays_forecast_data = pd.DataFrame(columns = ["product", "time",
                                                   "m_prob", "x_prob"])
    yesterdays_forecast_data = todays_forecast_data.copy()
    for product in ISWA_FULLDISK_PRODUCTS:
        #TODO remove hardcoded date when all available and replace with time_start and time_end for realtime
        selection = {"id":product,
                     "time.min":'2016-07-22T23:50:00.0',
                     "time.max":'2016-07-24T00:11:00.0',
                     "format":"json",
                     "options":"fields.all"}
#                     "parameters":"start_window,end_window,issue_time,M,X"}

        response = requests.get(ISWA_DATA_LINK ,
                                params=selection)
#        print(response.content)

        if response.status_code == 200:
            data = response.json()
            if data['data']:
                # first yesterdays forecast for verification
                yesterdays_forecast = grab_forecasts(product, data,
                                                     day=0)
                # now today for the ensemble forecast
                todays_forecast = grab_forecasts(product, data,
                                                 day=len(data['data'])-1)
                # append to the pandas databases
                todays_forecast_data = todays_forecast_data.append([todays_forecast],
                                                                   ignore_index=True)
                yesterdays_forecast_data = yesterdays_forecast_data.append([yesterdays_forecast],
                                                                           ignore_index=True)
    return yesterdays_forecast_data, todays_forecast_data


def grab_forecasts(product, data, day):
    """
    Grab M and X forecasts for particular date

    Parameters
    ----------
    product: The forecast product being grabbed
    data: Forecast data taken from API
    day: Time of interest

    Returns
    -------
    forecast: pandas database with forecast probability values
    """
    # get start of time window rather than issue time
    forecast_time = data['data'][day][0]
    # get m plus or m only forecasts
    if any(product == forecast for forecast in M_PLUS_FORECASTS):
        m_prob = np.float(data['data'][day][6])
    else:
        m_prob = np.float(data['data'][day][4])
    # x forecast same for 'plus' or 'only'
    x_prob = np.float(data['data'][day][7])
    # output results
    forecast = {"product":product, "time": forecast_time,
                "m_prob":m_prob, "x_prob":x_prob}
    return forecast

if __name__ == '__main__':
    main()

