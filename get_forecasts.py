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
"""

import requests
import datetime

ISWA_FULLDISK_PRODUCTS = {"ASSA_1_FULLDISK", "ASSA_24H_1_FULLDISK",
                          "AMOS_v1_FULLDISK", "BoM_flare1_FULLDISK",
                          "MO_TOT1_FULLDISK", "NOAA_1_FULLDISK",
                          "SIDC_Operator_FULLDISK", "UFCORIN_1_FULLDISK"}
M_PLUS_FORECASTS = ["AMOS_v1_FULLDISK","BoM_flare1_FULLDISK",
                    "SIDC_Operator_FULLDISK", "UFCORIN_1_FULLDISK"]
ISWA_DATA_LINK = "https://iswa.gsfc.nasa.gov/IswaSystemWebApp/flarescoreboard/hapi/data"

def main():
    """
    Grabbing the latest forecasts currently available on ISWA and putting them in a pandas database
    For ISWA, values are in 'data' and descriptions are in 'parameters'. No data is '-1'.
    """
    time_now = datetime.datetime.utcnow()
    #TODO always take midnight? means dropping SIDC
    time_start = (time_now-datetime.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S.%f')
    time_end = time_now.strftime('%Y-%m-%dT%H:%M:%S.%f')
    for product in ISWA_FULLDISK_PRODUCTS:
        #TODO remove hardcoded date when all available and replace with time_start and time_end for realtime
        selection = {"id":product,
                     "time.min":'2016-09-04T23:50:00.0',
                     "time.max":'2016-09-06T00:11:00.0',
                     "format":"json",
                     "options":"fields.all"}
        response = requests.get(ISWA_DATA_LINK , params=selection)
        if response.status_code == 200:
            data = response.json()
            if data['data']:
                # first yesterdays forecast for verification
                get_iswa_forecasts(product, data, day=0)
                # now today for the ensemble forecast
                get_iswa_forecasts(product, data, day=len(data['data'])-1)


def get_iswa_forecasts(product, data, day):
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
        m_prob = data['data'][day][6]
    else:
        m_prob = data['data'][day][4]
    # x forecast same for 'plus' or 'only'
    x_prob = data['data'][day][7]
    # at the moment just printing out to test, TODO create a pandas database
    print(product, forecast_time, m_prob, x_prob)
    return


if __name__ == '__main__':
    main()






#Notes
#get rid of u: encode('ascii', 'ignore') or just str()
#or:
#str(data['data'][0][0])
#or:
#>>> import json, ast
#>>> r = {u'name': u'A', u'primary_key': 1}
#>>> ast.literal_eval(json.dumps(r))
#{'name': 'A', 'primary_key': 1}

# equivalent of https://iswa.gsfc.nasa.gov/IswaSystemWebApp/flarescoreboard/hapi/data?id=NOAA_1_FULLDISK&time.min=2017-09-05T00:00:00.0&time.max=2017-09-10T00:00:00.0&format=json&options=fields.all
# fields.all:
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
#    data['data']
#    data['parameters']

#selection = {"id":"NOAA_1_FULLDISK",
#                  "time.min":"2016-09-05T00:00:00.0",
#                  "time.max":"2016-09-07T00:00:00.0",
#                  "format":"json",
#                  "options":"fields.all"}
#                  "parameters":"start_window,end_window,issue_time,M,X"}

#        print(response.content)

