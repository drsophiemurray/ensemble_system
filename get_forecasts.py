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
"""

import requests
import datetime

iswa_fulldisk_forecast_products = {"ASSA_1_FULLDISK", "ASSA_24H_1_FULLDISK",
                                    "AMOS_v1_FULLDISK", "BoM_flare1_FULLDISK",
                                    "MO_TOT1_FULLDISK", "NOAA_1_FULLDISK",
                                    "SIDC_Operator_FULLDISK"}
m_plus_forecasts = ["AMOS_v1_FULLDISK","BoM_flare1_FULLDISK", "SIDC_Operator_FULLDISK"]
iswa_data_link = "https://iswa.gsfc.nasa.gov/IswaSystemWebApp/flarescoreboard/hapi/data"

def main():
    """
    Grabbing the latest forecasts currently available on ISWA and putting them in a pandas database
    For ISWA, values are in 'data' and descriptions are in 'parameters'. No data is '-1'.
    """
    time_now = datetime.datetime.utcnow()
    time_start = (time_now-datetime.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S.%f')
    time_end = time_now.strftime('%Y-%m-%dT%H:%M:%S.%f')
    for product in iswa_fulldisk_forecast_products:
        selection = {"id":product,
                     "time.min":'2016-09-05T00:00:00.0',
                     "time.max":'2016-09-07T00:00:00.0',
                     "format":"json",
                     "options":"fields.all"}
        response = requests.get(iswa_data_link, params=selection)
        if response.status_code == 200:
            data = response.json()
            if data['data']:
                forecast_time = data['data'][0][0]
                if any(product==forecast for forecast in m_plus_forecasts):
                    m_prob = data['data'][0][6]
                else:
                    m_prob = data['data'][0][4]
                x_prob = data['data'][0][7]
                print(product,forecast_time,m_prob,x_prob)





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

