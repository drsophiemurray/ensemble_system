"""Here I plan to pull in full disk forecasts
In ideal world FLARECAST would work, but nope so will use Scoreboard
Just get whatever raw data there is for M and X flares
-most recent forecast since last run?
"""

fulldisk_forecast_products = [["ASSA_1_FULLDISK", "ASSA_24H_1_FULLDISK", "AMOS_v1_FULLDISK",
                               "BoM_flare1_FULLDISK", "MO_TOT1_FULLDISK", "NOAA_1_FULLDISK", "SIDC_Operator_FULLDISK"]]


scoreboard_data_link = "https://iswa.gsfc.nasa.gov/IswaSystemWebApp/flarescoreboard/hapi/data"
selection = {"id":"NOAA_1_FULLDISK",
             "time.min":"2016-09-05T00:00:00.0",
             "time.max":"2016-09-07T00:00:00.0",
             "format":"json",
             "options":"fields.all"}
#             "parameters":"start_window,end_window,issue_time,M,X"}
response = requests.get(scoreboard_data_link, params=selection)
#print(response.content)
data = response.json()

data['data']
data['parameters']
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




