"""
Testing pulling in all the FLARECAST ML forecasts
(even though real time currently doesnt work)
"""
import requests

def main():
    prediction_service = "api.flarecast.eu/prediction"
    fc_id = "prediction-00000000-0000-0000-0000-000000290b20"
    url = "http://%s/prediction/data?prediction_fc_id=%s" % (prediction_service, fc_id)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    print(data)

if __name__ == '__main__':
    main()
