"""
This will be the main wrapper.
Each code self-contained and doesnt have to be used with the ensemble as long as in correct data format.
All in pandas databases.
"""
import get_observations
import get_iswa_forecasts
import numpy as np
import rolling_climatology

def main():
    """
    Grab forecasts.
    Create an average ensemble, then weighted ensemble using defined properties,
    Grab latest data.
    Verify.
    """
    # Grab forecasts
    iswa_yday, iswa_today = get_iswa_forecasts.main()
    print(iswa_today)

    # Get average
    average_iswa_today_m = np.average(iswa_today['m_prob'])
    average_iswa_today_x = np.average(iswa_today['x_prob'])

    # Get weighted ensemble
#    ensemble_out = ensemble(iswa_today,
#                            type='probabilistic',
#                            metric='brier',
#                            uncertainty=True)
    # Write out to file

    # Get observations
    obs_list = get_observations.main('20160723')
    print(obs_list)

    # Get 120-day climatology (maybe above get last 120 days?) not sure
    clim_m, clim_x = rolling_climatology()

    # Run through verification
#    verify(forecasts=iswa_yday, events=obs_list,
#           visual=True)


if __name__ == '__main__':
    main()
