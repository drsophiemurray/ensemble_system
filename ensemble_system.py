"""
This will be the main wrapper
"""

import get_iswa_forecasts

def main():
    """
    Grab forecasts, create an ensemble, verify
    """
    iswa_yday, iswa_today = get_iswa_forecasts.main()
    # Get average
    average_iswa_today_m = np.average(iswa_today['x_prob'])
    average_iswa_today_x = np.average(iswa_today['m_prob'])
    # Get weighted ensemble

    # Verify yesterdays forecast



if __name__ == '__main__':
    main()
