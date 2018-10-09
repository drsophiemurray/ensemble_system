"""These functions will have come from multiple locations
- FLARECAST verification_module
- Aoife Bergin code
- Sophie's own code.

What I want:
Probabilistic
- ROC plot; area
- Relibility diagram; reliabilty, resolution, uncertainty
- Brier score
- Brier skill score
- RPS
- RPSS

Categorical:
- Truth table stuff like PC POD, POFD, FAR...
- TSS
- HSS

Other:
- Correlation coefficient
- Mean absolute error
-"""



def main():
    """
    Verification engine - ALL the metrics
    Read in the data and depending on type calculate various things
    and/or make plots
    """
    # Lets start with reliability and ROC plots :D



def polar_plot():
    """
    Create a polar plot with the different skill scores calculated for the different forecasts.
    Individual axes plot the forecasts with higher scores at a larger distance from the centre of the plot.
    The largest enclosed area represents the highest scoring forecast overall.
    """
    #stuff that aisling imports
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime as dt
    from shapely import geometry
    # colours for the different areas
    c = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:olive', 'tab:brown', 'tab:orange',  'tab:red', 'tab:purple',  'tab:pink']
    # open aisling's file
    with open('skill_score.csv', 'r') as f:
        reader = csv.reader(f)
        report = list(reader)
    # aisling hacked the rows and columns - will have to remove
    row_min, row_max = 1, 10  # The rows of the table; forecast by forecast, add one for np.arange effect
    col_min, col_max = 4, 12  # The columns of the table; score by score, add one for the np.arange effect
    # names of scores
    score_names = report[0][col_min:col_max]
    # names of forecasts
    forecast_names = [report[i][0] for i in np.arange(row_min, row_max, 1)]
    # score values for each forecasts (lists within list)
    score_array = [report[i][col_min:col_max] for i in np.arange(row_min, row_max, 1)]
    # get radii for each forecasts score (lists within list)
    radii = []
    for results in score_array:
        r = []
        for result in results:
            skill_array = [float(score_array[i][results.index(result)]) for i in np.arange(0, len(score_array), 1)]
            ranked_skills, ranked_names = zip(*sorted(zip(skill_array, forecast_names)))
            for string in ranked_names:
                if str(forecast_names[score_array.index(results)]) == str(string):
                    index = ranked_names.index(string)
                    r.append((float(index) + 1) / 10)
        radii.append(r)
    # next up total scores and areas overall (single lists)
    total = []
    areas = []
    for array in radii:
        elem = [array[0], array[1], array[5], array[6], array[7]]
        i = np.linspace((2. * np.pi) / float(len(elem)), (2 * np.pi), len(elem)).tolist()
        total.append(np.sum(elem))
        i.append(i[0])
        elem.append(elem[0])
        corners = []
        for j in np.arange(0, len(elem), 1):
            t = i[j]
            r = elem[j]
            corners.append([r * (np.cos(t)), r * (np.sin(t))])
        poly = geometry.Polygon(corners)
        areas.append(poly.area)
    # some sort of sorting?
    names = forecast_names
#    areas, radii, names = zip(*sorted(zip(areas, radii, forecast_names), reverse=True))
    # making the plot
    # start the plot
    ax = plt.subplot(111, projection='polar')
    for array in radii:
        # here aisling is choosing only certain scores
        # namely, ROC area, BSS, RPSS, TSS, and HSS
        elem = [array[0], array[1], array[5], array[6], array[7]]
        i = (np.linspace((2. * np.pi) / float(len(elem)), (2 * np.pi), len(elem)).tolist())
        elem.append(elem[0])
        i.append(i[0])
        plt.scatter(i, elem, label=str(names[radii.index(array)]),
                    color=c[radii.index(array)])
        plt.plot(i, elem, color=c[radii.index(array)], alpha=0.7)
        plt.fill_between(i, elem, facecolor=c[radii.index(array)], alpha=0.3)
    plt.xticks(i, [score_names[0], score_names[1], score_names[5], score_names[6], score_names[7]])
    plt.yticks( np.linspace(0, 1 , 10) , [])
    plt.ylim([0,1])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='best', bbox_to_anchor=(1.1, 0.8))
    print(forecast_names)
    print(total)
    print(areas)
    plt.show()




