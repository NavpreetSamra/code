from astar import AStar
import itertools as it
import pandas as pd
import numpy as np


class Frame(object):
    """
    """
    def __init__(self, auto=False):
        """
        """
        if auto:
            meshSize = (50, 50)
            weights = np.linspace(0,20,41)
            heurs = [{'comp': 'vel', 'metric': 'octile', 'alpha': 1.},
                     {'comp': 'stock', 'metric': 'octile', 'alpha': 1.}]

            points = [(1, 1), (1, 48), (20, 20), 
                      (20, 28), (28, 28), (28, 48),
                      (48, 1),  (48, 48)
                      ]
            self.points = points
            searches = list(it.permutations(points, 2))
            template = pd.DataFrame(columns=weights,
                                    index=searches)
            self.results = {'counts': {i['comp']: template.copy() for i in heurs},
                            'costs': {i['comp']: template.copy() for i in heurs}
                            }

            for comp in heurs:
                for weight in weights:
                    star = AStar(meshSize=meshSize, heur=comp, heur_weight=weight)
                    for search in searches:
                            star.search(search)
                            self.results['counts'][comp['comp']][weight].loc[search] = star.iterCount
                            self.results['costs'][comp['comp']][weight].loc[search] = star.pathCost[search[1]]
    
    def obstacle(self):
        """
        """
        heurs = [{'comp': 'vel', 'metric': 'octile', 'alpha': 1.},
                 {'comp': 'stock', 'metric': 'octile', 'alpha': 1.}]
        self.a = AStar(meshSize=(30, 30), heur=heurs[0], velocity_type='uniform', embedObstacle=True, heur_weight=10, plots='images/obstacle/vel')
        self.b = AStar(meshSize=(30, 30), heur=heurs[1], velocity_type='uniform', embedObstacle=True, heur_weight=1.8, plots='images/obstacle/stock')

    def harmonic(self):
        """
        """
        heurs = [{'comp': 'vel', 'metric': 'octile', 'alpha': 1.},
                 {'comp': 'stock', 'metric': 'octile', 'alpha': 1.}]
        self.a = AStar(meshSize=(30, 30), heur=heurs[0], heur_weight=4, plots='images/harmonic/vel')
        self.b = AStar(meshSize=(30, 30), heur=heurs[1], heur_weight=1, plots='images/harmonic/stock')
