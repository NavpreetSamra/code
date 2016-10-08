from astar import AStar
import itertools as it
import pandas as pd
import numpy as np


class Frame(object):
    """
    """
    def __init__(self):
        """
        """
        meshSize = (50, 50)
        weights = np.linspace(0,25,41)
        heurs = [{'comp': 'vel', 'metric': 'octile', 'alpha': 1.},
                 {'comp': 'stock', 'metric': 'octile', 'alpha': 1.}]

        points = [(1, 1), (1, 24), (1, 48),
                  (24, 1), (24, 24), (24, 48),
                  (48, 1), (48, 24), (48, 48)
                  ]
        searches = list(it.combinations(points, 2))
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

