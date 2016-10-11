import astar


class City(object):
    """
    """
    def __init__(self):
        ha = {'comp': 'vel', 'metric': 'cityblock', 'alpha': 1}
        hb = {'comp': 'stock', 'metric': 'cityblock', 'alpha': 1}
        self.a = astar.AStar(meshSize=(60, 40), heur=ha, heur_weight=2, links=4)
        self.b = astar.AStar(meshSize=(60, 40), heur=hb, heur_weight=2, links=4)
        self.a.build_city()
        self.b.build_city()

