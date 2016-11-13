import grid
from Queue import PriorityQueue
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist



class AStar(grid.Grid):
    """
    AStar class, designed to search paths from :py:class:`route_vision.grid.Grid`

    For more information please see `docs
    <https://marksweissma.github.io/code/slides.html>`_.

    :param tuple.int meshSize: x,y size of graph
    :param str velocity_type: type of profile, scalar or harmonic
    :param dict velocity attrs: define velocity profile
    :param dict heur: define heuristic type
    :param float heur_weight: define relative weight of prioritization\
            (cost + heur*estimate)
    :param tuple.float embedPath: specify value of location to set velocity
    :param bool embedObstacle: embed an obstacle
    :param bool auto: auto generate profile and build attrs
    :param int links: if auto generating graphs, specify if graph is\
            manhattan or octile
    """
    def __init__(self, meshSize=(3, 3), plots=None,
                 velocity_type='harmonic',
                 velocity_attrs={'scalar': 10, 'xf': 2., 'yf': 2.,
                                 'xa': 3., 'ya': 3.},
                 heur={'comp': 'vel', 'metric': 'octile', 'alpha': 1.},
                 heur_weight=5,
                 embedPath=None,
                 embedObstacle=None,
                 auto=True,
                 links=8
                 ):

        super(AStar, self).__init__(meshSize=meshSize)
        self.heur = heur
        self.plots = plots
        self.heur_weight = heur_weight
        if auto:
            self.velocity_profile(velocity_type, velocity_attrs)
            if embedPath:
                self.velocity_update(embedPath)
            if embedObstacle:
                self.build_obstacle()
            self.to_graph(links)
        self.rt2 = np.sqrt(2)

    def heuristic_component(self, node, neighbor):
        """
        E[t] for search

        :param tuple.int node: node id in graph currently being explored
        :param tuple.int neighbor: node id in graph being opened
        :return: cost estimate
        :rtype: float
        """
        ###########
        ## Build relative weighting based on distance completed
        ###########
        metric = self.heur['metric']
        comp = self.heur['comp']
        alpha = self.heur['alpha']

        vel = self.graph[neighbor]['vel']
        if metric == 'octile':
            dist1 = self.octile_dist(self.n2, neighbor)
            dist2 = self.octile_dist(self.n1, neighbor)
        else:
            dist1 = pdist([self.n2, neighbor], metric=metric)
            dist2 = pdist([self.n1, neighbor], metric=metric)

        if comp == 'vel':
            value = dist1/(1 + dist1 / ((dist1 + dist2) * vel))
        elif comp == 'veld':
            dvel = self.graph[node][neighbor]['vel']
            value = dist1 + (dist1 * self.graph[node][neighbor]['dist']) /\
                            ((dist1 + dist2) * np.mean([vel, dvel]))
        elif comp == 'eig':
            value = dist1 + 1 / np.mean([vel, self.norm_scores[neighbor]])
        elif comp == 'stock':
            value = dist1
        return 1 + alpha * (value - 1)

    def cost(self, a, b):
        """
        Cost of edge in time := distance/velocity

        :param tuple.int a: node id
        :param tuple.int b: node id
        """
        return self.graph[a][b]['dist'] / float(self.graph[a][b]['vel'])

    def _build_city(self):
        """
        """
        self.velocity_profile('uniform')
        self.velocity[2:55, 15:25] = .0001
        self.velocity[[10, 20, 30, 40, 50], :] = 2
        self.velocity[[10, 20, 30], :] = 2
        self.velocity[10:20, [10, 30]] = 2
        self.velocity[30:40, [10, 30]] = 2
        self.to_graph(4)

    def build_obstacle(self):
        """
        Build nodes (10:20, 10:15) to velocity = .1
        """
        self.velocity[10:20, 10:15] = .1

    def search(self, n):
        """
        Search path from n[0] to n[1]

        :param tuple.tuple.int n: pair of node ids in graph (source, target)
        """
        self.n1, self.n2 = n

        self.pq = PriorityQueue()
        self.pq.put((0, self.n1))

        self.pathFrom = {self.n1: None}
        self.pathCost = {self.n1: 0}
        self.closed = set([self.n1])

        if self.heur['comp'] == 'eig':
            self.eig_scores = nx.eigenvector_centrality_numpy(self.graph,
                                                              weight='vel')
            max_score = np.max([i for i in self.eig_scores.values()])
            self.norm_scores = {i: float(j)/max_score for i, j in
                                self.eig_scores.iteritems()}
        self.iterCount = 0
        self.callCount = 0
        while not self.pq.empty():
            self.iterCount += 1
            node = self.pq.get()[1]
            self.closed.add(node)
            if self.plots:
                self.plot()
            if node == self.n2:
                print 'found'
                if self.plots:
                    self.plot()
                break
            neighbors = [i for i in self.graph.neighbors(node)
                         if not isinstance(i, str)]
            for neighbor in neighbors:
                neighborCost = self.pathCost[node] + self.cost(node, neighbor)
                if neighbor not in self.pathCost or\
                        neighborCost < self.pathCost[neighbor]:
                    self.callCount += 1
                    self.pathCost[neighbor] = neighborCost
                    value = neighborCost + self.heur_weight *\
                                           self.heuristic_component(node,
                                                                    neighbor)
                    self.pq.put((value, neighbor))
                    self.pathFrom[neighbor] = node

        if self.plots:
            self.path_eval()
            self.plot_path(self.path_found)
            grid.plt.close(self.fig)
            self.fig.savefig(self.plots+str(self.iterCount).zfill(4)+'.png')
            grid.plt.close(self.fig)

    def plot(self):
        """
        Plot current state in search
        """
        openNodes = [self.pq.queue[i][1] for i in range(len(self.pq.queue))]
        closedNodes = self.closed
        self.plot_state(openNodes, closedNodes, source=[self.n1],
                        target=[self.n2])
        self.fig.savefig(self.plots+str(self.iterCount).zfill(4)+'.png')
        grid.plt.close(self.fig)

    def path_eval(self, plots='vel_path.png'):
        """
        Create list of nodes in path

        :param str plots: option to plot path and save to file
        """
        self.path_found = list([self.n2])
        node = self.n2
        while self.pathFrom[node]:
            node = self.pathFrom[node]
            self.path_found.insert(0, node)

        if plots:
            self.plot()
            self.plot_path(self.path_found)

    def octile_dist(self, a, b):
        """
        Compute octile distance

        :param array-like a: point a
        :param array-like b: point b
        :return: dist
        :rtype: float

        """
        dx = np.abs(a[1] - b[1])
        dy = np.abs(a[0] - b[0])
        dist = (dx + dy) + (self.rt2 - 2.) * np.min([dx, dy])
        return dist
