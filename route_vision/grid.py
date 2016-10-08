import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


class Grid(object):
    """
    Class to setup and create graphs for :py:class:astar.AStar analysis
    """
    def __init__(self, data=None, meshSize=(3, 3),
                 distMetric='euclidean'):

        self.meshSize = meshSize
        self.distMetric = distMetric
        if data:
            self.mesh = np.meshgrid(np.arange(data.shape[0]),
                                    np.arange(data.shape[1])
                                    )
        if not data:
            self.mesh = np.meshgrid(np.arange(meshSize[0]),
                                    np.arange(meshSize[1]))
            data = np.ones_like(self.mesh)
        self.data = data
        self.x = self.mesh[0]
        self.y = self.mesh[1]
        self.point_diff()

    def point_diff(self, p1=[1, 3], p2=[2, 6]):
        """
        """
        self.p1 = np.asarray(p1)
        self.p2 = np.asarray(p2)
        self.nl = np.linalg.norm(self.p2 - self.p1)
        self.x1 = self.x - p1[0]
        self.y1 = self.y - p1[1]
        self.q_ravel = np.c_[self.x1.ravel(), self.y1.ravel()]
        self.xd1 = np.abs(self.x - p1[0])
        self.yd1 = np.abs(self.y - p1[1])

        self.xd2 = np.abs(self.x - p2[0])
        self.yd2 = np.abs(self.y - p2[1])
        self.dist_grids()
        self.calc_pdist()

    def dist_grids(self):
        """
        """
        self.u = self.p2 - self.p1
        self.um = np.linalg.norm(self.u)
        self.v = np.c_[self.x1.ravel(), self.y1.ravel()]
        self.proj = np.dot(self.v, self. u) / self.um**2
        self.p = np.c_[self.proj * self.u[0], self.proj * self.u[1]]

    def calc_pdist(self):
        """
        """
        masks = [self.proj < 0, np.logical_and(self.proj >= 0, self.proj <= 1),
                 self.proj > 1]
        self.distVec = np.zeros_like(self.q_ravel)
        ps = [np.array([0, 0]), self.p[masks[1], :], self.u]
        for mask, p in zip(masks, ps):
            self.distVec[mask] = self.q_ravel[mask] - p
        self.dists = (np.linalg.norm(self.distVec, axis=1) /\
                      self.nl).reshape(self.meshSize)

    def velocity_profile(self, profile='uniform', attrs={'scalar':1.},
                         norm=True, scale=False):
        """
        Generate either uniform or harmonic profile prescribeed by attrs

        example- 'harmonic',{'scalar':1., 'xf':2., 'yf':2., 'xa':1.,'ya':1.}
        """
        if profile is 'uniform':
            self.velocity = attrs['scalar'] * np.ones_like(self.dists)

        if profile is 'harmonic':
            print 'harmonic'
            self.L = self.x.shape
            self.velocity = attrs['scalar'] +\
                            (attrs['xa'] * np.sin(np.pi*attrs['xf'] * np.pi *
                                                  self.x/self.L[1]) *\
                             attrs['ya'] * np.sin(np.pi*attrs['yf'] * np.pi *
                                                  self.y/self.L[0])
                             )

        if norm:
            self.velocity /= np.max(self.velocity)
        elif scale:
            self.velocity /= np.std(self.velocity)

    def velocity_update(self, update):
        """
        update velocity of prescribed cells to values
        """
        cells = update['cells']
        values = update['values']

        for cell, value in zip(cells, values):
            self.velocity[cell] = value

    def to_graph(self, directions=[(0, 1), (0, -1), (1, 0), (-1, 0),
                                   (1, 1), (1, -1), (-1, 1), (-1, -1)],
                 nodes=True,
                 edges=True
                 ):
        """
        create graph from grid
        """
        distances = np.linalg.norm(np.array(directions), axis=1)

        self.graph = nx.DiGraph()
        for i in range(self.meshSize[1]):
            for j in range(self.meshSize[0]):
                for direct, dist in zip(directions, distances):
                    n1 = (i, j)
                    n2 = (i + direct[0], j + direct[1])
                    if (n2[0] >= 0 and n2[0] <= (self.meshSize[1] - 1)) and\
                       (n2[1] >= 0 and n2[1] <= (self.meshSize[0] - 1)):

                        v1 = self.velocity[n1[0], n1[1]]
                        v2 = self.velocity[n2[0], n2[1]]
                        vel = np.mean([v1, v2])

                        attrs = {'vel': vel,
                                 'dist': dist,
                                 'direct': direct
                                 }
                        self.graph.add_edge(n1, n2, attr_dict=attrs)
                # self.graph[n1]['vel'] = v1
        if nodes:
            self.nodes = self.graph.nodes()
        if edges:
            self.edges = self.graph.edges()

    def plot_graph(self, fName='t.png', edge_color='vel', node_color=None):
        """
        plott edge velocities in graph
        """
        uGraph = nx.Graph(self.graph)
        self.pos = {i: np.array(i) for i in self.nodes}
        edge_colors = [self.graph[i[0]][i[1]][edge_color]
                       for i in uGraph.edges()]
        if node_color:
            node_colors = [self.graph[i][node_color] for i in uGraph.nodes()]
        else:
            node_colors = 'k'
        self.template_fig()
        # fig = plt.figure(frameon=False)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        nx.draw_networkx(uGraph, self.pos, with_labels=False,
                         edges=self.edges, edge_color=edge_colors,
                         nodes=self.nodes, node_color=node_colors,
                         width=2, node_size=50)
        self.fig.savefig(fName)
        # plt.close(self.fig)

    def _make_graph(self):
        self.velocity_profile('harmonic',{'xf':2, 'yf':2, 'xa':1,'ya':1,'scalar':1})
        self.to_graph()
        self.plot_graph()

    def plot_state(self, openNodes, closedNodes, edges=False):
        """
        Plot state in a search
        """
        uGraph = nx.Graph(self.graph)
        self.pos = {i: np.array(i) for i in self.nodes}
        if 'ax' not in self.__dict__:
            self.template_fig()

        nx.draw_networkx_nodes(uGraph, self.pos, nodelist=openNodes,
                node_size=50, node_color='b', ax=self.ax)

        nx.draw_networkx_nodes(uGraph, self.pos, nodelist=closedNodes,
                node_size=50, node_color='k', ax=self.ax)
        if edges:
            nx.draw_networkx_edges(uGraph, self.pos, uGraph.edges(closedNodes), ax=self.ax)

    def template_fig(self):
        """
        Create template for figures
        """
        self.fig = plt.figure(frameon=False)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        self.ax.set_xlim([-1, self.meshSize[1]])
        self.ax.set_ylim([-1, self.meshSize[0]])

    def plot_obstacle(self, nodes, color='r', size='75'):
        """
        Plot obstacle or feature nodes
        """
        uGraph = nx.Graph(self.graph)
        nx.draw_networkx_nodes(uGraph, self.pos, nodelist=nodes,
                node_size=size, node_color=color, ax=self.ax)

    def plot_path(self, nodes):
        """
        Plot path of nodes
        """
        uGraph = nx.Graph(self.graph)

        edges = [(i, j) for i, j in zip(nodes[:-1], nodes[1:])]
        print edges

        nx.draw_networkx_nodes(uGraph, self.pos, nodelist=nodes,
                node_size=100, node_color='lime', ax=self.ax)

        # nx.draw_networkx_edges(uGraph, self.pos, edges=edges, ax=self.ax, edge_color='hotpink', weight=2) 

