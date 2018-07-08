import numpy as np
from scipy.signal import convolve2d as c2d
import networkx as nx
from matplotlib import pyplot as plt


class Grid(object):
    """
    Class to setup and create graphs for :py:class:`astar.AStar` analysis

    :param array-like data: array of velocities to build grid from
    :param tuple.int meshSize: if data not provided create template size for mesh
    :param str distMetric: distance metric for relative computations (will be deprecated)
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
            self.mesh = np.meshgrid(np.arange(meshSize[1]),
                                    np.arange(meshSize[0]))
            data = np.ones_like(self.mesh)
        self.data = data
        self.x = self.mesh[0]
        self.y = self.mesh[1]

    def _point_diff(self, p1=[1, 3], p2=[2, 6]):
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

    def _dist_grids(self):
        """
        """
        self.u = self.p2 - self.p1
        self.um = np.linalg.norm(self.u)
        self.v = np.c_[self.x1.ravel(), self.y1.ravel()]
        self.proj = np.dot(self.v, self. u) / self.um**2
        self.p = np.c_[self.proj * self.u[0], self.proj * self.u[1]]

    def _calc_pdist(self):
        """
        """
        masks = [self.proj < 0, np.logical_and(self.proj >= 0, self.proj <= 1),
                 self.proj > 1]
        self.distVec = np.zeros_like(self.q_ravel)
        ps = [np.array([0, 0]), self.p[masks[1], :], self.u]
        for mask, p in zip(masks, ps):
            self.distVec[mask] = self.q_ravel[mask] - p
        self.dists = (np.linalg.norm(self.distVec, axis=1) /
                      self.nl).reshape(self.meshSize)

    def velocity_profile(self, profile='uniform', attrs={'scalar': 1.},
                         norm=True, scale=False, conv=None):
        """
        Generate either uniform or harmonic profile perscribed by attrs

        example- 'harmonic', {'scalar':3., 'xf':2., 'yf':2., 'xa':1.,'ya':1.}

        :param str profile: currently supports 'uniform' and 'harmonic'
        :param dict attrs: attributes of profile
        :param bool norm: normalize velocity Vmax := 1
        :param bool scale: scale velocity by std
        :param array conv: INDEV: convolution kernel for vectorized\
                velocity fields
        """
        if profile is 'uniform':
            self.velocity = attrs['scalar'] * np.ones(self.meshSize)

        if profile is 'harmonic':
            print 'harmonic'
            self.L = self.x.shape
            self.velocity = attrs['scalar'] +\
                            (attrs['xa'] * np.sin(np.pi*attrs['xf'] * np.pi *
                                                  self.x/self.L[1]) *\
                             attrs['ya'] * np.sin(np.pi*attrs['yf'] * np.pi *
                                                  self.y/self.L[0])
                             )

        if conv:
            self.conv_vel = self.velocity
            kernel = conv['kernel']
            for _ in conv['iter']:
                self.conv_vel = c2d(kernel, self.conv_vel)

        if norm:
            self.velocity /= np.max(self.velocity)
        elif scale:
            self.velocity /= np.std(self.velocity)
        self.vmax = np.max(self.velocity)

    def velocity_update(self, cells, values):
        """
        Update velocity of prescribed cells to values (1:1)

        :param array-like cells: array of cells to update with values
        :param array-like values: array of values to set cells
        """

        for cell, value in zip(cells, values):
            self.velocity[cell] = value

    def to_graph(self, links=8,
                 nodes=True,
                 edges=True
                 ):
        """
        Create graph from grid

        :param int links: number of node links, 4 := NSEW, 8:= NNeESeSSwWNw
        :param bool nodes: set non string nodes in graph as class attribute
        :param bool edges: set non string edges in graph as class attribute
        """
        if links == 8:
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif links == 4:
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        distances = np.linalg.norm(np.array(directions), axis=1)
        self.graphs = {}
        for d in directions:
            self.graphs[d] = nx.DiGraph()

        self.graph = nx.DiGraph()
        self.vdirect = {}
        for i in range(self.meshSize[0]):
            for j in range(self.meshSize[1]):
                vattrs = {}
                for direct, dist in zip(directions, distances):
                    n1 = (i, j)
                    n2 = (i + direct[0], j + direct[1])
                    if all([n2[0] >= 0,  n2[0] < self.meshSize[0],
                       n2[1] >= 0, n2[1] < self.meshSize[1]]):

                        v1 = self.velocity[n1[0], n1[1]]
                        v2 = self.velocity[n2[0], n2[1]]
                        vel = np.mean([v1, v2])

                        attrs = {'vel': vel,
                                 'dist': dist,
                                 'direct': direct
                                 }
                        self.graph.add_edge(n1, n2, attr_dict=attrs)
                        vattrs[direct] = vel
                self.graph[n1]['vel'] = v1
        if nodes:
            self.nodes = [i for i in self.graph.nodes() if not any([isinstance(i[0], str), isinstance(i[1], str)])]
        if edges:
            self.edges = [i for i in self.graph.edges() if not any([isinstance(i[0], str), isinstance(i[1], str)])]

    def plot_graph(self, fName='t.png', edge_color='vel', node_color=None, node_size=50, nodes=None, nodePath=None):
        """
        Plot edge velocities in graph

        :param str fName: name to save file to
        :param str edge_color: attribute of graph to pull for edge colors
        :param str node_color: attribute of graph to pull for node colors
        :param int node_size: node size in plots
        :param list nodes: list of nodes to star
        :param list nodePath: list of nodes to highlight
        """
        uGraph = nx.Graph()
        uGraph.add_edges_from(self.edges)
        self.pos = {i: np.array(i) for i in self.nodes}
        edge_colors = [self.graph[i[0]][i[1]][edge_color]
                       for i in uGraph.edges()]
        if node_color:
            node_colors = [self.graph[i][node_color] for i in uGraph.nodes()]
        else:
            node_colors = 'k'
        self.template_fig()
        nx.draw_networkx(uGraph, self.pos, with_labels=False,
                         edges=self.edges, edge_color=edge_colors,
                         nodes=self.nodes, node_color=node_colors, cmap=plt.cm.viridis,
                         width=2, node_size=node_size, edge_cmap=plt.cm.viridis)
        if nodePath:
            nx.draw_networkx_nodes(uGraph, self.pos, nodelist=nodePath, node_size=75, node_color='red')
        if nodes:
            nx.draw_networkx_nodes(uGraph, self.pos, nodelist=nodes, node_size=300, node_shape='*', node_color='hotpink')
        self.fig.savefig(fName)

    def _make_graph(self):
        self.velocity_profile('harmonic',{'xf':2, 'yf':2, 'xa':1,'ya':1,'scalar':1})
        self.to_graph()
        self.plot_graph()

    def plot_state(self, openNodes, closedNodes, edges=False, source=None, target=None):
        """
        Plot state in a search

        :param array-like openNodes: open nodes in search
        :param array-like closedNodes: closed nodes in search
        :param array-lik edges: plot edges
        :param array-lik source: plot source
        :param array-lik target: plot target
        """
        uGraph = nx.Graph()
        uGraph.add_edges_from(self.edges)
        self.pos = {i: np.array(i) for i in self.nodes}
        if 'ax' not in self.__dict__:
            self.template_fig()

        nx.draw_networkx_nodes(uGraph, self.pos, nodelist=openNodes,
                node_size=50, node_color='b', ax=self.ax)

        nx.draw_networkx_nodes(uGraph, self.pos, nodelist=closedNodes,
                node_size=50, node_color='k', ax=self.ax)
        if edges:
            nx.draw_networkx_edges(uGraph, self.pos, uGraph.edges(closedNodes), ax=self.ax)
        if source:
            nx.draw_networkx_nodes(uGraph, self.pos, nodelist=source,
                    node_size=100, node_color='lime', node_shape='*', ax=self.ax)
        if target:
            nx.draw_networkx_nodes(uGraph, self.pos, nodelist=target,
                    node_size=100, node_color='lime', node_shape='*', ax=self.ax)

    def template_fig(self):
        """
        Create template for figures
        """
        if 'fig' not in self.__dict__:
            self.fig = plt.figure(frameon=False)
        if 'ax' not in self.__dict__:
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)
            self.ax.set_xlim([-1, self.meshSize[0]])
            self.ax.set_ylim([-1, self.meshSize[1]])

    def _plot_obstacle(self, nodes, color='r', size='75'):
        """
        Plot obstacle or feature nodes
        """
        uGraph = nx.Graph()
        uGraph.add_edges_from(self.edges)
        nx.draw_networkx_nodes(uGraph, self.pos, nodelist=nodes,
                node_size=size, node_color=color, ax=self.ax)

    def plot_path(self, nodes):
        """
        Plot path of nodes
        """
        uGraph = nx.Graph()
        uGraph.add_edges_from(self.edges)

        edges = [(i, j) for i, j in zip(nodes[:-1], nodes[1:])]
        print edges

        nx.draw_networkx_nodes(uGraph, self.pos, nodelist=nodes,
                node_size=100, node_color='red', ax=self.ax)

        # nx.draw_networkx_edges(uGraph, self.pos, edges=edges, ax=self.ax, edge_color='hotpink', weight=2) 

def gaussian_kernel(size, ySize=None):
    """
    Create normalized gausian kernel size x ySize
    :param int size: x dimension of kernel
    :param int ySize: y dimension of kernel default to square
    """
    if not ySize:
        ySize = size
    x, y = np.mgrid[-size:size+1, -ySize:ySize+1]
    g = np.exp(-(x**2/float(size) + y**2/float(ySize)))
    return g/g.sum()

def convolve_axis(filt, arr, axis=0):
    """
    Convolve filter along axis through array

    :param array-like filt: filter to convolve
    :param array-like arr: array to be convolved
    :param int axis: axis to convolve along
    """
    return np.apply_along_axis(lambda x: np.convolve(x, filt, mode='same'), axis=axis, arr=arr)
