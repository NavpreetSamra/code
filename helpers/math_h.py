import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import itertools


def basismap_3d(a, b, centroids=None):
    """
    Find rigid body rotation and centroid translation between two states
    Default body centers are centroids

    :param array-like a: nx3 set of coordinates in rigid body
    :param array-like b: nx3 set of coordinates in transformed body
    :param array-like centroids: 2x3 array to specify centroids [[ax,ay,az],[bx,by,bz]] \ 
                                 defaults to [centroid(a), centroid(b)]
    """
    a = enforce_2d(a)
    b = enforce_2d(b)
    num_points = a.shape[0]
    if centroids:
        centroids = np.asarray(centroids)
        a_centroid = centroids[0, :]
        b_centroid = centroids[1, :]
    else:
        a_centroid = np.mean(a, axis=0)
        b_centroid = np.mean(b, axis=0)

    a_centered = a - np.tile(a_centroid, (num_points, 1))
    b_centered = b - np.tile(b_centroid, (num_points, 1))

    ab = np.transpose(a_centered) * b_centered

    u, s, vt = np.linalg.svd(ab)

    r = vt.T * u.T

    # Reflection orientation correction
    if np.linalg.det(r) < 0:
        vt[2, :] *= -1
        r = vt.T * u.T

    t = -r * a_centroid.T + b_centroid.T

    return r, t


def scatter_quick(data, save=None, proj=111):
    """
    Quick scatter plots 2d and 3d
    Note: nx3 plots will be colored to 3rd dim

    :param array-like data: n x [2, 3, 4] array to scatter plot
    :param str save: default behaviour display plot else save to string name
    :param int proj: projected view for 3d plots

    """
    if data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(proj, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 2])

    elif data.shape[1] == 2:
        plt.scatter(data[:, 0], data[:, 1])

    elif data.shape[1] == 4:
        fig = plt.figure()
        ax = fig.add_subplot(proj, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 3])

    if save:
        plt.savefig(save)
    else:
        plt.show()

def enforce_2d(data):
    """
    Enforce an array-like entity is a multidimensional ndarray

    :param array-like  data: array of points

    :return data_nd: multidimensional array of data
    :rtype np.ndarray:
    """

    data = np.asarray(data)
    if len(data.shape) > 1:
        return data
    else:
        data_nd = data.reshape(1, data.shape)
        return(data_nd)


class TreeGraph(nx.Graph):
    """
    Subclass of networkx Graph for path search applications
    """

    def tree_diameter(self):
        """
        Find diameter node pairs and path in graph
        """

        self.diameter = None

        nodes = self.nodes()
        node_min = np.min(nodes)
        node_max = np.max(nodes)

        edges = []
        [edges.extend(list(i)) for i in self.edges()]
        edge_array = np.asarray(edges).flatten()
        self.counts, self.bins = np.histogram(edge_array,
                                              bins=np.arange(node_min, node_max)
                                              )
        logical = self.counts == 1
        leaves = [int(i) for i in self.bins[logical]]
        iter_paths = itertools.combinations(leaves, 2)

        for tree_path in iter_paths:
            cost = nx.shortest_path_length(self, tree_path[0], tree_path[1])
            if cost > self.diameter:
                self.diameter = cost
                self.diameter_path = nx.shortest_path(self, tree_path[0], tree_path[1])

