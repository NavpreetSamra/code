import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import itertools
import copy
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def basismap_3d(a, b, centroids=None):
    """
    Find rigid body rotation and centroid translation between two states
    Default body centers are centroids

    :param array-like a: nx3 set of coordinates in rigid body
    :param array-like b: nx3 set of coordinates in transformed body
    :param array-like centroids: 2x3 array to specify centroids [[ax,ay,az],[bx,by,bz]] \ 
                                 defaults to [centroid(a), centroid(b)]

    :return: (r, t) rotation matrix, translation array
    :rtype: tuple
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

    if data.shape[1] == 2:
        plt.scatter(data[:, 0], data[:, 1])

    else:
        fig = plt.figure()
        ax = fig.add_subplot(proj, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, -1])

    if save:
        plt.savefig(save)
    else:
        plt.show()

def enforce_2d(data):
    """
    Enforce an array-like entity is a multidimensional ndarray

    :param array-like  data: array of points

    :return data_nd: multidimensional array of data
    :rtype: np.ndarray
    """

    data = np.asarray(data)
    if len(data.shape) > 1:
        return data
    else:
        data_nd = data.reshape(1, data.shape)
        return(data_nd)


def spiral(a, cond=(lambda b: len(b) > 0), cpy=False):
    """
    Spiral unpack a list of lists via counterclock rotation \
            until a condition is met

    NB: Function utilizes pop, use cpy to preserve original if you are \
            unsure of impact in the applied namespace

    :param array-like a: if a is not list, it will be converted by list(a)
    :param func cond: conditional expression to stop at while spiraling a.
    :param bool cpy: deepycopy input
    """

    output = []

    if not isinstance(a, list):
        a = list(a)
    if cpy:
        a = copy.deepcopy(a)

    while cond(a):
        output.extend(a.pop(0))
        # Pivot matrix counter clockwise
        a = [list(x) for x in zip(*a)][::-1]
    return output


