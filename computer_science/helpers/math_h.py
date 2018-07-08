import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import itertools as it
import copy
from scipy.signal import butter, lfilter, freqz
from scipy import linalg
from math import cos, sin


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


def scatter_quick(data, save=None, proj=111, t=False):
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
        if t:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=range(data.shape[0]))
        else:
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
        data_nd = data.reshape(1, data.shape[0])
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


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def project(xyz, v2=np.array([0., 0., 1.])):
    """
    """
    n2 = v2/np.linalg.norm(v2)
    n2 = enforce_2d(n2)
    centroid, n1 = fit_plane(xyz)
    xyz = xyz - centroid
    xyz_n1 = enforce_2d(xyz.dot(n1)).T
    xyzPlane = xyz - xyz_n1 * n2
    return xyzPlane


def fit_plane(points):
    """
    """
    points = np.array(points).T
    points = points.reshape((points.shape[0], -1))
    centroid = points.mean(axis=1)
    x = points - centroid[:, np.newaxis]
    M = np.cov(x)
    normal = np.linalg.svd(M)[0][:, -1]
    return centroid, normal


def center_rotate(xyz, v2=np.array([0., 0., 1.])):
    v1 = xyz.mean(axis=0)
    xyz = xyz - v1
    return rotate_v1_v2(v1, v2, xyz)


def rotate_v1_v2(v1, v2, xyz):
    v1 = enforce_2d(v1)
    v2 = enforce_2d(v2)
    n1 = np.asarray(v1) / np.linalg.norm(v1)
    n2 = np.asarray(v2) / np.linalg.norm(v2)

    axis = np.cross(n1, n2)
    theta = n1.dot(n2.T)

    rot = rot_from_axis_angle(axis, theta)
    return rot.dot(xyz.T).T


def rot_from_axis_angle(axis, theta):
    axis = axis / np.linalg.norm(axis)
    return linalg.expm3(np.cross(np.eye(3), axis * theta))


def matrix_from_v1_v2(v1, v2):
    n1 = np.asarray(v1) / np.linalg.norm(v1)
    n2 = np.asarray(v2) / np.linalg.norm(v2)
    theta = n1.dot(n2.T)
    axis = np.cross(v1, v2)
    return rot_from_axis_angle(axis, theta)


class Line(object):
    """
    """
    def __init__(self, p1, p2):
        self.x = x
        self.y = y
        sel.v = np.array([x. y])
        self.mag = np.linalg

    def angle(self, l2):
        pass

