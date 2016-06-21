import numpy as np


def basismap_3d(a, b):
    """
    Find rigid body rotation and centroid translation between two states
    Default body centers are centroids

    :param array-like a: nx3 set of original coordinates in rigid body
    :param array-like b: nx3 set of original coordinates in transformed body
    """
    a = enforce_2d(a)
    b = enforce_2d(b)
    num_points = a.shape[0]

    a_centroid = np.mean(a, axis=0)
    b_centroid = np.mean(b, axis=0)

    a_centered = a - np.tile(a_centroid, (num_points, 1))
    b_centered = b - np.tile(b_centroid, (num_points, 1))

    ab = np.transpose(a_centered) * b_centered

    u, s, vt = np.linalg.svd(ab)

    r = vt.T * u.T

    # special reflection case (orientation correction)
    if np.linalg.det(r) < 0:
        vt[2, :] *= -1
        r = vt.T * u.T

    t = -r * a_centroid.T + b_centroid.T

    return r, t


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


def gen_empty_write_record(self):
    """
    """
    self.write_record = np.array(
                                 tuple([' '] * len(self.fields)),
                                 dtype=self.formats
                                )
