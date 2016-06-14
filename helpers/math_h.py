import numpy as np

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
