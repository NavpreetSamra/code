import numpy as np
from scipy import stats as scs
from matplotlib import pyplot as plt

def sample_sd(data):
    """
    Calculate standard deviation of sample data (Sn-1) assumes data 1d
    :param array-like data: 1d array of sample data values
    """
    data = np.array(data)
    return np.sqrt(np.sum((data - np.mean(data)) ** 2) / (len(data) - 1))


def standard_error(data):
    """
    Calculate standard error of sample data (assumes Sn-1) assumes data 1d
    :param array-like data: 1d array of sample data values
    """
    data = np.array(data)
    return sample_sd(data) / np.sqrt(len(data))


def power(data, null_mean, ci=0.95):
    """
    Calculate Power (1-:math:`beta`) given sample data,  
    null hypothesis mean and confidence interal
    :param array-like data: 1d array of sample values
    :param float null_mean: null hypothesis mean value
    :param float ci: confidence intervale (0,1)

    :return: power
    :rtype: float
    """
    m = data.mean()
    se = standard_error(data)
    z1 = scs.norm(null_mean, se).ppf(ci + (1 - ci) / 2)
    z2 = scs.norm(null_mean, se).ppf((1 - ci) / 2)
    power = 1 - scs.norm(data.mean(), se).cdf(z1) + scs.norm(data.mean(), se).cdf(z2)
    return power



def create_normal(mu, sigma, width=3, plot=True, ax=None, fig=None, **kwargs):
    """
    Create normal random variable :py:func:`scipy.stats`
    With center mu and standard deviation sigma and plot if desired
    :param float mu: Average of distribution
    :param float sigma: Standard Deviation of distribution
    :param bool plot: True to return :py:class:`matplotlib.pyplot.Axes` object
    :return: rv
    :rtype: :py:func:`scipy.stats.norm`
    :return: fig
    :rtype: matplotlib.pyplot.figure
    :return: axes
    :rtype: matplotlib.pyplot.axes
    """
    rv = scs.norm(mu, sigma)
    if not plot:
        return rv
    delta = width * sigma
    x_vec = np.linspace(mu - delta, mu + delta, 1000)
    y_vec = rv.pdf(x_vec)
    if not fig:
        fig = plt.figure()
    if not ax:
        ax = fig.add_subplot(111)
    ax.plot(x_vec, y_vec, **kwargs)

    return rv, fig, ax
