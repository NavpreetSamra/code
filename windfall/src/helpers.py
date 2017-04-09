
import pandas as pd
import numpy as np
from scipy import interp
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="paper", font="monospace")



def df_table(df, fname):
    """
    Helper function to create rst tables from :py:class:`pandas.DataFrame`

    :param `pandas.DataFrame` df: data frame
    :param str fname: string file name
    """
    with open(fname, 'a') as w:
        w.write(tabulate.tabulate(df, tuple(df.columns), tablefmt='rst'))
        w.write('\n')


def corr_heatmap(df, fname, vmax=.8, absolute=False):

    """
    Helper function to create correlation heatmap imagesfrom :py:class:`pandas.DataFrame`

    :param `pandas.DataFrame` df: data frame
    :param str fname: string file name
    :param vmax float: absolute value bound for colorbar
    """
    corr = df.corr()
    if absolute:
        corr = np.abs(corr)
    sns.heatmap(corr, vmax=vmax, square=True)
    plt.savefig(fname)
    plt.close()


def stacked_bar(df, col, target, fName):
    """
    Helper function for creating stacked bar charts based on the values in a column
    and the categories of a target
    """
    df2 = df.groupby([col, target])[col].count().unstack(target).dropna()
    df2.plot(kind='bar', stacked=True)
    plt.savefig(fName)
    plt.close()


def build_param_grid():
    """
    Helper function to build parameter grid for hyperparameter tuning

    :return: classifiers and parameter grids
    :rtype: dict
    """
    g = {'lasso': {'clf': Lasso(), 'grid': {'classifier__alpha': np.logspace(1, 6, 12)}},
         'ridge': {'clf': Ridge(), 'grid': {'classifier__alpha': np.logspace(-1, 4, 12)}},
         'rf': {'clf': RandomForestRegressor(n_estimators=60),
                'grid': {'classifier__max_depth': [7, 10, 15],
                         'classifier__min_samples_split': [4, 10, 25]}
                },
         'gb': {'clf': GradientBoostingRegressor(n_estimators=60), 'grid': {'classifier__max_depth':[1,2,3,4], 'classifier__min_samples_split': [2,8,16]}}

         }

    return g


def _collinear_vif(self):
    """
    Check for collinear features
    """
    for ind in range(self.X.shape[1]):
        value = vif(self.X, ind)
        if value > self.vifMagnitude:
            print self.columns[ind] + ' has vif ' + str(value)
            self.collinear = True
    if self.collinear:
        raise Exception('Collinear feature risk')


class ItemSelector(BaseEstimator, TransformerMixin):
    """
    For data grouped by feature, select subset of data at a provided key.

    :param key-like: valid selector for whatever is fit
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


    @staticmethod
    def hold(train, targetField, floorThresh=(),
                      ceilThresh=(), dropZeros=True):

            mask = np.ones((design.shape[0],)).astype(bool)
            print sum(mask)

            for col, criteria in floorThresh:
                print col, criteria
                design = design.fillna({col: criteria})
                submask = design[col] > criteria
                mask = mask & submask.values
                print sum(mask)

            for col, criteria in ceilThresh:
                design = design.fillna({col: criteria})
                submask = design[col] < criteria
                mask = mask & submask.values
                print sum(mask)

            design = StandardScaler().fit_transform(design)
            design = add_constant(design)
            model = OLS(target, design)
            mask = mask &\
                  (model.fit().outlier_test()['student_resid'].abs() < 2)

            print sum(mask)
            return mask

