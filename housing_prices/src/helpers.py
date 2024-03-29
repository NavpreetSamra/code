
import pandas as pd
import numpy as np
from scipy import interp
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="paper", font="monospace")


def month_year_ftime(pdSeries):
    """
    Return month-year from :py:class:`pandas.Series` with date information

    (converts stringsto datetime object before converting back to string)

    :param `pandas.Series` pdSeries: series to convert

    :return: month year series
    :rtype: `pandas.Series`
    """
    return pd.to_datetime(pdSeries, errors='coerce')\
          .apply(lambda x: x.strftime('%m-%Y'))


def _build_key(cols):
    """
    Convert flexible string / tuple / array-like key types to list based keys
    to ensure :py:class:`pandas.DataFrame` will maintain type after slicing
    """
    if isinstance(cols, str):
        cols = [cols]
    elif any([isinstance(cols, i) for i in [np.ndarray, pd.Series, tuple]]):
        cols = list(cols)
    elif not isinstance(cols, list):
        raise Exception('cols not valid key type')

    return cols


def build_train_transform_kwargs():
    """
    Build dict of kwargs for :func:`Cleaner.transform` for training

    :return: kwargs
    :rtype: dict
    """
    d = {'_pull_raw': {'cols': ['estimated_value', 'last_sale_amount']},
         '_apply_power': {'cols': ['square_footage'], 'powers': [1.81], 'scales': [1000.]},
         '_inflate': {'timeCol': ['last_sale_date'], 'histCol': ['last_sale_amount']}
         # '_build_categoricals': {'categoricalCols': ['zipcode']}
         # '_build_quarters': {'cols': []},
         }
    return d


def build_transform_kwargs():
    """
    Build dict of kwargs for :func:`Cleaner.transform` for predicting

    :return: kwargs
    :rtype: dict
    """
    d = {'_pull_raw': {'cols': ['last_sale_amount']},
         '_apply_power': {'cols': ['square_footage'], 'powers': [1.81], 'scales': [1000.]},
         '_inflate': {'timeCol': ['last_sale_date'], 'histCol': ['last_sale_amount']}
         # '_build_categoricals': {'categoricalCols': ['zipcode']}
         # '_build_quarters': {'cols': []},
         }
    return d


def build_param_grid():
    """
    Helper function to build parameter grid for hyperparameter tuning

    :return: classifiers and parameter grids
    :rtype: dict
    """
    g = {'lasso': {'clf': Lasso(), 'grid': {'classifier__alpha': np.logspace(-3, 7, 12)}},
         # 'ridge': {'clf': Ridge(), 'grid': {'classifier__alpha': np.logspace(-1, 4, 12)}},
         'rf': {'clf': RandomForestRegressor(n_estimators=100, n_jobs=-1),
                'grid': {'classifier__max_depth': [14, 18, 20],
                         'classifier__min_samples_split': [ 12, 15, 28]}
                },
         'gb': {'clf': GradientBoostingRegressor(n_estimators=100), 'grid': {'classifier__max_depth':[2,3,4], 
                                                                             'classifier__learning_rate':np.logspace(-3,0,4)}},
         # 'svm': {'clf': SVR(kernel='rbf'), 'grid': {'classifier__C': np.logspace(-3,3,7), 'classifier__gamma': np.logspace(-5,2,4)}}

         }

    return g


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

