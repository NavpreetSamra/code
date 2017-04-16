import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve

from scipy import interp

from statsmodels.stats.outliers_influence import (variance_inflation_factor
                                                  as vif)

import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="paper", font="monospace")


def _build_results_df(results, x):
    df = pd.DataFrame(index=results.keys(), columns=x.columns)
    for name, clf in results.iteritems():
        est = clf.best_estimator_
        if hasattr(est, 'feature_importances_'):
            vals = clf.best_estimator_.feature_importances_ /\
                sum(clf.best_estimator_.feature_importances_)

        elif hasattr(est, 'coef_'):
            vals = clf.best_estimator_.coef_ /\
                sum(clf.best_estimator_.coef_)
        df.loc[name] = vals

    return df


def build_param_grid():
    """
    Helper function to build parameter grid for hyperparameter tuning

    :return: classifiers and parameter grids
    :rtype: dict
    """
    g = {'lr': {'clf': LogisticRegression(class_weight='balanced'),
                'grid': {'C': np.logspace(1, 4, 6), 'penalty': ['l1', 'l2']}},
         'rf': {'clf': RandomForestClassifier(n_estimators=100, n_jobs=-1),
                'grid': {'max_depth': [10, 20, 30],
                         'min_samples_split': [10, 20, 60]}
                }

         }

    return g


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
    Helper function to create correlation heatmap images\
            from :py:class:`pandas.DataFrame`

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
    Helper function for creating stacked bar charts based on the values
    in a column and the categories of a target
    """
    df2 = df.groupby([col, target])[col].count().unstack(target).dropna()
    df2.plot(kind='bar', stacked=True)
    plt.savefig(fName)
    plt.close()


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


def select_transform_features(design, target,
                              featureModel=LinearSVC(C=0.01, penalty="l1",
                                                     dual=False)
                              ):
    """
    Build :py:class:`sklearn.Pipeline` with mean imputing and\
            mean + variance scaling

    :param sklearn.estimator clf: estimator
    :parm key-like cols: columns to pull from design to build features\
            for training
    """

    pipeline = Pipeline([('impute', Imputer()),
                         ('scale', StandardScaler()),
                         ('feature_select', SelectFromModel(featureModel)
                          )])

    fit = pipeline.fit(design, target)
    data = fit.transform(design)
    mask = fit.named_steps['feature_select'].get_support()

    data = pd.DataFrame(data, columns=design.columns[mask])

    return data, fit


def _feature_reducer(df, thresh=5.):
    """
    """
    df[df.columns[df.sum().values > 0]]
    return _collinear_vif(df, thresh)


def _collinear_vif(df, thresh=5.):
    """
    Check for collinear features
    """

    x = df.values
    dropped = set([])
    for i in range(x.shape[1]):
        ind = i - len(dropped)
        value = vif(x, ind)
        print ind, value, x.shape
        if value > thresh:
            dropped.add(df.columns[i])
            x = np.delete(x, ind, 1)
    return df[[i for i in df if i not in dropped]]



def build_roc(X, y, clf, folds, fname):
    """
    Function to build ROC curves and save figure

    :param np.ndarray X: Design array
    :param np.ndarray y: target array
    :param estimator clf: estimator
    :param int folds: number of folds
    :param str fname: filename to save out to

    :return: auc scores
    :rtype: list
    """
    kf = KFold(n_splits=folds)

    tprs = []
    scrs = []
    base_fpr = np.linspace(0, 1, 101)

    plt.figure(figsize=(5, 5))

    for i, (train, test) in enumerate(kf.split(X,y)):
        model = clf.fit(X[train], y[train])
        y_score = model.predict_proba(X[test])
        fpr, tpr, _ = roc_curve(y[test], y_score[:, 1])
        scrs.append(roc_auc_score(y[test], y_score[:, 1]))

        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(fname)
    plt.close()
    return scrs

