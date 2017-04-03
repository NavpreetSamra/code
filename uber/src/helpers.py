import pandas as pd
import numpy as np
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

def build_replacements():
    """
    Helper function to fill NaNs

    :return: taxonomy
    :rtype: dict
    """
    d = {'bgc_delta': 90.,
         'signup_os': 'other',
         'signup_channel': 'other',
         'vehicle_make': 'other',
         'vehicle_model': 'other',
         }
    return d

def build_param_grid():
    """
    Helper function to build parameter grid for hyperparameter tuning
    
    :return: classifiers and parameter grids
    :rtype: dict
    """
    g = {'lr': {'clf': LogisticRegression(class_weight='balanced'), 'grid': {'classifier__C': np.logspace(-4,3,8)}},
         'rf': {'clf': RandomForestClassifier(n_estimators=60),
                'grid': {'classifier__max_depth': [2,3,4],
                         'classifier__min_samples_split': [40,50,60]}
                }

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

class Data(object):
    """
    Class for exploring raw data
    """
    def __init__(self, df, target=('first_completed_date'), exclusions=(['id']),
                 timeCols=('signup_date', 'bgc_date'),
                 lookupCols=('city_name', 'signup_os', 'signup_channel'),
                 dumCols=('city_name', 'signup_os', 'signup_channel'),
                 thresh=tuple([['vehicle_year', 1970.]]),
                 boolCols=tuple(['vehicle_year']),
                 replacements=build_replacements(), isTest=False, template=None, lookup=None, auto=True):
        self.dfRaw = df[[i for i in df if i not in exclusions]]
        if isTest:
            self.df = pd.DataFrame(index=range(df.shape[0]), columns=template.columns)
        self.df = pd.DataFrame(index=df.index)
        if lookup:
            self.lookup = lookup
        else:
            self.lookup = {}

        self.timeCols = list(timeCols)
        self.dumCols = list(dumCols)
        self.lookupCols = list(lookupCols)
        self.template = template


        if auto:
            # !!!TODO move to obj[name].__func__
            if timeCols:
                self._convert_time()
            if not isTest:
                self._generate_lookup(lookupCols)
            if thresh:
                self._floor(thresh)
            if replacements:
                self._fillna(replacements)
            if lookupCols:
                self._apply_lookup(lookupCols, replacements)
            if dumCols:
                self._dummify(dumCols)
            if boolCols:
                self._boolify(boolCols)
            self.df[target] = self.dfRaw[target].notnull().astype(int)

    def _convert_time(self):
        for i in self.timeCols:
            self.df[i] = pd.to_datetime(self.dfRaw[i], errors='coerce')

        self.df['bgc_delta'] = (self.df.bgc_date - self.df.signup_date).dt.days

        self.df.drop(self.timeCols, axis=1, inplace=True)

    def _generate_lookup(self, lookupCols):
        for col in lookupCols:
            self.lookup[col] = self.dfRaw[col].unique()

    def _apply_lookup(self, lookupCols, replacements=()):
        for col in lookupCols:
            cats = list(self.lookup[col])
            if col in replacements and replacements[col] not in cats:
                cats += [replacements[col]]
            self.dfRaw[col] = self.dfRaw[col].astype('category', categories=cats)


    def _floor(self, threshFloor):
        for i in threshFloor:
            col, value = i
            self.df[col] = self.dfRaw[col]
            mask = self.dfRaw[col] < value
            self.df[col][mask] = np.nan

    def _fillna(self, replacements):
        self.df.fillna(replacements, inplace=True)

    def _boolify(self, boolCols):
        for col in boolCols:
            self.df[col] = self.df[col].notnull().astype(int)

    def _dummify(self, dumCols):
        dummified = pd.get_dummies(self.dfRaw[list(dumCols)], columns=dumCols, prefix=dumCols, prefix_sep='_')
        self.df = self.df.merge(dummified, right_index=True, left_index=True)
        self.df.drop([i for i in dumCols if i in self.df], inplace=True, axis=1)

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
    kf = KFold(n=len(y), n_folds=folds)

    tprs = []
    scrs = []
    base_fpr = np.linspace(0, 1, 101)

    plt.figure(figsize=(5, 5))

    for i, (train, test) in enumerate(kf):
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
