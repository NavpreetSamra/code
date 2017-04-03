import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from helpers import ItemSelector, build_replacements, build_param_grid


class ModelOwner(object):
    """
    Model front end splits training data off, defines grid search scoring metric, runs model

    :param `pd.DataFame` df: raw data
    :param dict test_kwargs: keyword arguments for sklearn train_test_split
    :param sklearn.metric scorer: scorer for grid search
    """
    def __init__(self, df, test_kwargs, scorer=make_scorer(f1_score)):
        train, test = tts(df, **test_kwargs)

        self.scorer = scorer
        self.cleaner = Cleaner().fit(train)

        self.designTrain = self.cleaner.transform(train)
        self.designTest = self.cleaner.transform(test)

        self.targetTrain =\
            train.first_completed_date.notnull().astype(int)

        self.targetTest = test.first_completed_date.notnull().astype(int)


        self.results = {}

    def grid_search(self, param_grid=build_param_grid(), cv=5):
        """
        Grid search estimator based on param grids

        :param dict param_grid: dict of classifiers and param grids to search see :func:`helpers.build_param_grid`
        :param int cv: number of folds for cross validation

        """

        for name, pckg  in param_grid.iteritems():
            clf = pckg['clf']
            grid =pckg['grid']

            self.results[name] = GridSearchCV(estimator=build_pipeline(clf), param_grid=grid,
                                             cv=cv, scoring=self.scorer)\
                                             .fit(self.designTrain,
                                                  self.targetTrain)


def build_pipeline(clf):
    """
    Build pipeline with vehicle year impupter, feature union and standard scaler

    :return: classifier
    :rtype: sklearn.Pipeline
    """

    features = FeatureUnion([
                             ('field_selections', ItemSelector([ 
                                 'vehicle_year_bool', 'bgc_delta', 'signup_channel_Organic',
                                 'signup_channel_Paid']))])

    pipeline = Pipeline([('build_features', features), ('scale', StandardScaler()), ('classifier', clf)])

    return pipeline


class Cleaner(BaseEstimator, TransformerMixin):
    """
    Class for cleaning and feature building

    """
    def fit(self, df, cols=(), timeCols=('bgc_date', 'signup_date'),
            lookupCols=(['signup_channel']),
            removals=None, dumCols=(['signup_channel']),
            threshFloor=tuple([['vehicle_year', 1970.]]),
            boolCols=(['vehicle_year']), replacements=build_replacements(),
            customDrops=(['signup_channel_other'])):
        """
        Fit cleaner

        :param array-like cols: keys for columns to pull directy from raw data
        :param array-like timeCols: columns requiring string to datetime manipulation
        :param array-like lookupCols: for categorical columns to establish possible values from data
        :param array-like removals: columns to remove at end if neccesary
        :param array-like dumCols: columns to dummify
        :param tuple.array-like.(str, val) threshFloor: list of columns and associated values to floor to np.nan

        :return: self
        """


        self.dfRaw = df
        self.selected = list(cols)
        self.timeCols = list(timeCols)
        self.dumCols = list(dumCols)
        self.lookupCols = list(lookupCols)
        self.threshFloor = list(threshFloor)
        self.boolCols = list(boolCols)
        self.customDrops = list(customDrops)
        self.replacements = replacements
        self.lookup = {}

        self._build_categories(lookupCols, removals)

        return self

    def _build_categories(self, lookupCols, removals):
        for col in lookupCols:
            self.lookup[col] = set(self.dfRaw[col].unique())

    def transform(self, df):
        """
        Transform df with cleaner

        :param `pandas.DataFrame df: df of data to be transformed
        
        :return: transformed data
        :rtype: `pandas.DataFrame`
        """
        newDf = pd.DataFrame(index=df.index)

        newDf = self._floor(df, newDf)
        newDf = self._build_time_cols(df, newDf)
        newDf = self._select(df, newDf)
        newDf = self._apply_categories(df, newDf)

        newDf = self._fillna(newDf)
        newDf = self._dummify(newDf)
        newDf = self._boolify(newDf)
        newDf = self._custom_drop(newDf)
        return newDf

    def _floor(self, df, newDf):
        for i in self.threshFloor:
            col, value = i
            newDf[col] = df[col]
            mask = df[col] < value
            newDf[col][mask] = np.nan
        return newDf

    def _build_time_cols(self, df, newDf):
        for i in self.timeCols:
            newDf[i] = pd.to_datetime(df[i], errors='coerce')

        newDf['bgc_delta'] = (newDf.bgc_date - newDf.signup_date).dt.days
        newDf['weekend'] = (newDf.signup_date.dt.dayofweek >= 5).astype(int)
        newDf.drop(self.timeCols, axis=1, inplace=True)
        return newDf

    def _select(self, df, newDf):
        for i in self.selected:
            newDf[i] = df[i]
        return newDf

    def _apply_categories(self, df, newDf):
        for col in self.lookupCols:
            cats = set(self.lookup[col])
            if col in self.replacements:
                cats.add(self.replacements[col])
            newDf[col] = df[col].astype('category', categories=cats)
        return newDf

    def _fillna(self, newDf):

        newDf.fillna(self.replacements, inplace=True)
        return newDf

    def _dummify(self, newDf):
        dummified = pd.get_dummies(newDf[self.dumCols], columns=self.dumCols,
                                   drop_first=True, prefix=self.dumCols, prefix_sep='_')
        newDf = newDf.merge(dummified, right_index=True, left_index=True)
        newDf.drop([i for i in self.dumCols if i in newDf], inplace=True, axis=1)
        return newDf

    def _boolify(self, newDf):
        for col in self.boolCols:
            newDf[col+'_bool'] = newDf[col].notnull().astype(int)
        return newDf

    def _custom_drop(self, newDf):
        return newDf.drop(self.customDrops, axis=1)


if __name__ == "__main__":
    pass
