import pandas as pd
# import numpy as np

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, r2_score, mean_squared_error 

from statsmodels.api import OLS

from helpers import ItemSelector,  build_param_grid


class ModelOwner(object):
    """
    Model front end splits training data off, defines
    grid search scoring metric, runs model

    :param `pandas.DataFame` df: raw data
    :param dict test_kwargs: keyword arguments for sklearn train_test_split
    :param sklearn.metric scorer: scorer for grid search
    """
    def __init__(self, df, testKwargs, outlierKwargs={},
                 # outlierKwargs={'floorThresh': [('square_footage', 0.),
                                                # ('inflatedValue', 0.)]},
                 targetField='estimated_value'):

        train, test = tts(df, **testKwargs)
        self.targetField = targetField

        cleaned = Cleaner().fit()

        train = cleaned.fit_transform(train)
        test = cleaned.transform(test)

        train = self.remove_outliers(train, self.targetField, **outlierKwargs)

        self.designTrain = train[[i for i in train if i != targetField]]
        self.targetTrain = train[targetField]
        self.designTest = test[[i for i in test if i != targetField]]
        self.targetTest = test[targetField]

        self.results = {}

    @property
    def designTrain(self):
        return self._designTrain

    @designTrain.setter
    def designTrain(self, df):
        self._designTrain = df

    @property
    def designTest(self):
        return self._designTest

    @designTest.setter
    def designTest(self, df):
        self._designTest = df

    @property
    def targetTrain(self):
        return self._targetTrain

    @targetTrain.setter
    def targetTrain(self, df):
        self._targetTrain = df

    @property
    def targetTest(self):
        return self._targetTest

    @targetTest.setter
    def targetTest(self, df):
        self._targetTest = df


    # @property
    # def outliers(self):
        # return self._outliers

    # @outliers.setter
    # def outliers(self, df):
        # self._outliers = df

    @staticmethod
    def remove_outliers(train, targetField):

            train = train.dropna()
            train = train.ix[(train.T != 0).any()]

            design = train[[i for i in train if i != targetField]]
            target = train[targetField]

            design = StandardScaler().fit_transform(design)
            model = OLS(target, design)
            mask = (model.fit().outlier_test()['student_resid'].abs() < 2)

            return train.ix[mask]

    def grid_search(self, param_grid=build_param_grid(), cv=5, scorer=make_scorer(mean_squared_error)):
        """
        Grid search estimator based on param grids

        :param dict param_grid: dict of classifiers and param grids to
            search see :func:`src.helpers.build_param_grid`
        :param int cv: number of folds for cross validation

        """

        for name, pckg in param_grid.iteritems():
            clf = pckg['clf']
            grid = pckg['grid']

            self.results[name] = GridSearchCV(estimator=\
                                              self.build_pipeline(clf),
                                              param_grid=grid,
                                              cv=cv, scoring=scorer)\
                                             .fit(self.designTrain,
                                                  self.targetTrain)

    @staticmethod
    def build_pipeline(clf, cols=('square_footage', 'inflatedValue')):
        features = FeatureUnion([('fields', ItemSelector(list(cols)))])
        pipeline = Pipeline([('build_features', features),
                             ('scale', StandardScaler()),
                             ('classifier', clf)])
        return pipeline


class Cleaner(BaseEstimator, TransformerMixin):
    def fit(self, df=None, cpiArgs=('../data/all_cpi.csv',
                               'DATE', 'cpiDate', 'CPIHOSNS', 'cpiVal')
            ):
        """
        """
        self.cpiDate = cpiArgs[2]
        self.cpiVal = cpiArgs[4]
        self._read_cpi(*cpiArgs)
        self._build_inflation()

        return self

    def _read_cpi(self, cpiFPath, timeCol, timeRename, cpiCol, cpiRename):
        df = pd.read_csv(cpiFPath)
        self.cpi = pd.DataFrame(index=df.index,
                                columns=[timeRename, cpiRename])

        self.cpi[timeRename] = month_year_ftime(df[timeCol])
        self.cpi[cpiRename] = df[cpiCol]

    def _build_inflation(self):
        """
        """
        self.currentCpi = float(self.cpi.sort_values(self.cpiDate)[self.cpiVal].iloc[-1])
        self.minCpi = self.cpi.sort_values(self.cpiDate)[self.cpiVal].iloc[0]

        self.cpi['inflation'] = 1 + (self.currentCpi - self.cpi[self.cpiVal]) / self.cpi[self.cpiVal]

    def transform(self, df, cols=('estimated_value', 'last_sale_amount'),
                  powers=tuple([('square_footage', 1.81, 1000.)]),
                  timeCols=('last_sale_date'),
                  histCol=('last_sale_amount'),
                  dtCols=('year_built'),
                  drops=('last_sale_date','last_sale_amount')
                  ):
        """
        """
        timeCols = _build_key(timeCols)
        histCol = _build_key(histCol)
        cols = _build_key(cols)
        drops = _build_key(drops)

        transformedDf = pd.DataFrame(index=df.index)

        transformedDf = self._pullRaw(df, transformedDf, cols)
        transformedDf = self._apply_power(df, transformedDf, powers)

        transformedDf = self._convert_time(df, transformedDf, timeCols)

        transformedDf = self._inflate(df, transformedDf,
                                      timeCols, histCol)

        transformedDf = self._diffTime(df, transformedDf, dtCols)

        transformedDf = self._custom_drops(transformedDf, drops)

        return transformedDf

    @staticmethod
    def _pullRaw(df, transformedDf, cols):
        """
        """
        for col in cols:
            transformedDf[col] = df[col]

        return transformedDf


    @staticmethod
    def _apply_power(df, transformedDf, powers):
        for power in powers:
            col, val, scale = power
            transformedDf[col] = df[col]**val / scale

        return transformedDf

    @staticmethod
    def _convert_time(df, transformedDf, timeCols):
        for col in timeCols:
            transformedDf[col] = month_year_ftime(df[col])
        return transformedDf

    def _inflate(self, df, transformedDf, timeCol, histCol):
        """
        """
        timeCol = _build_key(timeCol)[0]
        histCol = _build_key(histCol)[0]

        cpiDf = transformedDf.reset_index().merge(self.cpi, left_on=timeCol,
                                    right_on=self.cpiDate,
                                    how='left').set_index('index')

        # cpiDf.fillna({self.cpiVal: self.minCpi}, inplace=True)

        transformedDf['inflatedValue'] = cpiDf['inflation'] * df[histCol]

        return transformedDf

    @staticmethod
    def _diffTime(df, transformedDf, dtCols):
        dtCols = _build_key(dtCols)[0k
        deltaT = pd.to_datetime('today') - pd.to_datetime(df[dtCols], errors='coerce')
        transformedDf['dt'] = deltaT.dt.days

        return transformedDf
   

    @staticmethod
    def _custom_drops(df, drops):
        return df.drop(drops, axis=1)


def month_year_ftime(pdSeries):
    """
    Return month-year from :py:class:`pandas.Series` with date information
    (will convert strings to datetime objects

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
    elif isinstance(cols, tuple):
        cols = list(cols)
    elif not isinstance(cols, list):
        raise Exception('cols not valid key type')

    return cols
