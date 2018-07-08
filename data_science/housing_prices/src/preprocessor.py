import pandas as pd
import numpy as np

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, r2_score # , mean_squared_error

from statsmodels.api import OLS

from helpers import _build_key, month_year_ftime

from helpers import ItemSelector, build_param_grid, build_train_transform_kwargs, build_transform_kwargs


class ModelOwner(object):
    """
    Model front end splits training data off, defines
    grid search scoring metric, runs model

    :param `pandas.DataFame` df: raw data
    :param dict testKwargs: keyword arguments for sklearn train_test_split
    :param dict outlierKwargs: keyword arguments for outlier removal 'dropVal' replacement and 'studentResid' thresholding
    :param str targetField: target from cleaned :py:class:`pandas.DataFrame` for fitting
    """
    def __init__(self, df, testKwargs={'test_size': .2, 'random_state': 22},
                 outlierKwargs={'dropVal': 0, 'studentResid': 2},
                 targetField='estimated_value'):

        train, test = tts(df, **testKwargs)
        self.targetField = targetField

        self.cleaner = Cleaner().fit(train)

        train = self.cleaner.transform(train)
        test = self.cleaner.transform(test)

        train = self.remove_outliers(train, self.targetField, **outlierKwargs)

        self.partition_data(train, test, targetField)

        self.results = {}

    @property
    def designTrain(self):
        """"
        Design Matrix for training

        :rtype: :py:class:`pandas.DataFrame`
        """
        return self._designTrain

    @designTrain.setter
    def designTrain(self, df):
        self._designTrain = df

    @property
    def designTest(self):
        """"
        Design Matrix for testing

        :rtype: :py:class:`pandas.DataFrame`
        """
        return self._designTest

    @designTest.setter
    def designTest(self, df):
        self._designTest = df

    @property
    def targetTrain(self):
        """"
        Target array for training

        :rtype: :py:class:`pandas.Series`
        """
        return self._targetTrain

    @targetTrain.setter
    def targetTrain(self, df):
        self._targetTrain = df

    @property
    def targetTest(self):
        """"
        Target array for testing

        :rtype: :py:class:`pandas.Series`
        """
        return self._targetTest

    @targetTest.setter
    def targetTest(self, df):
        self._targetTest = df

    def partition_data(self, train, test, targetField):
        """
        Split data into design :py:class:`pandas.DataFrame` and target :py:class:`pandas.Series` for along train test split

        :param pandas.DataFrame train: data for training
        :param pandas.DataFrame test: data for testing
        :param str targetField: target from train/ test :py:class:`pandas.DataFrame`
        """

        self.designTrain = train[[i for i in train if i != targetField]]
        self.targetTrain = train[targetField]
        self.designTest = test[[i for i in test if i != targetField]]
        self.targetTest = test[targetField]

    @staticmethod
    def remove_outliers(train, targetField, dropVal, studentResid, verbose=True):
        """
        Remove outliers from training data based on statsmodels OLS Fit studentized residuals and specified drop values across features

        :param pandas.DataFrame train: data for training
        :param str targetField: target from train/ test :py:class:`pandas.DataFrame`
        :param obj dropVal: value to drop rows across
        :param float studentResid: number to threshold absolute value of student residuals above
        :param bool verbose: flag to print out OLS summary information and number of outlier removed
        """

        train = train.dropna()
        if dropVal is not None:
            train = train.ix[(train.T != dropVal).all()]

        design = train[[i for i in train if i != targetField]]
        target = train[targetField]

        design = StandardScaler().fit_transform(design)
        model = OLS(target, design)
        mask = np.ones((train.shape[0])).astype(bool)
        if studentResid is not None:
            mask = (model.fit().outlier_test()['student_resid'].abs() < 2)

        if verbose:
            print model.fit().summary()
            print 'Removed:' + str(train.shape[0] - sum(mask))

        return train.ix[mask]

    def grid_search(self, cols, param_grid=build_param_grid(),
                    cv=5, scorer=make_scorer(r2_score)):
        """
        Grid search estimator based on param grids

        :parm key-like cols: columns to pull from design to build features for training
        :param dict param_grid: dict of classifiers and param grids to
            search see :func:`windfall.src.helpers.build_param_grid`
        :param int cv: number of folds for cross validation
        :param sklearn.metric scorer: scorer for grid search
        """

        for name, pckg in param_grid.iteritems():
            clf = pckg['clf']
            grid = pckg['grid']
            pipeline = self.build_pipeline(clf, cols)

            self.results[name] = GridSearchCV(estimator=pipeline,
                                              param_grid=grid,
                                              cv=cv, scoring=scorer
                                              )\
                                             .fit(self.designTrain,
                                                  self.targetTrain)



    @staticmethod
    def build_pipeline(clf, cols):
        """
        Build :py:class:`sklearn.Pipeline` with feature union of cols followed by Imputer StandardScaler and an estimator

        :param sklearn.estimator clf: estimator
        :parm key-like cols: columns to pull from design to build features for training
        """

        features = FeatureUnion([('fields', ItemSelector(_build_key(cols)))])
        pipeline = Pipeline([('build_features', features),
                             ('impute', Imputer()),
                             ('scale', StandardScaler()),
                             ('classifier', clf)])
        return pipeline


class Cleaner(BaseEstimator, TransformerMixin):
    """
    Class for cleaning raw housing attribute data
    """
    def fit(self, df, cpiArgs=('../data/all_cpi.csv',
                               'DATE', 'cpiDate', 'CPIHOSNS', 'cpiVal'),
            lookupCols=('zipcode')

            ):
        """

        Fit method note transformer must be called on original data to aggregate zip information, this will be fixed in subsequent versions

        :param `pandas.DataFrame` df: DataFrame to dit
        :param array-like cpiArgs: (path to cpi data, original date column, date column rename, original cpi column, cpi column rename)
        :param key-like lookupCols: columns to store unique values for in self.lookups
        """
        self.cpiDate = cpiArgs[2]
        self.cpiVal = cpiArgs[4]
        self.lookups = {}

        self._read_cpi(*cpiArgs)
        self._build_inflation()
        self._build_categories(df, lookupCols)

        self.isTransformed = False

        return self

    def _read_cpi(self, cpiFPath, timeCol, timeRename, cpiCol, cpiRename):
        """
        """
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

    def _build_categories(self, df, lookupCols):
        """
        """
        lookupCols = _build_key(lookupCols)
        for col in lookupCols:
            self.lookups[col] = df[col].unique()

    def transform(self, df, transformKwargs=build_transform_kwargs(), dropCols=('last_sale_date', 'last_sale_amount'), customZips=True):
        """
        Tranformer for clening raw housing data

        :param pandas.DataFame df: raw data
        :param dict transformKwargs: dict of transformation operations
            see :func:`windfall.src.helpers.build_transform_kwargs`
            and :func:`windfall.src.helpers.build_train_transform_kwargs`

        :param key-like dropCols: columns to drop from data
        :param bool customZips: switch for leveraging aggregated historical sale zipcode data
        """
        transformedDf = pd.DataFrame(index=df.index)

        for operation, kwargs, in transformKwargs.iteritems():
            transformedDf = self.__getattribute__(operation)(df, transformedDf, **kwargs)

        if customZips:
            transformedDf = self._custom_zips(df, transformedDf)
        transformedDf = self._custom_drops(df, transformedDf, _build_key(dropCols))

        if not self.isTransformed:
            self.isTransformed = True
        return transformedDf

    @staticmethod
    def _pull_raw(df, transformedDf, cols):
        """
        """
        for col in cols:
            transformedDf[col] = df[col]

        return transformedDf

    @staticmethod
    def _apply_power(df, transformedDf, cols=None, powers=None, scales=None):
        for col, val, scale in zip(cols, powers, scales):
            transformedDf[col] = df[col]**val / scale

        return transformedDf

    def _inflate(self, df, transformedDf, timeCol=None, histCol=None):
        """
        """
        timeCol = _build_key(timeCol)[0]
        histCol = _build_key(histCol)[0]
        transformedDf[timeCol] = month_year_ftime(df[timeCol])

        cpiDf = transformedDf.reset_index().merge(self.cpi, left_on=timeCol,
                                                  right_on=self.cpiDate,
                                                  how='left')\
                                           .set_index('index')

        transformedDf['inflatedValue'] = cpiDf['inflation'] * df[histCol]

        return transformedDf

    @staticmethod
    def _custom_drops(df, transformedDf, dropCols=None):
        return transformedDf.drop(dropCols, axis=1)

    def _build_categoricals(self, df, transformedDf, categoricalCols=None):
        categoricalCols = _build_key(categoricalCols)
        for col in categoricalCols:
            series = df[col].astype('category', categories=self.lookups[col])
            dummified = pd.get_dummies(series, prefix=col, drop_first=True)
            transformedDf = pd.concat([transformedDf, dummified], axis=1)

        return transformedDf

    def _custom_zips(self, df, transformedDf):
        if 'zipcode' not in transformedDf:
            transformedDf['zipcode'] = df['zipcode']
        if not self.isTransformed:
            self.zipavg = transformedDf.ix[transformedDf.inflatedValue > 0].groupby('zipcode').mean()[['inflatedValue']]
            self.zipavg['zipValue'] = self.zipavg.inflatedValue
            self.zipavg.drop('inflatedValue', inplace=True, axis=1)
        zipDf = transformedDf.reset_index().merge(self.zipavg, left_on='zipcode',
                                                                    right_index=True,
                                                                    how='left')\
                                                             .set_index('index')
        transformedDf['zipAvg'] = zipDf['zipValue'].fillna(0)

        transformedDf.drop('zipcode', inplace=True, axis=1)
        transformedDf['inflatedValue'] -= transformedDf['zipAvg']
        return transformedDf

    @staticmethod
    def _build_quarters(df, transformedDf, cols=None):
        transformedDf['quarters'] = ((pd.to_datetime(df.last_sale_date).dt.quarter - 2.5).abs() == .5).astype(int)
        return transformedDf

    @staticmethod
    def _diffTime(df, transformedDf, dtCol=None):
        dtCol = _build_key(dtCol)[0]
        deltaT = pd.to_datetime('today') - pd.to_datetime(df[dtCol], errors='coerce')
        transformedDf['dt'] = deltaT.dt.days

        return transformedDf
