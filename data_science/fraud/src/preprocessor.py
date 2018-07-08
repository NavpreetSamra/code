import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, roc_auc_score

from sklearn.svm import LinearSVC

from imblearn.combine import SMOTEENN

from helpers import _build_key, build_param_grid, select_transform_features


class ModelOwner(object):
    """
    Class for owning modeling infrastructure. Merges Company and Risk data.
    Train Test Splits. Oversamples.

    :param str companiesPath: path to company table csv
    :param str riskPath: path to risk table csv
    :param str target: target column for prediction
    :param `imblearn` oversampler: imblearn oversampler object with fit_sample\
            (X, y) method
    :param dict testKwargs: sklearn train_test_split keyword arguments
    :param bool auto: auto run build and evaluate methods on instantiation of\
            self
    """
    def __init__(self, companiesPath, riskPath,  target='is_fraud',
                 oversampler=SMOTEENN(),
                 testKwargs={'test_size': .3, 'random_state': 24}, auto=True):

        rawData = Reader(companiesPath, riskPath)
        self.fitRiskData = RiskData().fit(rawData.dfRisk)
        dfRisk = self.fitRiskData.transform(rawData.dfRisk)
        df = rawData.dfCompanies.merge(dfRisk, how='left', on='company_id')

        train, test = tts(df, stratify=df[target], **testKwargs)

        self.partition_data(train, test, target, oversampler)

        self.results = {}
        self.estimators = {}
        self.featureEval = None

        if auto:
            self.build(build_param_grid())
            self.evaluate()

    def build(self, paramGrid,  featureModel=LinearSVC(C=0.0006, penalty="l1",
                                                       dual=False)
              ):

        """
        Select signifigant feautres for model with *featureModel* and 
        Build cross validated estimator based on *paramGrid,* and store
        **results** and **estimators** attributes.

        For more information see :py:func:`gusto.src.helpers.build_param_grid`

        :param dict paramGrid: dictionary of models to run with grids\
                for cross validation
        :param sklearn.estimator featureModel: Model for selecting signifigant feautures
        """

        self.trainData, cleaner = select_transform_features(self.designTrain,
                                                            self.targetTrain,
                                                            featureModel
                                                            )
        self.testData = cleaner.transform(self.designTest)
        self.results, self.estimators =\
            self.grid_search(self.trainData, self.targetTrain, paramGrid)

    def evaluate(self):
        """
        Build `pandas.DataFrame` of relative features weights for models with coef_\
                or feature_importances attribute

        Populates featureEval attribute.
        """

        self.featureEval = pd.DataFrame(index=self.estimators.keys(), columns=self.trainData.columns)
        for name, clf in self.estimators.iteritems():
            if hasattr(clf, 'feature_importances_'):
                values = clf.feature_importances_
            if hasattr(clf, 'coef_'):
                values = clf.coef_ / float(pd.DataFrame(clf.coef_).abs().sum(axis=1).values)
            self.featureEval.loc[name] = values

    @property
    def designTrain(self):
        """
        Design Matrix for training

        :rtype: :py:class:`pandas.DataFrame`
        """
        return self._designTrain

    @designTrain.setter
    def designTrain(self, df):
        self._designTrain = df

    @property
    def designTest(self):
        """
        Design Matrix for testing

        :rtype: :py:class:`pandas.DataFrame`
        """
        return self._designTest

    @designTest.setter
    def designTest(self, df):
        self._designTest = df

    @property
    def targetTrain(self):
        """
        Target array for training

        :rtype: :py:class:`pandas.Series`
        """
        return self._targetTrain

    @targetTrain.setter
    def targetTrain(self, df):
        self._targetTrain = df

    @property
    def targetTest(self):
        """
        Target array for testing

        :rtype: :py:class:`pandas.Series`
        """
        return self._targetTest

    @targetTest.setter
    def targetTest(self, df):
        self._targetTest = df

    @property
    def trainData(self):
        """
        Fit and transformed data for model training

        :rtype: :py:class:`pandas.DataFrame`
        """
        return self._trainData

    @trainData.setter
    def trainData(self, df):
        self._trainData = df

    @property
    def testData(self):
        """
        Fit and transformed data for modeling testing

        :rtype: :py:class:`pandas.DataFrame`
        """
        return self._testData

    @testData.setter
    def testData(self, df):
        self._testData = df

    def partition_data(self, train, test, target, oversampler):
        """
        Split data into design :py:class:`pandas.DataFrame` and target\
                :py:class:`pandas.Series` for along train test split

        :param pandas.DataFrame train: data for training
        :param pandas.DataFrame test: data for testing
        :param str target: target from train/ test :py:class:`pandas.DataFrame`
        :param `imblearn` oversampler: Oversampler object with fit_sample method
        """

        designTrain = train[[i for i in train if i != target]]
        targetTrain = train[target]

        self.designTrain, self.targetTrain =\
            oversampler.fit_sample(designTrain, targetTrain)
        self.designTrain = pd.DataFrame(self.designTrain,
                                        columns=designTrain.columns)
        self.targetTrain = pd.Series(self.targetTrain, name=targetTrain.name)

        self.designTest = test[[i for i in test if i != target]]
        self.targetTest = test[target]

    @staticmethod
    def grid_search(designTrain, targetTrain, param_grid=build_param_grid(),
                    cv=3, scorer=make_scorer(roc_auc_score)):
        """
        Grid search estimator based on param grids

        :param key-like cols: columns to pull from design to build features\
                for training
        :param dict param_grid: dict of classifiers and param grids to
            search see :func:`gusto.src.helpers.build_param_grid`
        :param int cv: number of folds for cross validation
        :param sklearn.metric scorer: scorer for grid search

        :return: results- dict of GridSearchCV obejcts from hyperparameter tuning
        :return: estimators- dict of hyperparameter tuned estimators fit to the training data
        :rtype: tuple.dict

        """

        results = {}

        for name, pckg in param_grid.iteritems():
            clf = pckg['clf']
            grid = pckg['grid']

            results[name] = GridSearchCV(estimator=clf,
                                         param_grid=grid,
                                         cv=cv, scoring=scorer
                                         )\
                                .fit(designTrain,
                                     targetTrain)
        estimators = {}
        for name, result in results.iteritems():
            estimators[name] = param_grid[name]['clf']\
                              .set_params(**result.best_params_)\
                              .fit(designTrain, targetTrain)

        return results, estimators


class Reader(object):
    """
    Class for reading data and 

    :param str companiesPath: path to company table csv
    :param str riskPath: path to risk table csv
    :param bool auto: auto run risk and company modifiers
    """
    def __init__(self, companiesPath, riskPath,  auto=True):
        self.companiesPath = companiesPath
        self.riskPath = riskPath

        if auto:
            self._run_risk()
            self._run_companies()

    def _run_risk(self,
                  dumCols=('signal_group', 'metric_type', 'score_value')
                  ):

        self.dfCompanies = pd.read_csv(self.companiesPath)
        self.dfRisk = pd.read_csv(self.riskPath)

        dumCols = _build_key(dumCols)
        # Composit key => less overhead during dummification
        self.dfRisk['key'] = self.dfRisk[dumCols]\
                             .fillna('NaN')\
                             .apply(lambda x: "_".join(x), axis=1)

        self.dfRisk.drop(dumCols, inplace=True, axis=1)

    def _run_companies(self, bools=(['bank_account_type', 'Checking'],)):
        for col, val in bools:
            self.dfCompanies[col] = (self.dfCompanies[col] == val)\
                                    .astype(int)
        self.dfCompanies['percent_state'] = (self.dfCompanies.num_ee_state_matches.astype(float)
            / self.dfCompanies.num_employees).fillna(0)
        self.dfCompanies['has_employees'] = (self.dfCompanies.num_employees > 0).astype(int)


class RiskData(BaseEstimator, TransformerMixin):
    """
    Transformer for Risk Table
    """
    def fit(self, df, keyCol='key'):
        """
        :param `pandas.DataFrame` df: risk metrics  table from :class:`Reader`
        :param keyCol: columns to fit

        :return: self
        """

        self.keyCategories = df[keyCol].unique()
        return self

    def transform(self, df, groupId=('company_id',), groupTarget=('key',),
                  aggTarget='n_scores'
                  ):
        """
        Transform Risk Data table

        :param `pandas.DataFrame` df: risk metrics  table from :class:`Reader`
        :param key-like groupId: primary key
        :param key-like groupTarget: aggregation key target field
        :param str aggTarget: aggreagation  summation field

        :return: transformed df
        :rtype: `pandas.DataFrame`
        """
        groupId = _build_key(groupId)
        groupTarget = _build_key(groupTarget)
        groupKey = groupId + groupTarget

        data = df.groupby(groupKey).sum()[[aggTarget]].reset_index()
        data = pd.get_dummies(data, columns=groupTarget)
        aggValues = data.pop(aggTarget)
        companies = data.pop(groupId[0])
        data = data.multiply(aggValues, axis='index')
        data = pd.concat([companies, data], axis=1)

        aggregated = data.groupby(groupId).sum().reset_index()
        return aggregated
