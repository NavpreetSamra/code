import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts
import numpy as np


class Features(object):
    """
    Class for cleaning and transforming order and trip data for modeling. Generates design and test :py:class:`pandas.DataFrame` attributes for modeling
    
    :param str orderPath: path to orders csv
    :param str tripsPath: path to trips csv
    :param tuple dumCols: columns to dummify
    :param bool trainSet: switch for operating on training data (with target)
    :param pandas.DataFrame templateDf: template from data frame from training to prescribe format of test data
    :param float percent: percent threshold for margnial distribution (department_name) transformation 
    :param dict tt_split: keyword arguments for train test split
    :param bool split: switch for train test split
    :param bool test: switch for test data
    """
    def __init__(self, orderPath, tripsPath, columns, dumCols=(), trainSet=True, templateDf=None, percent=.1, tt_split=None, split=False, test=False):
        self.orderDf = pd.read_csv(orderPath)
        self.tripsDf = pd.read_csv(tripsPath, index_col='trip_id')
        self.orderDf = self.orderDf.ix[np.in1d(self.orderDf.trip_id, self.tripsDf.index)]
        self.convert_time(trainSet)
        self.create_trips(columns)
        self.build_dummies(dumCols)
        if split:
            if not tt_split:
                tt_split = {}
            self.train_test_split(tt_split)
        if templateDf is not None:
            self.testDf = pd.DataFrame(data=0,index=self.tripsDf.index, columns=templateDf.columns)
            for i in self.testDf:
                if i in self.design:
                    self.testDf[i] = self.design[i]

    def train_test_split(self, kwargs):
        """
        train test split on data from sklearn

        :param dict kwargs: keyword arguments for train test split
        """
        self.design, self.designTest, self.target, self.targetTest = tts(self.design, self.target, **kwargs)

    def build_dummies(self, cols):
        """
        Dummify prescribed columns

        :param tuple cols: columns to dummify
        """
        for i in cols:
            dummies = pd.get_dummies(self.tripsDf[i], prefix=i, prefix_sep='_', drop_first=True)
            self.design = pd.concat([self.design, dummies], axis=1)

    def convert_time(self, trainSet):
        """
        Convert start (end if applicable) timestamps to seconds. Builds day of week, hour of day features\
                and target (dt in seconds)

        :param bool trainSet: switch for train set to build target
        """
        start = self.tripsDf.shopping_started_at
        start = pd.to_datetime(start)
        self.tripsDf.shopping_started_at = start
        self.tripsDf['dow'] = (start.dt.dayofweek < 5).astype(int)
        self.tripsDf['hod'] = start.dt.hour
        if trainSet:
            end = self.tripsDf.shopping_ended_at
            end = pd.to_datetime(end)
            self.tripsDf.shopping_ended_at = end
            self.target = (end - start).dt.seconds


    def create_trips(self, columns):
        """
        Filter trips table columns for design matrix
        Aggregate orders table data (sum(quantity) and len(unique(department names) for trip)\
                and join to design matrix (keyed off trip id)

        :param list columns: columns to include from trips data in design
        """

        self.design = self.tripsDf[[i for i in columns if i in self.tripsDf]]
        grouped = self.orderDf.groupby('trip_id')
        self.design['total'] = grouped.sum()['quantity']
        self.design['counts'] = grouped.size()
	
    def _marginal_clean(self, percent=None):
	if percent:
	    depts = self.orderDf.department_name
	    marginalDist = depts.value_counts()
	    total = sum(depts.notnull())
	    mask= [(float(i)/total < percent/100.) for i in marginalDist]
	    inclusions = marginalDist.index[mask]
	    depts[depts.isin(inclusions)] = 'marginal_dist'
        else:
            pass


def grid_search(data,
                estimator=RFR(n_estimators=40),
                param_grid={"max_depth": [40, 50, 60], "min_samples_split" : [40, 50, 60], "max_features": ['auto', 'sqrt', 'log2']},
                cv=5):
    """
    Build a model of type estimator with paramters prescribed by cross validated grid search. After cross validation, best estimator is built
    on parameter combination and trained on entire training set. Returns both production ready model and grid search object

    :param Features data: data object, requires design, target, and designTest targetTest attributes :py:class:`pandas.DataFrame`
    :param classifier/estimator estimator: base estimator to grid search :py:class:`sklearn.GridSearchCV`
    :param dict param_grid: paramter grid to search in grid search
    :param int cv: number of folds for cross validation

    :return: model grid data
    :rtype: tuple.(estimator, GridSearchCV, Features)
    """
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv)
    grid.fit(data.design, data.target)
    model = grid.best_estimator_.fit(pd.concat([data.design, data.designTest]), pd.concat([data.target, data.targetTest]))
    return model, grid, data


def main(model=None, grid=None, grid_kwargs=None, data=None, template=None):
    """
    Main function for training model to create predictions

    :param classifier/estimator model: base estimator to grid search :py:class:`sklearn.GridSearchCV`
    :param sklearn.GridSearchCV grid: grid searched object
    :param dict grid_kwargs: keyword arguments for :py:class:`sklearn.GridSearchCV`
    :param Features data: data object, requires design, target, and designTest targetTest attributes :py:class:`pandas.DataFrame`
    :param Features template: train object template to create test design

    :return: model grid 
    :rtype: tuple.(estimator, GridSearchCV)
    """
    if not model:
        if not grid_kwargs:
            grid_kwargs = {}
        model, grid, template= grid_search(**grid_kwargs)
    if not data:
        data = Features('../data/order_items.csv', '../data/test_trips.csv', ['shopper_id'], trainSet=False, templateDf=template.design)
    predictions = pd.DataFrame(data=model.predict(data.testDf).astype(int), index=data.testDf.index, columns=['time'])
    predictions.to_csv('predictions.csv')
    return model, grid

if __name__ == "main":
    pass
