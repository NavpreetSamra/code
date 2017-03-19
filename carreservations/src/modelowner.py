import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts


class ModelOwner(object):
    """
    Constructor for model owner

    Reads csvs

    Interfaces with :class:`Data` and :func:`grid_search`

    :param str resPath: path to reservations csv
    :param str vehicles: path to reservations csv
    :param tuple drops: columns to drop
    """
    def __init__(self, resPath='data/reservations.csv',
                 vehPath='data/vehicles.csv',
                 drops=('actual_price', 'street_parked', 'description')):
        self.reservations = pd.read_csv(resPath)
        self.vehicles = pd.read_csv(vehPath, index_col='vehicle_id')
        self.data = Data(self.reservations, self.vehicles, drops)
        self.model, self.grid, self.data = grid_search(self.data)


class Data(object):
    """
    Constructor for Data operations

    Aggregates and joins data

    Splits data into design/target/train/test quadrants as attributes

    :param str reservations: path to reservations csv
    :param str vehicles: path to reservations csv
    :param tuple drops: columns to drop
    :param tuple dummies: columns to dummify
    :param tuple targets: columns to exclude from design 
    :param str modelTarget: field to include in target attribute
    :param bool auto: build features and split design and target on instantiation
    """
    def __init__(self, reservations, vehicles, drops,
                 dummies=(['reservation_type']),
                 targets= ('totalReservations', 'hasReservations',
                          'reservation_type_1', 'reservation_type_2',
                          'reservation_type_3'),
                 modelTarget='totalReservations',
                 auto=True):
        """
        """
        self.reservations = reservations
        self.vehicles = vehicles
        targets = list(targets)
        if auto:
            self._build_target(list(dummies))
            self._build_features(list(drops))
            self.dfTrain, self.dfTest = tts(self.df)

        self.trainDesign = self.dfTrain[[i for i in self.dfTrain if i not in  targets]]
        self.testDesign = self.dfTest[[i for i in self.dfTest if i not in targets]]

        self.trainTarget = self.dfTrain[modelTarget]
        self.testTarget = self.dfTest[modelTarget]

    def _build_target(self, dummies):

        grouped = self.reservations.groupby('vehicle_id')
        counts = grouped.size()
        self.vehicles['totalReservations'] = 0.
        self.vehicles['totalReservations'].ix[counts.index] = counts.values
        self.vehicles['hasReservations'] =\
            (self.vehicles.totalReservations > 0.).astype(int)

        df = pd.get_dummies(self.reservations, columns=dummies)
        resCounts = df.groupby('vehicle_id').sum()
        self.df = self.vehicles.merge(
                    resCounts,
                    how='left', left_index=True,
                    right_index=True)
        self.df.fillna(0, inplace=True)

    def _build_features(self, drops):

        self.df['deltaPercent'] = ((self.df.actual_price -
                                    self.df.recommended_price) /
                                   self.df.actual_price)

        self.df.drop(drops, inplace=True, axis=1)


def grid_search(data,
                estimator=RFR(n_estimators=40),
                param_grid={"max_depth": [2, 5, 10, 15], "min_samples_split" : [20, 30, 40], "max_features": ['auto', 'sqrt', 'log2']},
                cv=5):
    """
    Build a model of type estimator with paramters prescribed by cross validated grid search. After cross validation, best estimator is built
    on parameter combination and trained on entire training set. Returns both production ready model and grid search object

    :param Data data: data object, requires (train/test)(Design/Target) attributes :py:class:`pandas.DataFrame`
    :param classifier/estimator estimator: base estimator to grid search :py:class:`sklearn.GridSearchCV`
    :param dict param_grid: paramter grid to search in grid search
    :param int cv: number of folds for cross validation

    :return: model grid data
    :rtype: tuple.(estimator, GridSearchCV, Data)
    """
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv)
    grid.fit(data.trainDesign, data.trainTarget)
    model = grid.best_estimator_.fit(pd.concat([data.trainDesign, data.testDesign]), pd.concat([data.trainTarget, data.testTarget]))
    return model, grid, data
