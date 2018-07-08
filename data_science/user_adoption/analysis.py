import pandas as pd
import numpy as np
from datetime import timedelta
import itertools as it
from sklearn.cross_validation import KFold
import networkx as nx
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class Cleaned(object):
    """
    Class for pulling, cleaning, and aggregating data

    For details inspect embedded source of encapsulated methods

    :param str/pd.DataFrame fUsers: Path or DataFrame of user data
    :param str/pd.DataFrame fEngage: Path or DataFrame of user engagement data
    :param tuple.(int,int) adoption: (number of logins, within window)
    :param array-like drops: names of columns to drop


    TODO: list attributes
    """
    def __init__(self, fUsers, fEngage, adoption=(3, 7),
                 drops=['name', 'email']):

        self.fUsers = fUsers
        self.fEngage = fEngage
        self.adoption = {'hits': adoption[0],
                         'window': timedelta(days=adoption[1])}
        self.drops = drops
        self.dfs = {}

        self._pull_data()
        self._dummify_source()
        self._split_non_users()
        self._convert_user_time()
        self._evaluate_users()
        self._merge_non_users()

    def _pull_data(self):
        """
        Pull, partition, drop
        """
        if isinstance(self.fUsers, str):
            self.dfs['users'] = pd.read_csv(self.fUsers)
        elif isinstance(self.fUsers, pd.DataFrame):
            self.dfs['users'] = self.fUsers
        else:
            raise Exception('fUsers not path or DataFrame')

        if isinstance(self.fEngage, str):
            self.dfs['engage'] = pd.read_csv(self.fEngage)
        elif isinstance(self.fEngage, pd.DataFrame):
            self.dfs['engage'] = self.fEngage
        else:
            raise Exception('fEngage not path or DataFrame')

        if self.drops:
            self.dfs['users'].drop(self.drops, inplace=True, axis=1)

        columnsRename = [i if i != 'last_session_creation_time' else
                         'last_ses' for i in self.dfs['users'].columns]

        self.dfs['users'].columns = columnsRename
        self.dfs['users'].set_index('object_id', drop=False, inplace=True)
        self.dfs['users']['local_rank'] = 0

        self.dfs['users']['creation_time'] =\
            pd.to_datetime(self.dfs['users']['creation_time']).dt.date
        self.dfs['users']['adopted'] = False

    def _dummify_source(self):
        """
        Create dummified creation_source DataFrame\
                Store in as sources in dfs attribute
        """
        self.dfs['sources'] = pd.get_dummies(self.dfs['users']\
                                                     ['creation_source'])

    def _split_non_users(self):
        """
        Split accounts that have never logged in to separate df
        """
        mask = self.dfs['users']['last_ses'].isnull()
        self.dfs['non_users'] = self.dfs['users'].loc[mask]
        self.dfs['users'] = self.dfs['users'].loc[np.invert(mask)]

    def _convert_user_time(self):
        """
        Convert time_stamp creation_time and last_session_creaetd\
                fields of pd.DF to DateTime objects

        NB this does not clean non_user DataFrame time stamps
        """
        self.dfs['users']['last_ses'] =\
            pd.to_datetime(self.dfs['users']['last_ses'].astype(int),
                           unit='s'
                           ).dt.date


        self.dfs['engage']['time_stamp'] =\
            pd.to_datetime(self.dfs['engage']['time_stamp']).dt.date

    def _evaluate_users(self):
        """
        Evaluate whether or not a user meets adoption criteria
        """
        user_history = self.dfs['engage'].groupby('user_id')

        for user, df in user_history:
            if evaluate_logins(sorted(df.time_stamp.values), self.adoption):
                self.dfs['users']['adopted'].loc[user] = 1

    def _merge_non_users(self):
        """
        Merge non users back into users df
        """
        self.dfs['users'] = pd.concat([self.dfs['users'],
                                       self.dfs['non_users']])
        del self.dfs['non_users']

class AdoptionModel(object):
    """
    Class for evaluating models on :class:FeatureAnalysis objects

    :param FeatureAnalysis cleaned: Cleaned data with engineered features
    :param sklearn model: scikit learn model, currently works with Logistic Regression, DecistionTree\
            Classifier and RandomForest Classifier
    :param bool isLogistic: boolean for model type. note this notation due to different access points\
            for coefficients and feature importances in fit models
    :param bool isTree: boolean for model type.
    :param bool isForest: boolean for model type.
    :param int folds: Number of folds to run in cross validation
    :param float vifMagnitude: threshold for variance inflation factor
    :param dict autoRun: dictionary for automating runs, examples and front end INDEV
    """
    def __init__(self, cleaned, model, isLogistic=False, isTree=False, isForest=False, folds=10,
                 vifMagnitude=3.0,
                 autoRun={'dfs': [('users', ['opted_in_to_mailing_list',
                                             'enabled_for_marketing_drip',
                                             'org_size', 'local_rank',
                                             'children'
                                             ]),
                                  ('sources', ['GUEST_INVITE', 'ORG_INVITE',
                                               'PERSONAL_PROJECTS', 'SIGNUP'])],
                          'var_scale': ['org_size', 'children']
                          }
                 ):
        self.cleaned = cleaned
        self.model = model
        self.isLogistic = isLogistic
        self.isTree = isTree
        self.isForest = isForest
        self.folds = folds
        self.vifMagnitude = vifMagnitude
        self.df = pd.DataFrame(index=self.cleaned.dfs['users'].index)
        self.collinear = False
        if autoRun:
            for i, j in autoRun['dfs']:
                self.model_prep(i, j)
            self.model_prep('users', ['adopted'])

            if 'var_scale' in autoRun:
                for i in autoRun['var_scale']:
                    self.scale(i)

            self.design()
            self.feature_importance()

    def model_prep(self, cdf, columns):
        """
        Merge DataFrames into df attribute to create design matrix

        :param str cdf: key of cleaned.dfs to access DataFrame
        :param int/str columns: index/column name of DataFrame to grab
        """
        self.df = pd.concat([self.df, self.cleaned.dfs[cdf][columns]], axis=1)

    def scale(self, i):
        """
        Scale column of df by standard deviation

        :param str i: column header
        """
        # move too standard scalar 
        self.df[i] /= self.df[i].std()

    def design(self, y=None):
        """
        Split target from features. Store in attributes X, y

        :param str y: column name of the y feature to split\
                default is the last column
        """
        if not y:
            self.X = self.df.values[:, :-1]
            self.header = self.df.columns[:-1]
            self.y = self.df.values[:,-1]
        else:
            self.X = self.df[[i for i in self.df.columns if i != y]].values
            self.header = self.df.columns[[i for i in self.df.columns
                                           if i != y]]
            self.y = self.df[y].values
        self.X = self.X.astype('float')
        self.y = self.y.astype('int')
        self._collinear_vif()

    def _collinear_vif(self):
        """
        Check for collinear features
        """
        for ind in range(self.X.shape[1]):
            value = vif(self.X, ind)
            if value > self.vifMagnitude:
                print self.header[ind] + ' has vif ' + str(value)
                self.collinear = True
        if self.collinear:
            raise Exception('Collinear feature risk')

    def feature_importance(self):
        """
        Run analysis with KFolds. (With room to add other aspects)
        """
        self.kfold_features()


    def kfold_features(self):
        """
        KFold model prediction with feature importance and model
        scores stored per fold.
        """
        self.scores = []
        self.featureValues = []
        kfolds = KFold(self.X.shape[0], n_folds=self.folds)
        for train_index, test_index in kfolds:
            X_train, X_test = self.X[train_index, :], self.X[test_index, :]
            y_train, y_test = self.y[train_index], self.y[test_index]
            fit = self.model.fit(X_train, y_train)
            if self.isTree:
                self.featureValues.append(fit.tree_.compute_feature_importances())
            elif self.isForest:

                self.featureValues.append(fit.feature_importances_)
            elif self.isLogistic:
                self.featureValues.append(fit.coef_[0])
            self.scores.append(fit.score(X_test, y_test))


class FeatureAnalysis(Cleaned):
    """
    Calculate organization sizes, networks, and graph attributes
    """
    def org_size(self):
        """
        Calculate org_size and degree where applicable

        Note org_size is number of Users with that a common org_id, not actual size
        """
        self.groups = {}
        self.dfs['users']['degree'] = 0
        self.groups['org_id'] = self.dfs['users'].groupby('org_id')
        self.dfs['users']['org_size'] = self.groups['org_id'].transform('count')['adopted']

    def build_org_networks(self):
        """
        Build graphs for connected users where applicable

        Note degree and children will have high correlation
        """
        self.graphs = {}
        for group, df in self.groups['org_id']:
            graph_df = df[['invited_by_user_id', 'object_id']][df.invited_by_user_id.notnull()].astype(int)
            if graph_df.shape[0] > 1:
                graph = nx.from_pandas_dataframe(graph_df.astype(int),
                                     'invited_by_user_id', 'object_id')
                self.graphs[group] = graph
                degrees = np.array(graph.degree().items()).astype(int)
                self.dfs['users']['degree'].loc[degrees[:, 0]] = degrees[:, 1]
                for g in nx.connected_component_subgraphs(graph):
                    nodes = g.nodes()
                    lenNodes = len(nodes)
                    if lenNodes > 3:
                        self.dfs['users']['local_rank'].loc[nodes] = len(nodes)

        self.dfs['users']['children'] = self.dfs['users']['degree'] -\
                                        self.dfs['users']['invited_by_user_id'].notnull()


class UserAdoption(object):
    """
    Front end for studing User Adoption data
    """

    def __init__(self, fUsers, fEngage, data=None):

        # Note to readers: this is a front end for a specific
        # analysis that makes use
        # of the rest of this package's architecture. We recommend
        # that you begin with class Cleaned and/or proceed downward
        # before looking at the contents in this constructor as
        # they will likey be unintelligable without familiarity with
        # the rest of the module.

        models = {'LOGISTIC': LogisticRegression(),
                  'TREE': DecisionTreeClassifier(),
                  'FOREST': RandomForestClassifier()
                  }
        runs = {}
        runs[1] =\
               {'dfs': [('users', ['opted_in_to_mailing_list',
                                   'enabled_for_marketing_drip'
                                   ]),
                        ('sources', ['GUEST_INVITE', 'ORG_INVITE',
                                     'PERSONAL_PROJECTS', 'SIGNUP'])]
                }

        runs[2] =\
               {'dfs': [('users', ['opted_in_to_mailing_list',
                                   'enabled_for_marketing_drip',
                                   'org_size', 'local_rank',
                                   'children'
                                   ]),
                        ('sources', ['GUEST_INVITE', 'ORG_INVITE',
                                     'PERSONAL_PROJECTS', 'SIGNUP'])],
                'var_scale': ['org_size', 'children']
                }

        self.results = {}
        self.importances = {}
        self.std = {}
        self.scores = {}

        if data:
            self.data = data
        else:
            self.data = FeatureAnalysis(fUsers, fEngage)
            self.data.org_size()
            self.data.build_org_networks()

        for i, run in runs.items():
            self.results[i] = {}
            self.importances[i] = {}
            self.std[i] = {}
            self.scores[i] = {}
            for m, model in models.items():
                isLogistic, isTree, isForest = [False, False, False]
                if m is 'LOGISTIC':
                    isLogistic = True
                elif m is 'TREE':
                    isTree = True
                elif m is 'FOREST':
                    isForest = True

                result = AdoptionModel(self.data, model,
                                       isLogistic=isLogistic,
                                       isTree=isTree,
                                       isForest=isForest,
                                       autoRun=run
                                       )

                self.results[i][m] = result
                self.importances[i][m] = np.mean(result.featureValues, axis=0)
                self.std[i][m] = np.std(result.featureValues, axis=0)
                self.scores[i][m] = np.mean(result.scores, axis=0)


def evaluate_logins(logins, adoption):
    """
    Evalue login  Dates to meet adoption criteria

    :param array-like logins: array of DateTime objects
    :param dict adoption: adoption criteria {'hits': int \
            'window': int} specifying the number of logins\
            or 'hits' required in a given 'window' to\
            constitute an adopted user

    :return: boolean
    :rtype: bool
    """
    hits = adoption['hits']
    window = adoption['window']
    if len(logins) < hits:
        return False
    else:
        logins.sort()

    hits -= 1
    for l1, l2, in it.izip(logins[:-hits], logins[hits:]):
        if l2 - l1 <= window:
            return True

    return False
