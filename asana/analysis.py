import pandas as pd
import numpy as np
from datetime import timedelta
import itertools as it
from sklearn.cross_validation import KFold
import networkx as nx
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


class UserAdoption(object):
    """
    The purpose of this analysis is to understand what features of a user\
            contribute to user adoption. Two critical points to remember\
            throughout this process are 1) this data is artificial\
            2) we do not have infinite time and as a result prioritize\
            deriving meaningful and interpretable results that can manifest\
            into productive business driven insights. The following details\
            an inspection and exploration process of the data available in an\
            effort to determine what factors have the greatest impact on\
            predicting user adoption. From there we will attempt to embed a\
            type of user as as feature and improve our ability to predict user\
            adoption. which we hope has a direct positive impact on our\
            the *maven* as detailed\
            by Malcom Gladwell in The Tipping Point.


            Our inspection of the data begins with 2 broad categorizations:

            1) User
                * name
                * object_id
                * email
                * org_id
                * last_session_create_time
                * opted_in_to_mailing_list
                * enabled_for_marketing_drip
                * login history (timestamp)

            2) Path to Asana
                * creation_source (5 categories)
                * creation_time
                * invited_by_user_id

    Because all of this data except login history is organized in a single\
            table, we proceed by calculating whether or not a user has adopted\
            given the criteria (3 logins in a 7 day window) and joining that\
            data to our table of user data. To do so, we group the login data\
            by user id, sort by timestamp represented as pd.DateTime and\
            pass a window over the array to check for length of time between\
            a login and the login after the following login, (i, i+2). If that\
            length is less or equal to 7 days we break out of our iteration and\
            record the user as adopted. An important note to recall throughout\
            this analysis is that because we have utilized login time,\
            and frequency we must be very careful when including logins/\
            login time in any prediction of user adoption, otherwise we\
            will suffer from leakage. <link>

        Next we remove name and email from our table. While it's possible\
                certain types of email addresses can be indicative of\
                browser types and trends do exist within of names\
                (see Freakanomics) that analysis is deprioritized here\
                and those fields are removed from our feature space.

        Since creation_source is nominal with 5 unique values we dummify\
                the column and store it. It is important to note that the\
                stored data has not dropped a column, so a category\
                should be dropped before modeling to combat colinearity.


        <Feature correlation and VIF>


        With a clean and numeric representation of the data we are now able\
                to investigate the impact of features on adoption.\
                To do so we will look at extracting feature importance\
                from three modeling techniques: Logistic Regression,\
                Decision Trees and Random Forests. The magnitude of coefficients\
                of (scaled) features in Logistic Regression represent that\
                feature's relative contribution to final classification.\
                Decision Trees, while highly prone to overfitting for\
                predicative modeling are useful for their intrepretability\
                of feature importance by calculating the gain of information\
                of a split on a feature at a node. Lastly RF <> ...

        With this analysis we find:
            <table>


        An initial exploration into feature importance gives insight into\
                what is avaiable at the surface of our data. However, to\
                better understand Asana users and what drives adoption\
                we look to understand the impact and types of users and\
                organizations. While there is a wide breadth of directions\
                we can proceed from here, we will focus on one organizational\
                attribute -size- and one aspect of users - sign up connectivity-\
                which is a first step in identifying mavens.

        Calculating organization size is straighforward. We group users by\
                org_id and count them. From there we begin to build a framework to analyze mavens\
                with the available data. We are only able to investigate users\
                who have an *invited_by_user_id*.

        We hypothesize that inertia of tool usage exists within organizations\
                - that when a user is succesful with a tool, the tool is more\
                likely to spread because the user will recommend it to other\
                members of the organization. Furthermore, *mavens* are users\
                who work to become power users and then act as advocates/\
                evangelists for the software which is likely to increase\
                the signup rate and adoption rate of those users.\
                The maven will share the power of the tool, yielding a\
                shallower learning curve for new users and an increase\
                in adoption both due to learning curve and maven usage.\
                In order to integrate connectivity as a feature for modeling\
                we build a directed graph of each organization, where edges\
                represent the link from the user who sent the invite to the
                user who received it. This enables us to add what size graph
                a user is a part of as well as how many direct children\
                that user has. This is just the initial framework. From here\
                we can incorporate usage frequecy at time of recommendation,\
                success rate of invited memebers, and the velocity of the rank\
                of the graph as potentially interesting and meaningful features\
                to incorporate in our model. Ultimately succesfully modeling users\
                and organizations requires more information then we currently have.\
                With more information about organizations and specific users, user\
                and organizational profiles can help drive product development\
                and pricing (pricing, because we now have a way to evaluate\
                traction within the organization). 

    """

class Cleaned(object):
    """
    """
    def __init__(self, fUsers, fEngage, adoption=(3, 7),
                 drops=['name', 'email']):

        self.adoption = {'hits': adoption[0],
                         'window': timedelta(days=adoption[1])}
        self.dfs = {}  # DataFrames, not dist file systems

        self._pull_data(fUsers, fEngage, drops)
        self._dummify_source()
        self._split_non_users()
        self._convert_user_time()
        self._evaluate_users()
        self._merge_non_users()

    def _pull_data(self, fUsers, fEngage, drops):
        """
        """
        if isinstance(fUsers, str):
            self.dfs['users'] = pd.read_csv(fUsers)
        elif isinstance(fUsers, pd.DataFrame):
            self.dfs['users'] = fUsers
        else:
            raise Exception('fUsers not path or DataFrame')

        if isinstance(fEngage, str):
            self.dfs['engage'] = pd.read_csv(fEngage)
        elif isinstance(fEngage, pd.DataFrame):
            self.dfs['engage'] = fEngage
        else:
            raise Exception('fEngage not path or DataFrame')

        if drops:
            self.dfs['users'].drop(drops, inplace=True, axis=1)

        columnsRename = [i if i != 'last_session_creation_time' else
                         'last_ses' for i in self.dfs['users'].columns]

        self.dfs['users'].columns = columnsRename
        self.dfs['users'].set_index('object_id', drop=False, inplace=True)
        self.dfs['users']['invited_by_user'] =\
            self.dfs['users'].invited_by_user_id.notnull()

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
        """
        user_history = self.dfs['engage'].groupby('user_id')

        for user, df in user_history:
            if evaluate_logins(sorted(df.time_stamp.values), self.adoption):
                self.dfs['users']['adopted'].loc[user] = 1

    def _merge_non_users(self):
        """
        """
        self.dfs['users'] = pd.concat([self.dfs['users'],
                                       self.dfs['non_users']])
        del self.dfs['non_users']


class FeatureAnalysis(Cleaned):
    """
    """

    def org_size(self):
        """
        """
        self.groups = {}
        self.dfs['users']['degree'] = 0
        self.groups['org_id'] = self.dfs['users'].groupby('org_id')
        self.dfs['users']['org_size'] = self.groups['org_id'].transform('count')['adopted']

    def build_org_networks(self):
        """
        """
        self.graphs = {}
        for group, df in self.groups['org_id']:
            graph_df = df[df.invited_by_user_id.notnull()]
            if graph_df.shape[0] > 1:
                graph = nx.from_pandas_dataframe(graph_df,
                                     'invited_by_user_id', 'object_id')
                self.graphs[group] = graph
                degrees = np.array(graph.degree().items()).astype(int)
                self.dfs['users']['degree'].loc[degrees[:, 0]] = degrees[:, 1]

        self.dfs['users']['children'] = self.dfs['users']['degree'] - self.dfs['users']['invited_by_user'].astype(int)


class AdoptionModel(object):
    """
    """
    def __init__(self, cleaned, model, folds=10):
        self.cleaned = cleaned
        self.model = model
        self.folds = folds
        self.df = pd.DataFrame(index=self.cleaned.dfs['users'].index)

    def model_prep(self, cdf, columns):
        """
        Merge DataFrames into df attribute to create design matrix

        :param str cdf: key of cleaned.dfs to access DataFrame
        :param int/str columns: index/column name of DataFrame to grab
        """
        self.df = pd.concat([self.df, self.cleaned.dfs[cdf][columns]], axis=1)

    def design(self, y=None):
        """
        Split target from features. Store in attributes X, y

        :param str y: column name of the y feature to split\
                default is the last column
        """
        if not y:
            self.X = self.df.values[:-1]
            self.header = self.df.columns[:-1]
            self.y = self.df.values[-1]
        else:
            self.X = self.df[[i for i in self.df.columns if i != y]].values
            self.header = self.df.columns[[i for i in self.df.columns
                                           if i != y]]
            self.y = self.df[y].values

    def colinear_vif(self, magnitude=3.):
        """
        Check for colinear features
        """
        for ind, col in enumerate(self.X):
            value = vif(self.X, ind)
            if value > magnitude:
                print self.columns[ind] + ' has vif ' + str(value)
                self.collinear = True

    def kfold_features(self):
        """
        """
        self.results = []
        self.feature = []
        self.scores = cross_val
        kfolds = KFold(self.df.shape[0], n_folds=self.folds)
        for train_index, test_index in kfolds:
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fit = self.model.fit(X_train, y_train)
            self.results.append(fit.score(X_test, y_test))

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
