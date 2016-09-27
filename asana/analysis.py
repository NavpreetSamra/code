import pandas as pd
import numpy as np
from datetime import timedelta
import itertools as it
from sklearn.cross_validation import KFold
import networkx as nx
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


class UserAdoption(object):
    """
    **INTRODUCTION**
        The purpose of this analysis is to understand what features of a user\
                contribute to user adoption.
        Two critical points to remember throughout this process are:

        1. This data is artificial
        2. We do not have infinite time

        As a result of point 2, we prioritize\
            deriving meaningful and interpretable results that can manifest\
            into productive business driven insights. The following details\
            an inspection and exploration process of the data available in an\
            effort to determine what factors have the greatest impact on\
            predicting user adoption. From there we will begin to embed the\
            **maven** user as as feature and improve our ability to predict\
            as well as understand drivers of user adoption. (Maven\
            as described by Malcom Gladwell in The Tipping Point <link>)

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
            data to our *users table*. To do so, we group the login data\
            by *user_id*, sort by timestamp represented as :py:class:`pandas.DatetimeIndex` and\
            pass a window over the array to check for length of time between\
            a login and the login after the following login, (i, i+2). If that\
            time is less then or equal to 7 days we break out of our iteration\
            and record the user as adopted. An important note to recall\
            throughout this analysis is that because we have utilized login time,\
            and frequency to create the target for our prediction\
            we must be very careful when including logins/\
            login time in any prediction of user adoption, otherwise we\
            will suffer from leakage. We also note that at casual inspection\
            there are no drastic outliers in the *users table*, however there is\
            potentially an issue with the *engagement table* data that we will address\
            in the next section. Further analysis of *users table*\
            with studentized-t residuals can be used to investigate, however\
            beacuse our analysis will mainly pertain to\
            categorical data, outlier analysis will likely not play a \
            signifigant role.

    **CLEANING**
        We remove *name* and *email* from our table. While it's possible\
            certain types of email addresses can be indicative of\
            browser types and trends do exist within of names\
            (see Freakanomics) that analysis is deprioritized here\
            and those fields are removed from our feature space.

        Since *creation_source* is nominal with 5 unique values we dummify\
            the column and store it. It is important to note that the\
            stored data has not dropped a column, so a category\
            should be dropped before modeling to combat collinearity.

        One path we can follow here is to investigate adoption rate within\
            organizations. This is logical, however given this approach, we\
            would look to preform this after embedding the **maven**\
            (assuming we can find them), and to produce the most useful\
            information from the analysis, we need more information about\
            the organizations themselve (size, industry, growth, etc).\
            This would be one of the next few steps \
            to follow our approach begun in the `MODELING` section\
            below, but going back to point 2 in `INTRODUCTION`, in\
            the interest of prioritizing our time we have not yet\
            performed that analysis.


        We choose to remove time from our predictions for two reasons,

            1. Time (ours, not the datas). Because we use time\
                    in our calculation of adoption rate, we must be \
                    extremely careful with building features that\
                    incorporate the same data. Also, given the\
                    suggested time to spend on this analysis we\
                    choose to allocate our time on what we hope\
                    will be more accesible and lower hanging but\
                    also interesting fruit.
            2. To properly understand the impact of time\
                    on our system we need more information\
                    about the product, feature releases, market\
                    ecosystem, company history, and outreach.

        That all being said, we can still look at time and gain useful\
                information. Plotting logins by day we see an\
                increasing rate of logins per day (a good sign\
                for product health!) until 2014-05-21.

        The data after that point demonstrates a system level impact that\
                indicates a catastrophic failing. A likley scenerio here\
                is that the logging of logins began to fail, whether\
                because of a snapshot push or other issues, it begins to spread\
                and then the bottom drops out on 2014-06-05. While world,\
                or product/company catastrophy is possible, given our\
                general knowledge they are likley not the cause

    .. image:: ../images/LoginsPerDay.png
       :scale: 50 % 
       :align: center

    **INITIAL ANALYSIS**
        With a clean and numeric representation of the data we are now able\
            to investigate the impact of features on adoption.\
            To do so we will look at extracting feature importance\
            from three modeling techniques: Logistic Regression, and\
            Decision Trees, and Random Forest. The magnitude of coefficients\
            of (scaled) features in Logistic Regression represent that\
            feature's relative contribution to final classification, by\
            it's contribution to the slope definig the hyperplane
            associated with that feature\
            (with the caveat of holding all other features constant).\
            Logistic Regression direct readability of the impact variations\
            in a feature have make it a useful tool for it's interpretability\
            as well as it's tolerance to overfitting when the number of features\
            fed into it are large compared ot the volume of data (because it\
            is searching for a single hyperplane to partition the data)\
            Decision Trees, while highly prone to overfitting for\
            predicative modeling are useful for their intrepretability\
            of feature importance by calculating the gain of information\
            of a split on a feature at a node. Random Forests can improve on\
            Decision Trees predicative abilities, by reducing overfitting,\
            as well as ranking feature importance. For those not familiar,\
            Random Forests are aggregate of multiple decision trees each of\
            which is operating on a random subset of the data.

        With this analysis we find:
        <table>


    **MODELING**
        An initial exploration into feature importance gives insight into\
            what is avaiable at the surface of our data. However, to\
            better understand Asana users and what drives adoption\
            we look to understand the impact and types of users and\
            organizations. While there is a wide breadth of directions\
            we can proceed from here, we will focus on one organizational\
            attribute -size- and one aspect of users - sign up connectivity-\
            which is a first step in identifying mavens.

        Calculating organization size is straighforward. We group users by\
            *org_id* and count them. From there we begin to build a framework to analyze mavens\
            with the available data. We are only able to investigate users\
            who have or exist in the  *invited_by_user_id* field

        We hypothesize that inertia of tool usage exists within organizations\
            and look to understand that spread through a usage and knowledge levels\
            When a user is succesful with a tool, the tool is more\
            likely to spread because the user will recommend it to other\
            members of the organization-0th level.\
            A team wide decision to use a common\
            tool for a specific function- 1st level\
            The extension of this upward to the 2nd level is **mavens**.\
            **Mavens** are users\
            who work to become power users and then act as advocates/\
            evangelists for the software which is likely to increase\
            the signup rate and adoption rate of those users.\
            The **maven** will share the power of the tool, yielding a\
            shallower learning curve for new users and an increase\
            in adoption due to learning curve + maven usage. Combining with\
            tool/software standardization across teams and throughout\
            verticals, **mavens** provide us a target profile to\
            model with a goal of understanding tool flow throughout\
            organizations. This knowledge is critical for a succesfull\
            product and company evolution.

        We will begin building the framework to model **mavens** by\
            investigating connectivity.\
            In order to integrate connectivity as a feature for modeling\
            we build an undirected graph of each organization. Because\
            a node can only have 1 parent, and we have that recorded\
            in our features we can use an undirected graph to make the\
            computation of calculated connected subcomponents faster.\
            We also threshold a connected component having at least a\
            local rank of 3 for emphasis. This can be further investigated\
            in future work.\
            We maintain the ability to easily switch to a directed graph\
            moving forward if that structure suites our needs better.\
            Now we are able to add connected component rank of\
            a user as well as how many children\
            that user has.


        With these new features we look at the impact on feature importance\
            and predictive ability in our two models:

                <table>

        Again, is just the initial framework. From here\
            we could incorporate usage frequecy at time of recommendation,\
            success rate of invited memebers, and the velocity of the rank\
            of the graph as potentially interesting and meaningful features\
            in our model.

    **CONCLUSION**
        Ultimately succesfully modeling users\
            and organizations requires more information and time then\
            we currently have.\
            With more information about organizations and specific users, user\
            and organizational profiles can help drive product development\
            and pricing (pricing, because we now have a way to evaluate\
            traction within organizations). Moving forward we can look to\
            incorporate more detailed information about types of organization\
            as well as user application and use history.

        While these next steps all appear to be contain great power,\
            they are meaningless if the computation to perform the analysis is\
            intractable. The pulling of data can be moved to a database style\
            context which is optimized for grouping and reorganizing data\
            around an index. The structure of the code below lends itself\
            well to partitioning for parallel processing with (key, value) pairs.\
            An advantage of user and organizationl profiling is the creation of\
            intuitive ways to partition the data. Once profiles are complete\
            we can track the change in profiles by comparing subsets windowed\
            by time  which agian provides\
            natural partitions to our processing. If our data become so verbose\
            that Spark and AWS computation become intractable, we can sample\
            from our data for modeling.
    """

class Cleaned(object):
    """
    Class for pulling, cleaning, and aggregating data

    :param str/pd.DataFrame fUsers: Path or DataFrame of user data
    :param str/pd.DataFrame fEngage: Path or DataFrame of user engagement data
    :param tuple.(int,int) adoption: (number of logins, within window)
    :param array-like drops: names of columns to drop
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
        # self._evaluate_users()
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


class AdoptionModel(object):
    """
    """
    def __init__(self, cleaned, model, importances=None, isTree=False, isForest=False, folds=10,
                 vifMagnitude=3.0,
                 autoRun={'dfs': [('users', ['opted_in_to_mailing_list',
                                             'enabled_for_marketing_drip',
                                             'org_size', 'local_rank',
                                             'children']),
                                  ('sources', ['GUEST_INVITE', 'ORG_INVITE',
                                               'PERSONAL_PROJECTS', 'SIGNUP'])],
                          'var_scale': ['org_size', 'children']
                          }
                 ):
        self.cleaned = cleaned
        self.model = model
        self.importances = importances
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
        """
        self.kfold_features()



    def kfold_features(self):
        """
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
            else:
                self.featureValues.append(fit.__dict__[self.importances][0])
            self.scores.append(fit.score(X_test, y_test))



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
