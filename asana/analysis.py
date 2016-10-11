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


class UserAdoption(object):
    """
    **INTRODUCTION**
        The purpose of this analysis is to understand what features of a user\
                contribute to user adoption.
        Before proceeding, two critical points to remember throughout\
                this process are:

        1. This data is artificial
        2. We do not have infinite time

        As a result of point 2, we prioritize\
            guiding our work toward interpretable results that can manifest\
            into productive business driven insights. The following details\
            an inspection and exploration process of the provided data in an\
            effort to determine what factors have the greatest impact on\
            predicting user adoption. From there we will begin to embed the\
            **maven** user as\
            a feature with the goal of improving prediction\
            as well as better understanding drivers of user adoption. The `Maven\
            <https://en.wikipedia.org/wiki/Maven>`_ as described by Malcom\
            Gladwell in\
            `The Tipping Point <https://en.wikipedia.org/wiki/The_Tipping_Point>`_


        Data inspection reveals attributes belonging to\
            2 broad categorizations:

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

        With the exception of *login history*, this data is organized in a single\
            table. We proceed by calculating whether or not a user has adopted\
            given the criteria (3 logins in a 7 day window) and joining that\
            data to our *users table*. To do so, we group the login data\
            by *user_id*, sort by timestamp represented as :py:class:`pandas.DatetimeIndex` and\
            pass a window over the array to check for length of time between\
            a login and the login after the following login, (i, i+2). If that\
            delta in time is less then or equal to 7 days we break out of our iteration\
            and record the user as *adopted*. An important note\
            throughout this analysis: we have utilized login time\
            and frequency to create the target for our prediction and consequently\
            we must be very careful when including number of logins\
            in any prediction of user adoption, otherwise we\
            will suffer from leakage. We also note that at casual inspection\
            there are no drastic outliers in the *users table*, however there is\
            potentially an issue with the *engagement table* data that we will address\
            in the next section. Further analysis of *users table*\
            with studentized-t residuals can be used to investigate outliers, however\
            beacuse our analysis will mainly pertain to\
            categorical data, outlier analysis will likely not play a \
            signifigant role.

    **CLEANING**
        We remove *name* and *email* from our table. While it's possible\
            certain types of email addresses can be indicative of\
            browser types and trends do exist within names\
            (see Freakanomics) that analysis is deprioritized here\
            and those fields are removed from our feature space.

        Since *creation_source* is nominal with 5 unique values we dummify\
            the column and store it. It is important to note that the\
            stored data has not dropped a column, so a category\
            should be dropped before modeling to combat collinearity.

        One path we can follow here is to investigate adoption rate within\
            organizations. This is logical, however given this approach, we\
            would look to perform this after embedding the **maven**\
            (assuming we can find them). To produce the most useful\
            information from the analysis, we need more information about\
            the organizations themselves (size, industry, growth, etc).\
            This would be one of the next few steps\
            to follow our approach begun in the `MODELING` section\
            below. Going back to point 2 in `INTRODUCTION`, in\
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
                indicates a catastrophic failure. A likley scenerio here\
                is that the logging of logins began to fail, whether\
                because of a snapshot push or other issues, it begins to spread\
                and then the bottom drops out on 2014-06-05. While world,\
                or product/company catastrophy is possible, given our\
                general knowledge they are likley not the cause.

    .. image:: ../images/LoginsPerDay.png
       :scale: 75 %
       :align: center

    .. image:: ../images/users_data.png
       :scale: 75 %
       :align: center



    The number of accounts created by day likely confirms\
        this theory, and \
        suggests that the increase in rate of account\
        creation at the end of this timeframe exceeded the load the system\
        was designed to handle leading to the failure in capturing logins,\
        without ruling out some of the possibilities enumerated above.

    **INITIAL ANALYSIS**
        With a clean and numeric representation of the data we are now able\
            to investigate the impact of features on adoption.\
            To do so we will look at extracting feature importance\
            from three modeling techniques: Logistic Regression, \
            Decision Trees, and Random Forests. The magnitude of coefficients\
            of (scaled) features in Logistic Regression represent that\
            feature's relative contribution to final classification, by\
            it's contribution to the slope defining the hyperplane\
            associated with that feature's component\
            (with the caveat of holding all other features constant).\
            Logistic Regression's direct readability of the impact variations\
            in a feature make it a useful tool for it's interpretability\
            as well as it's tolerance to overfitting when the number of features\
            fed into it are large compared to the volume of data (because it\
            is searching for a single hyperplane to partition the data)\
            Decision Trees, while highly prone to overfitting for\
            predicative modeling are useful for their interpretability\
            of feature importance by calculating the gain of information\
            of a split on a feature at a node. Random Forests can improve on\
            Decision Trees predicative abilities, by reducing overfitting,\
            as well as contribute to ranking feature importance albeit\
            with a different approach to determination.\
            For those not familiar,\
            Random Forests are aggregate of multiple decision trees each of\
            which is operating on a random subset of the data.

        With this analysis we find feature coefficients for Logistic Regression\
            and normalized gini impurity for tree/ensemble methods. The results in the\
            following tables represent values from averaging 10 fold cross\
            validation where standard deviations of all values are 1-2 orders\
            of magniutde less then the values presented (excluding 0s). The following models\
            were run with default parameters from\
            `sklearn <https://http://scikit-learn.org/stable/documentation.html>`_\
            as this is an initial investigation.\
            Because the number of users to number of features ratio is large\
            and with no clear outlier data yet\
            we do not expect to need regularization for our Logistic\
            Regression. We also take the opportunity here to address\
            why we decided not to dummify *org_id*. As discussed (and\
            will continue to be discussed) a lack of information about\
            the orgs is a contributing factor, but also would render\
            a Decision Tree intractable as well as a sizeable increase in\
            computation time for a Random Forest (we recognize that\
            RFs are parallizable but given the context of this analysis).


+--------+--------------+----------------+--------------+------------+----------+--------+
|Model   | mailing_list | marketing_drip | GUEST_INVITE | ORG_INVITE | PERSONAL | SIGNUP |
+========+==============+================+==============+============+==========+========+
|Tree    | 0.023        |    0.038       |    0.025     |   0.144    |     0.69 |  0.08  |
+--------+--------------+----------------+--------------+------------+----------+--------+
|Forest  | 0.068        |    0.074       |    0.132     |    0.1     |    0.562 | 0.065  |
+--------+--------------+----------------+--------------+------------+----------+--------+
|Logistic| 0.047        |    0.021       |    -0.015    |   -0.291   |    -0.85 | -0.21  |
+--------+--------------+----------------+--------------+------------+----------+--------+

        The take away from this analysis is that the methods of invitation\
            are more important then being on the mailing list or a marketing drip.\
            Also please note, that values of our tree based methods represent\
            magnitude and not direction (another perk of Logistic Regression).\
            Since all our features are categorical, coefficients of the Logistic\
            Regression are easy to compare.


    **MODELING**
        An initial exploration into feature importance gives insight into\
            what is available at the surface of our data. However, to\
            better understand Asana users and what drives adoption\
            we look to understand the impact and types of users and\
            organizations. While there is a wide breadth of directions\
            we can proceed from here, we will focus on one organizational\
            attribute -size- to help inform  one aspect of users\
            - sign up connectivity- which is a first step in\
            identifying **mavens**.

        Calculating organization size, in the context of this analysis is\
            the number of Asana users in the organzation,\
            is straighforward. We group users by\
            *org_id* and count them. From there we begin to build a framework\
            to analyze **mavens**.\
            With the available data we are only able to investigate users\
            who have or exist in the *invited_by_user_id* field

        We hypothesize that inertia of tool usage exists within organizations\
            and look to understand that spread through usage and knowledge levels\

            * 0th level - When a user is succesful with a tool, the tool is more\
            likely to spread because the user will recommend it to other\
            members of the organization.
            * 1st level - A team wide decision to use a common\
            tool for a specific function
            * 2nd level - The next extension is **mavens**.\
            **Mavens** are users\
            who work to become power users and then act as advocates/\
            evangelists for the software which is likely to increase\
            the signup rate and adoption rate of those users.\
            The **maven** will share the power of the tool, yielding a\
            shallower learning curve for new users and an increase\
            in adoption due to learning curve + **maven** usage. Combining with\
            tool/software standardization across teams and throughout\
            verticals, **mavens** provide us a target profile to\
            model with a goal of understanding tool flow throughout\
            organizations.

        We will begin building the framework to model **mavens** by\
            investigating connectivity.\
            In order to integrate connectivity as a feature for modeling\
            we build an undirected graph of each organization. Because\
            a node can only have 1 parent, and we have that recorded\
            in our features we can use an undirected graph to make the\
            computation of calculating connected subcomponents faster.\
            We also threshold a connected component having a local rank\
            greater then 3 for emphasis. This can be further investigated\
            in future work.\
            We maintain the ability to easily switch to a directed graph\
            moving forward if that structure suits our needs better.\
            Now we are able to add the local connected component rank of\
            a user as well as how many children\
            that user has to our design matrix.


        With these new features we look at the impact on feature importance\
            and predictive ability in our three models (Note org_size is\
            number of Asana accounts belonging to that user's *org_id*\
            and continuous variables have been normalized to have a\
            standard deviation of 1 in the design matrix\
            which implies dx for a continuous variable is interpreted\
            differently from a categorical variable in Logistic Regression\
            coefficients:

+--------+--------------+----------------+----------+------------+----------+-----------+---------+----------+--------+
|Model   | mailing_list | marketing_drip | org_size | local_rank | children | GUEST_INV | ORG_INV | PERSONAL | SIGNUP |
+========+==============+================+==========+============+==========+===========+=========+==========+========+
|Tree    |    0.084     |     0.054      |   0.4    |   0.234    |  0.133   |    0.03   |  0.024  |  0.013   | 0.027  |
+--------+--------------+----------------+----------+------------+----------+-----------+---------+----------+--------+
|Forest  |    0.036     |     0.029      |   0.55   |    0.23    |  0.104   |   0.012   |  0.013  |  0.013   | 0.013  |
+--------+--------------+----------------+----------+------------+----------+-----------+---------+----------+--------+
|Logistic|    0.041     |      0.01      |  -0.339  |    0.0     |  0.111   |   -0.035  |  -0.302 |  -0.867  | -0.229 |
+--------+--------------+----------------+----------+------------+----------+-----------+---------+----------+--------+

        While the tree based methods find our engineered features\
            rich in information. The Logistic Regression throws away *local_rank*.\
            Given how we've constructed these features in our design matrix\
            there is some redundancy\
            of information. Creating different models based on grouping sign up\
            methods could be a good direction to take our work and garner more\
            insight into features contributing to adoption. We note that our\
            pipeline includes a check of **Variance Inflation Factor** in\
            constructing our design matrix, and will raise an exception\
            if the magnitude of any feature exceeds a VIF of 3.

        Closer inspection reveals an overall downard trend of *org_size*\
            as we would expect from our Logit coefficient. One interesting\
            phenomenon that we did not expect and likely hinders the success\
            of *children* and *local_rank* features is the 0. adoption rate\
            among users with 11 or more children. This is contrary to our\
            **maven** hypothesis. One potential explanation for these users\
            is they send the reference link and are part of onboarding process but choose\
            another tool within their own team. As we have adressed multiple\
            times we do not have time within this analysis to include login\
            frequency in a constructive manner, however this attribute would likely\
            help us partition **mavens** from **connectors** and improve our model.

    .. image:: ../images/size_adopted.png
       :scale: 65 %
    .. image:: ../images/children_adoption.png
       :scale: 65 %
    .. image:: ../images/rank_adoption.png
       :scale: 65 %


    We continue to find *mailing_list* and *marketing_drip* to not\
        effectively contribute. At this point if we were to continue\
        to build our model we would try dropping these features and\
        measure the impact on accuracy as well as recall and precision.\

        We will take this opportunity to discuss class imbalance in our\
            system. With only a 13.8% adoption rate, when moving forward\
            with modeling efforts we will need to account for this. In a\
            parallel thread we also bring up that accuracy is likely not the best\
            scoring metric, but to give a more useful scoring metric\
            and direction to take the model we need to better understand\
            what the results are going to drive.

        These factors all combine to yield no signifigant change in the\
                accuracy of our model calculated during cross validation.

        Again, this is just the initial framework. From here\
            we could incorporate usage frequency at time of recommendation,\
            success rate of invited members, and the velocity of the rank\
            of the graph as potentially interesting and meaningful features\
            in our model.

    **CONCLUSION**
        Ultimately succesfully modeling users\
            and organizations requires more information and time then\
            we currently have.\
            With more information about organizations and specific users, user\
            and organizational profiles can help drive product development\
            and pricing (pricing, because we then have a way to evaluate\
            traction within organizations). Moving forward we can look to\
            incorporate more detailed information about types of organizations\
            as well as user application and use history/profiles.

        While these next steps all appear to contain great power,\
            they are meaningless if the computation to perform the analysis is\
            intractable. The pulling of data can be moved to a database style\
            context which is optimized for grouping and reorganizing data\
            around an index. The structure of the code below lends itself\
            well to partitioning for parallel processing with (key, value) pairs.\
            An advantage of user and organizationl profiling is the creation of\
            intuitive ways to partition the data. Once profiles are complete\
            we can track the change in profiles by comparing subsets windowed\
            by time  which again provides\
            natural partitions to our processing. If our data become so verbose\
            that Spark and AWS computation become intractable, we can sample\
            from our data for modeling.
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
