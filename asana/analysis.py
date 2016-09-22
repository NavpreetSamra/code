import pandas as pd
import numpy as np
from datetime import timedelta
import itertools as it


class Cleaner(object):
    """
    """
    def __init__(self, fUsers, fEngage, adoption=(3, 7),
                 drops=['name', 'email']):

        self.adoption = {'hits': adoption[0],
                         'window': timedelta(days=adoption[1])}
        self.dfs = {}  # DataFrames, not dist file systems

        self._pull_data(fUsers, fEngage, drops)
        self._split_non_users()
        self._convert_user_time()
        self._evaluate_users()
        self._dummify_source()

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
        self.dfs['users'].set_index('object_id', inplace=True)
        self.dfs['users']['adopted'] = np.zeros(self.dfs['users'].shape[0],
                                                ).astype(bool)

    def _split_non_users(self):
        """
        Split accounts that have never logged in to separate df
        """
        mask = self.dfs['users']['last_ses'].isnull()
        self.dfs['non_users'] = self.dfs['users'].iloc[mask]
        self.dfs['users'] = self.dfs['users'].iloc[np.invert(mask)]

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

        self.dfs['users']['creation_time'] =\
            pd.to_datetime(self.dfs['users']['creation_time']).dt.date

        self.dfs['engage']['time_stamp'] =\
            pd.to_datetime(self.dfs['engage']['time_stamp']).dt.date

    def _evaluate_users(self):
        """
        """
        user_history = self.dfs['engage'].groupby('user_id')

        for user, df in user_history:
            if evaluate_logins(sorted(df.time_stamp.values), self.adoption):
                self.dfs['users']['adopted'].loc[user] = 1

    def _dummify_source(self):
        """
        Create dummified creation_source DataFrame\
                Store in as sources in dfs attribute
        """
        self.dfs['sources'] = pd.get_dummies(self.dfs['users']\
                                                     ['creation_source'])


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
    if len(logins) < 3:
        return False
    else:
        logins.sort()
        hits = adoption['hits'] - 1
        window = adoption['window']

    for l1, l2, in it.izip(logins[:-hits], logins[hits:]):
        if l2 - l1 <= window:
            return True

    return False
