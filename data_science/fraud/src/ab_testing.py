import pandas as pd
import numpy as np
import itertools


def _build_df():
    """
    Helper function to build funnel :py:class:`pandas.Dataframe`\
            for A B testing analysis

    :return: DataFrame representation of funnel
    :rtype: pandas.DataFrame
    """
    c = [5005, 3654, 2856, 2055]
    a = [5012, 3838, 3073, 2255]
    b = [5025, 3987, 2974, 2070]

    df = pd.DataFrame(data=[c, a, b], index=['c', 'a', 'b'],
                      columns=['visitors', 'createdAccount',
                               'addedEmployees', 'ranPayroll'])
    return df


def ab_testing(df=_build_df(), numSamples=5000):
    """
    Helper function to leverage beta distribution for A B testing

    Builds analysis array, values represent % likelihood x[0] > x[1],
    columns represent transition between stages in funnel (stage1_stage2)

    :param pandas.Dataframe df: Funnel, index are tests (a,b, c{control}),\
            columns are stages in funniel.
    :param int numSamples: number of samples to use in pulling\
            from beta distribution

    :return: DataFrame of funnel evaluation
    :rtype: pandas.DataFrame

    """

    combinations = list(itertools.combinations(df.columns, 2))
    index = list(itertools.combinations(sorted(df.index), 2))

    results = pd.DataFrame(index=index,
                           columns=["_".join(i) for i in combinations])
    for pair in index:
        for i, j in combinations:
            aSuccess = df[j][pair[0]]
            bSuccess = df[j][pair[1]]

            aFailure = df[i][pair[0]] - aSuccess
            bFailure = df[i][pair[1]] - bSuccess

            a = np.random.beta(1 + aSuccess, 1 + aFailure, numSamples)
            b = np.random.beta(1 + bSuccess, 1 + bFailure, numSamples)

            results["_".join([i, j])][pair] = np.sum(a > b) / float(numSamples)

    return results
