import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="paper", font="monospace")


def df_table(df, fname):
    """
    Helper function to create rst tables from :py:class:`pandas.DataFrame`

    :param `pandas.DataFrame` df: data frame
    :param str fname: string file name
    """
    with open(fname, 'a') as w:
        w.write(tabulate.tabulate(df, tuple(df.columns), tablefmt='rst'))
        w.write('\n')


def corr_heatmap(df, fname, vmax=.8):

    """
    Helper function to create correlation heatmap imagesfrom :py:class:`pandas.DataFrame`

    :param `pandas.DataFrame` df: data frame
    :param str fname: string file name
    :param vmax float: absolute value bound for colorbar
    """
    sns.heatmap(df.corr(), vmax=vmax, square=True)
    plt.savefig(fname)
    plt.close()
