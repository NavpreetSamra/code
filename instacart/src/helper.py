import tabulate

def df_table(df, fname):
    """
    Helper function to create rst tables from :py:class:`pandas.DataFrame`

    :param `pandas.DataFrame` df: data frame
    :param str fname: string file name
    """
    with open(fname,'a') as w:
        w.write(tabulate.tabulate(df, df.columns, tablefmt='rst'))
        w.write('\n')
