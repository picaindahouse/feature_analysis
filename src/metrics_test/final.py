from tqdm.auto import tqdm

import performance
import trend
import metric_over_time
import stability


def overall(df, label, date=[], bad_class=1, on_bin=False, strat=None, bins=5):
    """
    Returns a function that stores all the evaluations
    of the df available in this package.

    Function returned stores 7 DataFrames:
    df(1) : Evaluation
    df(2) : Trend
    df(3) : Total
    df(4) : Total Ratio
    df(5) : Bad
    df(6) : Bad Rate
    df(7) : Stability

    Parameters
    ----------
    df : Dataframe
         The dataset whose features we are evaluating

    label : String
            Column containing binary output

    date : String, optional
           Column containing datetime object

    bad_class : int, optional
                Which int is the "bad class"

    on_bin : Boolean, optional
             Whether to find AUC/KS score on
             binned or continuous values of the
             continuous features in df

    strat : String, optional
            A strategy that can be used by SimpleImputer
            from sklearn.impute

    bins : int, optional
          Maximum number of bins each feature should have

    Returns
    -------
    df : Function
         A function that stores all 7 dataframes
    """
    n = 8
    if date == []:
        n = 3

    for x in tqdm(range(1, n)):
        if x == 1:
            df1 = performance.evaluation(df,
                                         label,
                                         strat=strat,
                                         on_bin=on_bin,
                                         to_drop=[date])
        elif x == 2:
            df2 = trend.trend(df,
                              label,
                              to_drop=[date],
                              bins=bins,
                              bad_class=bad_class)
        elif x == 3:
            df3 = metric_over_time.total(df, label, date)
        elif x == 4:
            df4 = metric_over_time.ratio(df, label, date)
        elif x == 5:
            df5 = metric_over_time.bad(df, label, date, bad_class=bad_class)
        elif x == 6:
            df6 = metric_over_time.bad_rate(df,
                                            label,
                                            date,
                                            bad_class=bad_class)
        elif x == 7:
            df7 = stability.stability(df, label, date)

    def df(i):
        if date == []:
            dfs = {1: df1, 2: df2}

        else:
            dfs = {1: df1, 2: df2, 3: df3, 4: df4, 5: df5, 6: df6, 7: df7}
        return dfs[i]

    return df