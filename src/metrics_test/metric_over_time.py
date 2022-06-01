"""
Functions to compare binned features over time.
"""

import pandas as pd

import feature_types
import performance


def total(df, label, date):
    """
    Find the total number of occurences of each bin for each
    feature in a DataFrame over the bins in date column.

    Parameters
    ----------
    df : Dataframe
         The dataset we are working with

    label : String
            Column containing binary output

    date : String
           Column containing datetime object

    Returns
    -------
    total_df : DataFrame
               A new DataFrame containing the total
               of each feature in df during each
               date bin.
    """
    new_df = feature_types.cont_to_cat(df.drop(date, axis=1), True)
    new_df = performance.fill_missing(new_df, None)
    new_df[date] = df[date]

    total_df = pd.DataFrame()
    cols = df.columns

    for feature in cols[~cols.isin([label, date])]:

        d0 = pd.get_dummies(new_df[[feature, date]], columns=[date])
        d = d0.groupby(feature, as_index=False).sum()
        d.columns = ["Bin"] + sorted(list(new_df[date].unique()))

        for x in d.columns[~d.columns.isin(["Bin"])]:
            d = d.astype({x: int})

        d.insert(loc=0, column="Variable", value=feature)

        if feature_types.data_type(df, feature) in ["NC", "DC"]:
            d = d.sort_values(by=["Bin"],
                              key=lambda x:
                                  ([float(a.split(",")[0][1:]) for a in x]))
            d.index = range(len(d))

        total_df = pd.concat([total_df, d])
    return total_df


def ratio(df, label, date):
    """
    Find the ratio of each bin for each feature
    in a DataFrame over the bins in date column.

    Parameters
    ----------
    df : Dataframe
         The dataset we are working with

    label : String
            Column containing binary output

    date : String
           Column containing datetime object

    Returns
    -------
    total_df : DataFrame
               A new DataFrame containing the ratio
               of each feature in df during each
               date bin.
    """
    new_df = feature_types.cont_to_cat(df.drop(date, axis=1), True)
    new_df = performance.fill_missing(new_df, None)
    new_df[date] = df[date]

    ratio_df = pd.DataFrame()
    cols = df.columns

    for feature in cols[~cols.isin([label, date])]:

        d0 = pd.get_dummies(new_df[[feature, date]], columns=[date])
        d = d0.groupby(feature, as_index=False).sum()
        d.columns = ["Bin"] + sorted(list(df[date].unique()))

        for x in d.columns[~d.columns.isin(["Bin"])]:
            d = d.astype({x: int})
            d[x] = round(d[x]/sum(d[x])*100, 2)

        d.insert(loc=0, column="Variable", value=feature)
        if feature_types.data_type(df, feature) in ["NC", "DC"]:
            d = d.sort_values(by=["Bin"],
                              key=lambda x:
                                  ([float(a.split(",")[0][1:]) for a in x]))
            d.index = range(len(d))

        ratio_df = pd.concat([ratio_df, d])
    return ratio_df


def bad(df, label, date, bad_class=1):
    """
    Find the number of "bad" in each bin for each feature
    in a DataFrame over the bins in date column.

    Parameters
    ----------
    df : Dataframe
         The dataset we are working with

    label : String
            Column containing binary output

    date : String
           Column containing datetime object

    bad_class : int, optional
                Which int is the "bad class"

    Returns
    -------
    total_df : DataFrame
               A new DataFrame containing the number
               of "bad" in each feature in df during each
               date bin.
    """
    new_df = feature_types.cont_to_cat(df.drop(date, axis=1), True)
    new_df = performance.fill_missing(new_df, None)
    new_df[date] = df[date]

    bad_df = pd.DataFrame()
    cols = df.columns

    for feature in cols[~cols.isin([label, date])]:

        d0_total = pd.get_dummies(new_df[[feature, date]], columns=[date])
        d_total = d0_total.groupby(feature, as_index=False).sum()
        d_total.columns = ["Bin"] + sorted(list(df[date].unique()))

        d0 = pd.get_dummies(new_df[[feature, date]].loc[new_df[label] == 1], columns=[date])
        d = d0.groupby([feature], observed=False).sum()
        d = d.reindex(d_total["Bin"].unique()).fillna(0).astype(int)

        d.columns = [x.split("_")[-1] for x in d.columns]

        for x in sorted(list(new_df[date].unique())):
            if str(x) in d.columns:
                continue
            d[str(x)] = 0
        d = d[[str(x) for x in sorted(list(new_df[date].unique()))]]
        d = d.reset_index(level=0)

        d.columns = ["Bin"] + sorted(list(new_df[date].unique()))

        for x in d.columns[~d.columns.isin(["Bin"])]:
            d = d.astype({x: int})

            if bad_class == 0:
                d[x] = d_total[x] - d[x]

        d.insert(loc=0, column='Variable', value=feature)

        if feature_types.data_type(df, feature) in ["NC", "DC"]:
            d = d.sort_values(by=["Bin"],
                              key=lambda x:
                                  ([float(a.split(",")[0][1:]) for a in x]))
            d.index = range(len(d))

        bad_df = pd.concat([bad_df, d])
    return bad_df


def bad_rate(df, label, date, bad_class=1):
    """
    Find the ratio of "bad" in each bin for each feature
    in a DataFrame over the bins in date column.

    Parameters
    ----------
    df : Dataframe
         The dataset we are working with

    label : String
            Column containing binary output

    date : String
           Column containing datetime object

    bad_class : int, optional
                Which int is the "bad class"

    Returns
    -------
    total_df : DataFrame
               A new DataFrame containing the ratio
               of "bad" in each feature in df during each
               date bin.
    """
    new_df = feature_types.cont_to_cat(df.drop(date, axis=1), True)
    new_df = performance.fill_missing(new_df, None)
    new_df[date] = df[date]

    bad_df = pd.DataFrame()
    cols = df.columns
    for feature in cols[~cols.isin([label, date])]:

        d0_total = pd.get_dummies(new_df[[feature, date]], columns=[date])
        d_total = d0_total.groupby(feature, as_index=False).sum()
        d_total.columns = ["Bin"] + sorted(list(df[date].unique()))

        d0 = pd.get_dummies(new_df[[feature, date]].loc[new_df[label] == 1], columns=[date])
        d = d0.groupby([feature], observed=False)
        d = d.sum().reindex(d_total["Bin"].unique()).fillna(0).astype(int)

        d.columns = [x.split("_")[-1] for x in d.columns]

        for x in sorted(list(new_df[date].unique())):
            if str(x) in d.columns:
                continue
            d[str(x)] = 0
        d = d[[str(x) for x in sorted(list(new_df[date].unique()))]]
        d = d.reset_index(level=0)

        d.columns = ["Bin"] + sorted(list(new_df[date].unique()))

        for x in d.columns[~d.columns.isin(["Bin"])]:
            d = d.astype({x: int})

            if bad_class == 0:
                d[x] = d_total[x] - d[x]

            d[x] = round(d[x]/d_total[x] * 100, 2)

        d.insert(loc=0, column='Variable', value=feature)

        if feature_types.data_type(df, feature) in ["NC", "DC"]:
            d = d.sort_values(by=["Bin"],
                              key=lambda x:
                                  ([float(a.split(",")[0][1:]) for a in x]))
            d.index = range(len(d))

        bad_df = pd.concat([bad_df, d])
    return bad_df