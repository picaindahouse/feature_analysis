"""
Performance of the features of a binary classification dataset.
Use a couple of metrics to evaluate said performance:

    1) Missing Count and Missing Rate
    2) Zero Count and Zero Rate
    3) AUC score
    3) KS score
    4) IV score

Each of the above can be found in its own unique dataframe.
Can also see all of them combined and presented in one dataframe.
"""


import numpy as np
import pandas as pd
import toad
from sklearn.impute import SimpleImputer
from toad.metrics import KS, AUC

import feature_types
import trend


def missing_rate(df, frame=True):
    """
    Compute the missing rate.

    This function computes the missing count and
    missing rate of each feature in a dataframe.

    Parameters
    ----------
    df : Dataframe
         The dataset to evaluate

    frame : Boolean, optional
            Whether to return a DataFrame

    Returns
    -------
    scores : dataframe or list
             dataframe of missing count and rate for each feature or
             list of missing count and rate for each feature
    """
    missing = pd.DataFrame(df.isnull().sum(), columns=["Missing Count"])
    missing["Missing Rate (%)"] = round(missing["Missing Count"]/len(df) * 100, 2)
    if frame:
        return missing
    return [missing["Missing Count"].tolist(),
            missing["Missing Rate (%)"].tolist()]


def zero_rate(df, frame=True):
    """
    Compute the zero rate.

    This function computes the zero count and
    zero rate of each feature in a dataframe.

    Parameters
    ----------
    df : Dataframe
         The dataset to evaluate

    frame : Boolean, optional
            Whether to return a DataFrame

    Returns
    -------
    scores : dataframe or list
             dataframe of zero count and rate for each feature or
             list of zero count and rate for each feature
    """
    zero = pd.DataFrame((df == 0).sum(), columns=["Zero Count"])
    zero["Zero Rate (%)"] = round(zero["Zero Count"]/len(df) * 100, 2)
    if frame:
        return zero
    return [zero["Zero Count"].tolist(),
            zero["Zero Rate (%)"].tolist()]


def auc_features(df, y, frame=True):
    """
    Computes the area under the receiver-operater characteristic (AUC)

    This function computes the AUC score for each feature
    in a binary classification.

    Parameters
    ----------
    df : Dataframe
         The dataset to evaluate

    y : List of binary numbers
        The binary classification 'label'

    frame : Boolean, optional
            Whether to return a DataFrame

    Returns
    -------
    scores : dataframe or list
             dataframe of AUC score for each feature or
             list of AUC score for each feature
    """
    auc = []
    for x in df:
        auc.append(AUC(df[x], y))
    if frame:
        return pd.DataFrame({"AUC": auc}, index=df.columns)
    return [round(x, 2) if x > 0.5 else round(1-x, 2) for x in auc]


def ks_features(df, y, frame=True):
    """
    Computes the Kolmogorov–Smirnov (KS) Score

    This function computes the KS score for each feature
    in a binary classification.

    Parameters
    ----------
    df : Dataframe
         The dataset to evaluate

    y : List of binary numbers
        The binary classification 'label'

    frame : Boolean, optional
            Whether to return a DataFrame

    Returns
    -------
    scores : dataframe or list
             dataframe of KS score for each feature or
             list of KS score for each feature
    """
    ks = []
    for x in df:
        ks.append(round(KS(df[x], y)*100, 2))
    if frame:
        return pd.DataFrame({"KS (%)": ks}, index=df.columns)
    return ks


def iv_features(df, label, frame=True):
    """
    Computes the Information Value (IV).

    This function computes the IV for each feature
    in a binary classification.

    Parameters
    ----------
    df : Dataframe
         The dataset to evaluate

    label : String
            Column containing binary output

    frame : Boolean, optional
            Whether to return a DataFrame

    Returns
    -------
    scores : dataframe or list
             dataframe of IV for each feature or
             list of IV for each feature
    """
    if frame:
        return pd.DataFrame({"IV": trend.trend(df, label, "IV")},
                            index=[df.drop(label, axis=1).columns])

    return trend.trend(df, label, "IV")


def find_missing(df):
    """
    Find all columns that contain missing values in a DataFrame

    Parameters
    ----------
    df : Dataframe
         The dataset to evaluate

    Returns
    -------
    missing : list of strings
             list of all columns with missing values
    """
    missing = []
    for x in df:
        if df[x].isnull().sum() > 0:
            missing.append(x)
    return missing


def fill_missing(df, strat):
    """
    Fill missing values in DataFrame

    Fill missing values in DataFrame.
    Either via a strat given by user, eg: Mean
    or by replacing empty space with -999999999 or "NA".

    Parameters
    ----------
    df : Dataframe
         The dataset to fill

    strat : String
            A strategy that can be used by SimpleImputer
            from sklearn.impute
    Returns
    -------
    missing : DataFrame
              A new DataFrame where all missing values are filled.
    """
    if strat is not None:
        imp = SimpleImputer(strategy=strat)
        return pd.DataFrame(imp.fit_transform(df),
                            columns=df.columns)

    new_df = df.copy()
    for feature in find_missing(df):
        feature_type = feature_types.data_type(df, feature)
        if feature_type in ["N", "NC"]:
            new_df[feature] = new_df[feature].fillna(-999999999)
        else:
            new_df[feature] = new_df[feature].fillna("NA")
    return new_df


def evaluation(df, label, strat=None, sort=None, on_bin=False, to_drop=[]):
    """
    Evaluate the features of a DataFrame

    Return a DataFrame which gives the Missing Count,
    Missing Rate, Zero Count, Zero Rate, AUC score,
    KS score, IV of each feature in the DataFrame
    in relation to the label.

    Parameters
    ----------
    df : Dataframe
         The dataset whose features we are evaluating

    label : String
            Column containing binary output

    strat : String, optional
            A strategy that can be used by SimpleImputer
            from sklearn.impute

    sort : String or List of Strings, optional
           Column(s) to sort the resulting df by

    on_bin : Boolean, optional
             Whether to find AUC/KS score on
             binned or continuous values of the
             continuous features in df

    to_drop : List of Strings, optional
              List containing features that are not to
              be evaluated

    Returns
    -------
    new_df : DataFrame
             A new DataFrame containing the evaluation
             of each feature in df.
    """
    new_df = df.copy().drop(to_drop + [label], axis=1)
    y = df[label]

    missing = missing_rate(new_df, False)
    zero = zero_rate(new_df, False)

    if on_bin:
        new_df = feature_types.cont_to_cat(new_df, label)

    new_df = fill_missing(new_df, strat)
    new_df = feature_types.labelled_df(new_df)

    evaluation = pd.DataFrame({"Features": new_df.columns,
                               "Missing Count": missing[0],
                               "Missing Rate (%)": missing[1],
                               "Zero Count": zero[0],
                               "Zero Rate (%)": zero[1],
                               "AUC": auc_features(new_df, y, False),
                               "KS (%)": ks_features(new_df, y, False),
                               "IV": iv_features(df.drop(to_drop, axis=1),
                                                 label, False)})

    if sort is not None:
        return evaluation.sort_values(by=sort, ascending=False)
    return evaluation