"""
Functions to help deal with feature types in DataFrame.

Identify data types using data_type and find_type.

Label encode using labelled_df.

Bin all continuous variables in a DataFrame using
cont_to_cat.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def data_type(df, x, bins=5):
    """
    Returns the type of data in a feature of a DataFrame.

    N: Numeric
    NC: Numeric Continuous
    D: Date/Time-Related Data
    DC: Date/Time-Related Data Continuous
    C: Categorical

    Parameters
    ----------
    df : Dataframe
         The dataset the feature is in

    x : String
        Name of feature

    bins: int, optional
          Maximum number of bins each feature should have
          (Only applies to non-categorical data)

    Returns
    -------
    type : String
           The data type
    """
    if is_numeric_dtype(df[x]):
        if df[x].nunique() <= bins:
            return "N"
        return "NC"

    elif df[x].dtype.kind in 'mM':
        if df[x].nunique() <= bins:
            return "D"
        return "DC"

    return "C"


def find_type(df, type_needed, bins=5):
    """
    Returns a list of features with specified data type.

    Will return a list of features which have the same
    data type as type_needed.

    Parameters
    ----------
    df : Dataframe
         The dataset

    type_needed : String or List of Strings
                  The type(s) being searched for.

    bins: int, optional
          Maximum number of bins each feature should have
          (Only applies to non-categorical data)

    Returns
    -------
    type : List of Strings
           A list containing features
    """
    features = []
    for feature in df:
        feature_type = data_type(df, feature, bins)
        if feature_type == "C":
            if feature_type == type_needed:
                features.append(feature)
        elif feature_type in type_needed:
            features.append(feature)
    return features


def labelled_df(df):
    """
    Returns a DataFrame with all its categorical variables
    label encoded.

    Parameters
    ----------
    df : DataFrame
         The dataset

    Returns
    -------
    type : DataFrame
           Dataset with all categorical variables
           label encoded.
    """
    cat_features = find_type(df, "C")
    new_df = df.copy()
    for feature in cat_features:
        le = LabelEncoder()
        le.fit(df[feature])
        new_df[feature] = le.transform(df[feature])
    return new_df


def cont_to_cat(df, fillNA=False, bins=5):
    """
    Returns a DataFrame with all its continuous variables binned.

    All continuous variables with more than 5 unique values
    are binned. Furthermore, if fillNA == True, "-999999999"
    for numbers and "nan" for dates is used. (Can find a more
    appropriate replacement of dates in the future)

    Parameters
    ----------
    df : Dataframe
         The dataset the feature is in

    fillNA : Boolean, optional
             Whether to fill missing values

    bins: int, optional
          Maximum number of bins each feature should have

    Returns
    -------
    type : DataFrame
           Dataset with all continuous variables binned.
    """
    new_df = df.copy()
    cont_features = find_type(df, ["NC", "DC"])
    for feature in cont_features:
        feature_type = data_type(new_df, feature)
        binned_feature = pd.qcut(new_df[feature], bins, duplicates="drop")
        new_df[feature] = binned_feature
        new_df = new_df.astype({feature: str})
        if fillNA:
            if feature_type == "NC":
                new_df[feature] = new_df[feature].apply(lambda x: "-999999999" if x == 'nan' else x)
            else:
                new_df[feature] = new_df[feature].apply(lambda x: "NA" if x == 'nan' else x)

    return new_df


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


def fill_missing(df, strat=None):
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
        feature_type = data_type(df, feature)
        if feature_type in ["N", "NC"]:
            new_df[feature] = new_df[feature].fillna(-999999999)
        else:
            new_df[feature] = new_df[feature].fillna("NA")
    return new_df


def psi(df, label, date, val1, val2):
    """
    Calculate the psi of each feature in a DataFrame over
    the two bins in date column.

    Parameters
    ----------
    df : Dataframe
         The dataset we are working with

    label : String
            Column containing binary output

    date : String
           Column containing datetime object

    val1 : Object
           Bin 1 to be compared in date column

    val2 : Object
           Bin 2 to be compared in date column

    Returns
    -------
    psi_df : DataFrame
             A new DataFrame containing the PSI
             of each feature in df over two bins
             in date column.
    """
    new_df = df.loc[(df[date] == val1) | (df[date] == val2)]
    cols = new_df.columns

    psi_df = pd.DataFrame({"Features": [],
                           "psi": []})
    for feature in cols[~cols.isin([label, date])]:
        b = pd.get_dummies(new_df[[feature, date]], columns=[date])
        b = b.groupby(feature, as_index=False, dropna=False).sum()
        b.columns = ["Bin", "A", "B"]

        b = b.loc[(b["A"] != 0) & (b["B"] != 0)]

        b["A (%)"] = b["A"]/sum(b["A"]) * 100
        b["B (%)"] = b["B"]/sum(b["B"]) * 100
        b = b.drop(["A", "B"], axis=1)

        b["diff"] = b["A (%)"] - b["B (%)"]
        b["log"] = np.log(b["A (%)"]/b["B (%)"])

        b["psi"] = b["diff"]/100 * b["log"]
        psi_values = sum(b["psi"])
        psi_df = pd.concat([psi_df,
                            pd.DataFrame({"Features": [feature],
                                          "psi": [psi_values]})])
    return psi_df