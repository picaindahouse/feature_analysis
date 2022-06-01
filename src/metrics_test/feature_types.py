"""
Functions to help deal with feature types in DataFrame.

Identify data types using data_type and find_type.

Label encode using labelled_df.

Bin all continuous variables in a DataFrame using
cont_to_cat.
"""


import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder


def data_type(df, x, bins=5):
    """
    Returns the type of data in a feature of a DataFrame.

    N: Numeric
    NC: Numeric Continuous
    D: Date/Time-Related Data
    DC: Date/Time-Related Data
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

    if df[x].dtype.kind in 'mM':
        if df[x].nunique() <= bins:
            return "D"
        return "DC"

    return "C"


def find_type(df, type_needed):
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

    Returns
    -------
    type : List of Strings
           A list containing features
    """
    features = []
    for feature in df:
        feature_type = data_type(df, feature)
        if feature_type in type_needed:
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