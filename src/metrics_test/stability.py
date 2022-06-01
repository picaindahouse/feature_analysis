import numpy as np
import pandas as pd

from feature_types import cont_to_cat


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


def stability(df, label, date):
    """
    Calculate the psi of each feature in a DataFrame over
    the different bins in date feature.

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
    stability_df : DataFrame
                   A new DataFrame containing the PSI
                   of each feature in df over date feature.
    """

    # Turn continuous variables categorical
    new_df = cont_to_cat(df.drop(date, axis=1))
    new_df[date] = df[date]

    # Find the dates we are grouping the data into
    dates = sorted(list(new_df[date].unique()))

    # Find the columns in the dataset and then the features we wish to find the psi values for
    cols = new_df.columns
    features = cols[~cols.isin([label, date])]

    # Create our new df we wish to output
    stability_df = pd.DataFrame()
    stability_df["Variable"] = features
    stability_df["Metric"] = "psi"

    # Find the psi values between consecutive dates
    for x in range(len(dates)):
        if x == 0:
            stability_df[dates[x]] = np.nan
            continue
        psi_values = psi(new_df, label, date, dates[x-1], dates[x]).psi

        stability_df[dates[x]] = [round(x, 5) for x in psi_values]

    return stability_df