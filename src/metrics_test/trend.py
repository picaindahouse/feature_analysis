import numpy as np
import pandas as pd

import feature_types
import performance


def trend(df, label, output="DF", to_drop=[], bins=5, bad_class=1):
    """
    Find the trend of the features in a DataFrame

    Return a DataFrame which gives Total, Total Ratio,
    Bad, Bad Rate, WoE, IV for each bin in each feature.

    Parameters
    ----------
    df : Dataframe
         The dataset we are working with

    label : String
            Column containing binary output

    output : String, optional
             Whether to output DataFrame ("DF")
             or output a list containing IV score
             of each feature ("IV")

    to_drop : List of Strings, optional
              List containing features that are not to
              be evaluated

    bins : int, optional
          Maximum number of bins each feature should have

    bad_class : int, optional
                Which int is the "bad class"

    Returns
    -------
    trend_df : DataFrame
               A new DataFrame containing the trend
               of each feature in df.

    IV : List
         If output = "IV", returns the IV score of
         each feature instead.
    """
    new_df = df.copy().drop(to_drop, axis=1)
    new_df = feature_types.cont_to_cat(new_df, True, bins)
    new_df = performance.fill_missing(new_df, None)

    trend_df = pd.DataFrame()
    cols = new_df.columns
    iv = []
    for feature in cols[~cols.isin([label])]:

        d0 = pd.DataFrame({"x": new_df[feature], "y": new_df[label]})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Bin', 'Total', 'Bad']

        if bad_class == 0:
            d["Bad"] = d["Total"] - d["Bad"]

        if feature_types.data_type(df, feature) in ["NC", "DC"]:
            d = d.sort_values(by=["Bin"],
                              key=lambda x:
                                  ([float(a.split(",")[0][1:]) for a in x]))
            d.index = range(len(d))

        d["Total Ratio (%)"] = round(d["Total"]/len(df)*100, 2)
        d["Bad Rate (%)"] = round(d["Bad"]/d["Total"]*100, 2)
        d = d[["Bin", "Total", "Total Ratio (%)", "Bad", "Bad Rate (%)"]]
        d['% of Events'] = np.maximum(d['Bad'], 0.5) / d['Bad'].sum()
        d['Non-Events'] = d['Total'] - d['Bad']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = round(np.log(d['% of Non-Events']/d['% of Events']), 4)
        d['IV'] = round(d['WoE'] * (d['% of Non-Events']-d['% of Events']), 4)
        d.insert(loc=0, column='Variable', value=feature)
        d = d.drop(['% of Events', '% of Non-Events', "Non-Events"], axis=1)
        trend_df = pd.concat([trend_df, d])
        iv.append(round(sum(d["IV"]), 4))

    if output == "IV":
        return iv
    return trend_df