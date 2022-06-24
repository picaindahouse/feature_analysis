import numpy as np
import pandas as pd
from toad.metrics import KS, AUC
from tqdm.auto import tqdm

from . import feature_types


# For binary classification only (at least for now)
class feature_analysis:
    def __init__(self, df, label, date=None, to_drop=None, bad_class=1):
        self.df = df.copy()
        self.df[label] = [1 if x == bad_class else 0 for x in self.df[label]]
        self.label = label
        self.date = date
        self.to_drop = to_drop
        self.bad = bad_class

        self.features_df = self.df
        if self.date is None:
            date = []
        else:
            date = [self.date]
        if to_drop is None:
            self.features_df = self.df.drop(date + [label], axis=1)
        else:
            self.features_df = self.df.drop(to_drop + date + [label], axis=1)

        self.y = self.df[label]

    def missing_rate(self, frame=True):
        """
        Compute the missing rate.

        This function computes the missing count and
        missing rate of each feature in a dataframe.

        Parameters
        ----------
        self : Instance
               Current instance of feature_anaylsis
               class

        frame : Boolean, optional
                Whether to return a DataFrame

        Returns
        -------
        scores : dataframe or list
                dataframe of missing count and rate for each feature or
                list of missing count and rate for each feature
        """
        df = self.features_df
        missing = pd.DataFrame(df.isnull().sum(), columns=["Missing Count"])
        missing["Missing Rate (%)"] = round(
            missing["Missing Count"] / len(df) * 100, 2)
        if frame:
            return missing
        return [missing["Missing Count"].tolist(),
                missing["Missing Rate (%)"].tolist()]

    def fill_NA(self, strat):
        """
        Parameters
        ----------
        self : Instance
               Current instance of feature_anaylsis
               class

        strat : String, optional
                A strategy that can be used by SimpleImputer
                from sklearn.impute
        """
        df = feature_types.fill_missing(self.df, strat)
        self = feature_analysis(df,
                                self.label, self.date, self.to_drop, self.bad)

    def zero_rate(self, frame=True):
        """
        Compute the zero rate.

        This function computes the zero count and
        zero rate of each feature in a dataframe.

        Parameters
        ----------
        self : Instance
               Current instance of feature_anaylsis
               class

        frame : Boolean, optional
                Whether to return a DataFrame

        Returns
        -------
        scores : dataframe or list
                dataframe of zero count and rate for each feature or
                list of zero count and rate for each feature
        """
        df = self.features_df
        zero = pd.DataFrame((df == 0).sum(), columns=["Zero Count"])
        zero["Zero Rate (%)"] = round(zero["Zero Count"] / len(df) * 100, 2)
        if frame:
            return zero
        return [zero["Zero Count"].tolist(),
                zero["Zero Rate (%)"].tolist()]

    def auc_scores(self, df, frame=True):
        """
        Computes the area under the receiver-operater characteristic (AUC)

        This function computes the AUC score for each feature
        in a binary classification.

        Parameters
        ----------
        self : Instance
               Current instance of feature_anaylsis
               class

        frame : Boolean, optional
                Whether to return a DataFrame

        Returns
        -------
        scores : dataframe or list
                dataframe of AUC score for each feature or
                list of AUC score for each feature
        """
        y = self.y
        auc = []
        for x in df:
            score = AUC(df[x], y)
            auc.append(round(score, 2) if score > 0.5 else round(1 - score, 2))

        if frame:
            return pd.DataFrame({"AUC": auc}, index=df.columns)
        return auc

    def ks_scores(self, df, frame=True):
        """
        Computes the Kolmogorov–Smirnov (KS) Score

        This function computes the KS score for each feature
        in a binary classification.

        Parameters
        ----------
        self : Instance
               Current instance of feature_anaylsis
               class

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
        y = self.y
        ks = []
        for x in df:
            ks.append(round(KS(df[x], y) * 100, 2))
        if frame:
            return pd.DataFrame({"KS (%)": ks}, index=df.columns)
        return ks

    def analysis(self, sort=None, on_bin=False):
        """
        Analyse the features of a DataFrame

        Return a DataFrame which gives the Missing Count,
        Missing Rate, Zero Count, Zero Rate, AUC score,
        KS score, IV of each feature in the DataFrame
        in relation to the label.

        Parameters
        ----------
        self : Instance
               Current instance of feature_anaylsis
               class

        sort : String or List of Strings, optional
               Column(s) to sort the resulting df by

        on_bin : Boolean, optional
                Whether to find AUC/KS score on
                binned or continuous values of the
                continuous features in df

        Returns
        -------
        new_df : DataFrame
                A new DataFrame containing the evaluation
                of each feature in df.
        """
        df = self.features_df

        missing = self.missing_rate(False)
        zero = self.zero_rate(False)

        if on_bin:
            df = feature_types.cont_to_cat(df, self.label)
        else:
            dates = feature_types.find_type(df, ["D", "DC"])
            for date in dates:
                df[date] = [str(x) if x == x else "NA" for x in df[date]]

        df = feature_types.fill_missing(df)
        df = feature_types.labelled_df(df)

        analysis_df = pd.DataFrame({"Features": df.columns,
                                    "Missing Count": missing[0],
                                    "Missing Rate (%)": missing[1],
                                    "Zero Count": zero[0],
                                    "Zero Rate (%)": zero[1],
                                    "AUC": self.auc_scores(df, False),
                                    "KS (%)": self.ks_scores(df, False),
                                    "IV": self.iv_scores("IV")})

        if sort is not None:
            return analysis_df.sort_values(by=sort, ascending=False)
        return analysis_df

    def iv_scores(self, output="DF", bins=5):
        """
        Find the trend of the features in a DataFrame

        Return a DataFrame which gives Total, Total Ratio,
        Bad, Bad Rate, WoE, IV for each bin in each feature.

        Parameters
        ----------
        self : Instance
                Current instance of feature_anaylsis
                class

        output : String, optional
                Whether to output DataFrame ("DF")
                or output a list containing IV score
                of each feature ("IV")

        bins : int, optional
            Maximum number of bins each feature should have

        Returns
        -------
        trend_df : DataFrame
                A new DataFrame containing the trend
                of each feature in df.

        IV : List
            If output = "IV", returns the IV score of
            each feature instead.
        """
        if self.date is not None:
            df = self.df.drop(self.date, axis=1)
        if self.to_drop is not None:
            df = self.df.drop(self.to_drop, axis=1)
        df = feature_types.cont_to_cat(df, True, bins)
        df = feature_types.fill_missing(df, None)

        trend_df = pd.DataFrame()
        cols = df.columns
        iv = []
        for feature in cols[~cols.isin([self.label])]:
            d0 = pd.DataFrame({"x": df[feature], "y": df[self.label]})
            d = d0.groupby("x",
                           as_index=False,
                           dropna=False).agg({"y": ["count", "sum"]})
            d.columns = ['Bin', 'Total', 'Bad']

            if feature_types.data_type(df, feature) in ["NC", "DC"]:
                d = d.sort_values(by=["Bin"],
                                  key=lambda x:
                                  ([float(a.split(",")[0][1:]) for a in x]))
                d.index = range(len(d))

            d["Total Ratio (%)"] = round(d["Total"] / len(df) * 100, 2)
            d["Bad Rate (%)"] = round(d["Bad"] / d["Total"] * 100, 2)
            d = d[["Bin", "Total", "Total Ratio (%)", "Bad", "Bad Rate (%)"]]
            d['% of Events'] = np.maximum(d['Bad'], 0.5) / d['Bad'].sum()
            d['Non-Events'] = d['Total'] - d['Bad']
            d['% Non-Events'] = np.maximum(d['Non-Events'],
                                           0.5) / d['Non-Events'].sum()
            d['WoE'] = round(np.log(d['% Non-Events'] / d['% of Events']), 4)
            d['IV'] = round(d['WoE'] * (d['% Non-Events'] - d['% of Events']),
                            4)
            d.insert(loc=0, column='Variable', value=feature)
            d = d.drop(['% of Events', '% Non-Events', "Non-Events"], axis=1)
            trend_df = pd.concat([trend_df, d])
            iv.append(round(sum(d["IV"]), 4))

        if output == "IV":
            return iv
        return trend_df

    def stability(self, bin_dates, bins=5):
        """
        Calculate the psi of each feature in a DataFrame over
        the different bins in date feature.

        Parameters
        ----------
        self : Instance
                Current instance of feature_anaylsis
                class

        bin_dates: boolean
                   True if want to bin dates else false

        bins : int, optional
               Maximum number of bins each feature should have

        Returns
        -------
        stability_df : DataFrame
                    A new DataFrame containing the PSI
                    of each feature in df over date feature.
        """
        if self.date is None:
            return "No date column given"
        # Turn continuous variables categorical
        if not bin_dates:
            new_df = feature_types.cont_to_cat(self.df.drop(self.date, axis=1),
                                               True,
                                               bins)
            new_df[self.date] = self.df[self.date]
        else:
            new_df = feature_types.cont_to_cat(self.df, True, bins)

        # Find the dates we are grouping the data into
        dates = sorted(list(new_df[self.date].unique()))

        cols = new_df.columns
        features = cols[~cols.isin([self.label, self.date])]

        # Create our new df we wish to output
        stability_df = pd.DataFrame()
        stability_df["Variable"] = features
        stability_df["Metric"] = "psi"

        # Find the psi values between consecutive dates
        for x in range(len(dates)):
            if x == 0:
                stability_df[dates[x]] = np.nan
                continue
            psi_values = feature_types.psi(new_df,
                                           self.label,
                                           self.date,
                                           dates[x - 1],
                                           dates[x]).psi

            stability_df[dates[x]] = [round(x, 5) for x in psi_values]

        return stability_df

    def total(self, bin_dates, bins=5):
        """
        Find the total number of occurences of each bin for each
        feature in a DataFrame over the bins in date column.

         Parameters
        ----------
        self : Instance
                Current instance of feature_anaylsis
                class

        bin_dates: boolean
                   True if want to bin dates else false

        bins : int, optional
               Maximum number of bins each feature should have

        Returns
        -------
        total_df : DataFrame
                A new DataFrame containing the total
                of each feature in df during each
                date bin.
        """
        if self.date is None:
            return "No date column given"

        # Turn continuous variables categorical
        if not bin_dates:
            new_df = feature_types.cont_to_cat(self.df.drop(self.date, axis=1),
                                               True,
                                               bins)
            new_df[self.date] = self.df[self.date]
        else:
            new_df = feature_types.cont_to_cat(self.df, True, bins)

        new_df = feature_types.fill_missing(new_df)

        total_df = pd.DataFrame()
        cols = self.df.columns

        for feature in cols[~cols.isin([self.label, self.date])]:

            d0 = pd.get_dummies(new_df[[feature, self.date]],
                                columns=[self.date])
            d = d0.groupby(feature, as_index=False).sum()
            d.columns = ["Bin"] + sorted(list(new_df[self.date].unique()))

            for x in d.columns[~d.columns.isin(["Bin"])]:
                d = d.astype({x: int})

            d.insert(loc=0, column="Variable", value=feature)

            if feature_types.data_type(self.df, feature) in ["NC"]:
                d = d.sort_values(by=["Bin"],
                                  key=lambda x:
                                  ([float(a.split(",")[0][1:]) for a in x]))
                d.index = range(len(d))

            total_df = pd.concat([total_df, d])
        return total_df

    def bad_rate(self, bin_dates, bins=5):
        """
        Find the ratio of "bad" in each bin for each feature
        in a DataFrame over the bins in date column.

        Parameters
        ----------
        self : Instance
                Current instance of feature_anaylsis
                class

        bin_dates: boolean
                   True if want to bin dates else false

        bins : int, optional
               Maximum number of bins each feature should have

        Returns
        -------
        total_df : DataFrame
                A new DataFrame containing the ratio
                of "bad" in each feature in df during each
                date bin.
        """
        if self.date is None:
            return "No date column given"

        if not bin_dates:
            new_df = feature_types.cont_to_cat(self.df.drop(self.date, axis=1),
                                               True,
                                               bins)
            new_df[self.date] = self.df[self.date]
        else:
            new_df = feature_types.cont_to_cat(self.df, True, bins)

        new_df = feature_types.fill_missing(new_df)

        bad_df = pd.DataFrame()
        cols = self.df.columns
        for feature in cols[~cols.isin([self.label, self.date])]:

            d0_total = pd.get_dummies(new_df[[feature, self.date]],
                                      columns=[self.date])
            d_total = d0_total.groupby(feature, as_index=False).sum()
            dates = sorted(list(new_df[self.date].unique()))
            d_total.columns = ["Bin"] + dates

            fil_df = new_df[[feature, self.date]].loc[new_df[self.label] == 1]
            d0 = pd.get_dummies(fil_df, columns=[self.date])
            d = d0.groupby([feature], observed=False)
            d = d.sum().reindex(d_total["Bin"].unique()).fillna(0).astype(int)

            d.columns = [x.split("_")[-1] for x in d.columns]

            for x in sorted(list(new_df[self.date].unique())):
                if str(x) in d.columns:
                    continue
                d[str(x)] = 0
            d = d[[str(x) for x in sorted(list(new_df[self.date].unique()))]]
            d = d.reset_index(level=0)

            d.columns = ["Bin"] + sorted(list(new_df[self.date].unique()))

            for x in d.columns[~d.columns.isin(["Bin"])]:
                d = d.astype({x: int})
                d[x] = round(d[x] / d_total[x] * 100, 2)

            d.insert(loc=0, column='Variable', value=feature)

            if feature_types.data_type(self.df, feature) in ["NC"]:
                d = d.sort_values(by=["Bin"],
                                  key=lambda x:
                                  ([float(a.split(",")[0][1:]) for a in x]))
                d.index = range(len(d))

            bad_df = pd.concat([bad_df, d])
        return bad_df

    def to_excel(self, bin_dates, name='report.xlsx', on_bin=False, bins=5):
        """
        Saves an excelsheet with all the dataframes available
        in overall having 1 worksheet to themselves.

        Parameters
        ----------
        self : Instance
                Current instance of feature_anaylsis
                class

        bin_dates: boolean
                   True if want to bin dates else false

        name: String, optional
                Name of excelsheet saved

        on_bin : Boolean, optional
                Whether to find AUC/KS score on
                binned or continuous values of the
                continuous features in df

        bins : int, optional
            Maximum number of bins each feature should have

        Saves
        -------
        Xslx : xslx file
            Excel file containing all the dataframes from
            overall.
        """
        for x in tqdm(range(1, 6)):
            if x == 1:
                df1 = self.analysis(on_bin=on_bin)
            elif x == 2:
                df2 = self.iv_scores(bins=bins)
            else:
                if self.date is None:
                    dfs = {"Analysis": df1,
                           "IV Scores": df2}
                    break

                elif x == 3:
                    df3 = self.stability(bin_dates, bins)

                elif x == 4:
                    df4 = self.total(bin_dates, bins)

                else:
                    dfs = {"Analysis": df1,
                           "IV Scores": df2,
                           "Stability": df3,
                           "Total": df4,
                           "Bad Rate": self.bad_rate(bin_dates, bins)}

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(name, engine='xlsxwriter')

        for sheetname, df_n in dfs.items():
            df_n.to_excel(writer, sheet_name=sheetname, index=False)
            worksheet = writer.sheets[sheetname]
            for idx, col in enumerate(df_n):
                series = df_n[col]
                max_len = max((series.astype(str).map(len).max(),
                               len(str(series.name)))) + 1
                worksheet.set_column(idx, idx, max_len)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
