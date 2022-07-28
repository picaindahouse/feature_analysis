from sklearn.feature_selection import r_regression
from sklearn.feature_selection import f_classif
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from binary_classification import feature_types

pd.options.mode.chained_assignment = None


class feature_analysis:
    def __init__(self, df, label, date=None, to_drop=None):
        self.df = df.copy()
        self.label = label
        self.date = date
        self.to_drop = to_drop

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

    def data_types(self):
        """
        Return the data type (numerical/categorical) of each feautre.

        Parameters
        ----------
        self : Instance
               Current instance of feature_anaylsis
               class

        Returns
        -------
        scores : list
                 list of data type for each feature
        """
        types = []
        for column in self.features_df.columns:
            data_type = feature_types.data_type(self.features_df, column)
            if data_type in ["C"]:
                types.append("Categorical")
            elif data_type in ["D", "DC"]:
                date = feature_types.cont_to_cat(self.df[[column]])
                self.features_df[column] = date
                types.append("DateTime")
            else:
                types.append("Numerical")
        return types

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

    def pearsons(self, df, frames=True):
        """
        Compute the Pearson's Correlation.

        This function computes the Pearson's Correlation
        of each feature in a dataframe.

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
                dataframe of Pearson's Correlation for each feature or
                list of Pearson's Correlation for each feature
        """
        corr = r_regression(df, self.y)
        if not frames:
            return [corr, abs(corr)]
        return pd.DataFrame({"Pearson's Corr": corr,
                            "|Pearson's Corr|": abs(corr)},
                            index=df.columns)

    def spearmans(self, df, frames=True):
        """
        Compute the Spearman's Correlation.

        This function computes the Spearman's Correlation
        of each feature in a dataframe.

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
                dataframe of Spearman's Correlation for each feature or
                list of Spearman's Correlation for each feature
        """
        corr = []
        for column in df.columns:
            corr.append(spearmanr(df[column], self.y)[0])
        if not frames:
            return [corr, [abs(x) for x in corr]]
        return pd.DataFrame({"Spearman's Corr": corr,
                            "|Spearman's Corr|": [abs(x) for x in corr]},
                            index=df.columns)

    def anova_F_score(self, df):
        """
        Compute the Anova F Score.

        This function computes the Anova F score
        of each feature in a dataframe in relation
        to label.

        Parameters
        ----------
        self : Instance
               Current instance of feature_anaylsis
               class

        df : Dataframe

        Returns
        -------
        scores : list
                list of Anova F score for each feature
        """
        return f_classif(df, self.y)[0]

    def kendall(self, df):
        """
        Compute the Kendall Tau.

        This function computes the Kendall Tau
        of each feature in a dataframe in relation
        to label.

        Parameters
        ----------
        self : Instance
               Current instance of feature_anaylsis
               class

        df : Dataframe

        Returns
        -------
        scores : list
                list of Kendall Tau for each feature
        """
        kendalls = []
        for column in df.columns:
            kendalls.append(kendalltau(df[column], self.y)[0])
        return kendalls

    def analysis(self, sort_column=None, on_bin=False):
        """
        Correlation analysis for the features in a DataFrame

        Return a DataFrame which gives the Missing Count,
        Missing Rate, Zero Count, Zero Rate, Pearson's Correlation,
        |Pearson's Correlation|, Spearman's Correlation,
        |Spearman's Correlation|, anova of each feature in the
        DataFrame in relation to the label.

        Note, Missing values are replaced by:
        1) Numerical -> Mean of column
        2) Catergorical -> "NA"

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
                A new DataFrame containing the analysis
                of each feature in df.
        """
        types = self.data_types()
        df = self.features_df

        missing = self.missing_rate(False)
        zero = self.zero_rate(False)

        if on_bin:
            df = feature_types.cont_to_cat(df, self.label)

        df = feature_types.fill_missing(df, "mean")
        df = feature_types.labelled_df(df)

        pearsons = self.pearsons(df, False)
        spearmans = self.spearmans(df, False)
        kendall_corr = self.kendall(df)
        abs_kendall = [abs(x) for x in kendall_corr]

        analysis_df = pd.DataFrame({"Features": df.columns,
                                    "Data Type": types,
                                    "Missing Count": missing[0],
                                    "Missing Rate (%)": missing[1],
                                    "Zero Count": zero[0],
                                    "Zero Rate (%)": zero[1],
                                    "Pearson's Corr": pearsons[0],
                                    "|Pearson's Corr|": pearsons[1],
                                    "Spearman's Corr": spearmans[0],
                                    "|Spearman's Corr|": spearmans[1],
                                    "Kendall's Corr": kendall_corr,
                                    "|Kendall's Corr|": abs_kendall,
                                    "Anova F": self.anova_F_score(df)})

        if sort_column is not None:
            return analysis_df.sort_values(by=sort_column, ascending=False)
        return analysis_df

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
        df = feature_types.fill_missing(self.df, "mean")
        if not bin_dates:
            new_df = feature_types.cont_to_cat(df.drop(self.date, axis=1),
                                               True,
                                               bins)
            new_df[self.date] = self.df[self.date]
        else:
            new_df = feature_types.cont_to_cat(df, True, bins)

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
        for x in tqdm(range(1, 3)):
            if x == 1:
                df1 = self.analysis(on_bin=on_bin)

            else:
                if self.date is None:
                    dfs = {"Analysis": df1}
                    break

                else:
                    df2 = self.stability(bin_dates, bins)
                    dfs = {"Analysis": df1,
                           "Stability": df2}

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
