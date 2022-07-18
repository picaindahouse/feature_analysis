from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import kendalltau
import numpy as np
import pandas as pd

from binary_classification import feature_types

pd.options.mode.chained_assignment = None


class multi_analysis:
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

    def chi_score(self, df):
        """
        Compute the Chi Statistic.

        This function computes the Chi Statistic
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
                list of Chi Statistic for each feature
        """
        chi = []
        for x in df.columns:
            if feature_types.data_type(df, x) in ["DC", "D"] or sum(df[x] < 0):
                new_df = feature_types.cont_to_cat(df[[x]])
                new_df = feature_types.labelled_df(new_df)
                chi.append(chi2(new_df, self.y)[0][0])
            else:
                chi.append(chi2(df[[x]], self.y)[0][0])
        return chi

    def mutual_info(self, df):
        """
        Compute the Mutual Info.

        This function computes the Mutual Info
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
                list of Mutual Info for each feature
        """
        info = []
        for x in df.columns:
            if feature_types.data_type(df, x) in ["DC", "D"]:
                new_df = feature_types.cont_to_cat(df[[x]])
                new_df = feature_types.labelled_df(new_df)
                info.append(mutual_info_classif(new_df, self.y)[0])
            else:
                info.append(mutual_info_classif(df[[x]], self.y)[0])
        return info

    def analyse(self, sort_column=None, on_bin=False):
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

        kendall_corr = self.kendall(df)
        abs_kendall = [abs(x) for x in kendall_corr]

        analysis_df = pd.DataFrame({"Features": df.columns,
                                    "Data Type": types,
                                    "Missing Count": missing[0],
                                    "Missing Rate (%)": missing[1],
                                    "Zero Count": zero[0],
                                    "Zero Rate (%)": zero[1],
                                    "Kendall's Corr": kendall_corr,
                                    "|Kendall's Corr|": abs_kendall,
                                    "Mutual Info": self.mutual_info(df),
                                    "Chi-Score": self.chi_score(df),
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
