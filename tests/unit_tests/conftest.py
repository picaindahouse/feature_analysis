import datetime
import pandas as pd
import pytest
pd.options.mode.chained_assignment = None


@pytest.fixture
def og_df():
    # Data Preprocessing:
    weather = pd.read_csv("datasets/weatherAUS.csv", index_col=0)
    weather.Date = pd.to_datetime(weather.Date)
    return weather


@pytest.fixture
def modified_df():
    # Data Preprocessing:
    weather = pd.read_csv("datasets/weatherAUS.csv", index_col=0)
    weather.Date = pd.to_datetime(weather.Date)
    labelled = [not x for x in weather["RainTomorrow"].isnull()]
    weather_test = weather.loc[labelled]

    def transform_date(x):
        if x < datetime.datetime(2012, 1, 1):
            return "1. Before 2012"
        elif x < datetime.datetime(2015, 1, 1):
            return "2. After 2012, Before 2015"
        else:
            return "3. After 2015"

    weather_test.Date = [transform_date(x) for x in weather_test.Date]
    return weather_test


@pytest.fixture
def label_df():
    return pd.read_csv("datasets/labelledDf.csv", index_col=0)


@pytest.fixture
def contocat_df():
    return pd.read_csv("datasets/contocatDf.csv", index_col=0)


@pytest.fixture
def filled_df():
    return pd.read_csv("datasets/filledDf.csv",
                       index_col=0,
                       keep_default_na=False)


@pytest.fixture
def analysis_df():
    return pd.read_excel("datasets/report.xlsx", sheet_name="Analysis")


@pytest.fixture
def stability_df():
    return pd.read_excel("datasets/report.xlsx", sheet_name="Stability")


@pytest.fixture
def total_df():
    return pd.read_excel("datasets/report.xlsx",
                         sheet_name="Total",
                         keep_default_na=False)


@pytest.fixture
def bad_df():
    return pd.read_excel("datasets/report.xlsx",
                         sheet_name="Bad Rate",
                         keep_default_na=False,
                         na_values="")


@pytest.fixture
def binned_date_df():
    return pd.read_csv("datasets/binned_date_df.csv",
                       index_col=0)


@pytest.fixture
def reg_df():
    weather = pd.read_csv("datasets/weatherAUS.csv", index_col=0)
    weather.Date = pd.to_datetime(weather.Date)
    return weather.loc[[not x for x in weather.MaxTemp.isnull()]]


@pytest.fixture
def reg_analysis_df():
    return pd.read_csv("datasets/reg_analysis.csv")


@pytest.fixture
def reg_stable_df():
    return pd.read_csv("datasets/reg_stable.csv")


@pytest.fixture
def multi_df():
    weather = pd.read_csv("datasets/weatherAUS.csv", index_col=0)
    weather.Date = pd.to_datetime(weather.Date)
    return weather.loc[[not x for x in weather.WindDir9am.isnull()]]


@pytest.fixture
def multi_analysis_df():
    return pd.read_csv("datasets/multi_analysis.csv").drop("Mutual Info",
                                                           axis=1)


@pytest.fixture
def multi_stable_df():
    return pd.read_csv("datasets/multi_stable.csv")
