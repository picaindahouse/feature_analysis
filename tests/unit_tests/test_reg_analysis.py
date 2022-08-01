import pandas as pd

from regression import feature_analysis

DATA_TYPES = ['DateTime', 'Categorical', 'Numerical',
              'Numerical', 'Numerical', 'Numerical',
              'Categorical', 'Numerical', 'Categorical',
              'Categorical', 'Numerical', 'Numerical',
              'Numerical', 'Numerical', 'Numerical',
              'Numerical', 'Numerical', 'Numerical',
              'Numerical', 'Numerical', 'Categorical',
              'Categorical']

MISSING = [[0, 0, 2, 11, 230, 240, 33, 33, 31, 12, 6,
            11, 7, 10, 44, 44, 202, 205, 5, 9, 11, 8],
           [0.0, 0.0, 0.4, 2.22, 46.37, 48.39, 6.65,
            6.65, 6.25, 2.42, 1.21, 2.22, 1.41, 2.02,
            8.87, 8.87, 40.73, 41.33, 1.01, 1.81,
            2.22, 1.61]]

ZERO = [[0, 0, 0, 309, 0, 10, 0, 0, 0, 0, 25, 1, 0, 0,
         0, 0, 28, 17, 1, 1, 0, 0],
        [0.0, 0.0, 0.0, 62.3, 0.0, 2.02, 0.0, 0.0, 0.0,
         0.0, 5.04, 0.2, 0.0, 0.0, 0.0, 0.0, 5.65, 3.43,
         0.2, 0.2, 0.0, 0.0]]


def test_data_types(reg_df):
    report = feature_analysis(reg_df, "MaxTemp")
    assert report.data_types() == DATA_TYPES


def test_missing_rate(reg_df):
    report = feature_analysis(reg_df, "MaxTemp")
    assert report.missing_rate(False) == MISSING


def test_zero_rate(reg_df):
    report = feature_analysis(reg_df, "MaxTemp")
    assert report.zero_rate(False) == ZERO


def test_analysis(reg_df, reg_analysis_df):
    report = feature_analysis(reg_df, "MaxTemp", "Date")
    new_analysis_df = report.analysis()
    assert pd.testing.assert_frame_equal(new_analysis_df,
                                         reg_analysis_df) is None


def test_stability(reg_df, reg_stable_df):
    report = feature_analysis(reg_df, "MaxTemp", "Date")
    new_stable_df = report.stability(True)
    assert pd.testing.assert_frame_equal(new_stable_df,
                                         reg_stable_df) is None
