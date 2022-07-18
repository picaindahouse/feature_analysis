import pandas as pd

from multi_classification import multi_analysis

DATA_TYPES = ['DateTime', 'Categorical', 'Numerical',
              'Numerical', 'Numerical', 'Numerical',
              'Numerical', 'Categorical', 'Numerical',
              'Categorical', 'Numerical', 'Numerical',
              'Numerical', 'Numerical', 'Numerical',
              'Numerical', 'Numerical', 'Numerical',
              'Numerical', 'Numerical', 'Categorical',
              'Categorical']

MISSING = [[0, 0, 3, 3, 12, 209, 219, 24, 24, 5, 0, 5, 6,
            10, 34, 35, 185, 188, 4, 9, 12, 11],
           [0.0, 0.0, 0.64, 0.64, 2.56, 44.66, 46.79,
            5.13, 5.13, 1.07, 0.0, 1.07, 1.28, 2.14,
            7.26, 7.48, 39.53, 40.17, 0.85, 1.92, 2.56,
            2.35]]

ZERO = [[0, 0, 0, 0, 289, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         27, 16, 1, 1, 0, 0],
        [0.0, 0.0, 0.0, 0.0, 61.75, 0.0, 1.92, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.77, 3.42,
         0.21, 0.21, 0.0, 0.0]]


def test_data_types(multi_df):
    report = multi_analysis(multi_df, "WindDir9am")
    assert report.data_types() == DATA_TYPES


def test_missing_rate(multi_df):
    report = multi_analysis(multi_df, "WindDir9am")
    assert report.missing_rate(False) == MISSING


def test_zero_rate(multi_df):
    report = multi_analysis(multi_df, "WindDir9am")
    assert report.zero_rate(False) == ZERO


def test_analysis(multi_df, multi_analysis_df):
    report = multi_analysis(multi_df, "WindDir9am", "Date")
    new_analysis_df = report.analyse().drop("Mutual Info", axis=1)
    assert pd.testing.assert_frame_equal(new_analysis_df,
                                         multi_analysis_df) is None


def test_stability(multi_df, multi_stable_df):
    report = multi_analysis(multi_df, "WindDir9am", "Date")
    new_stable_df = report.stability(True)
    assert pd.testing.assert_frame_equal(new_stable_df,
                                         multi_stable_df) is None
