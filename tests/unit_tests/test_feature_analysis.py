import pandas as pd

from binary_class import feature_analysis

MISSING = [[0, 637, 322, 1406, 60843,
            67816, 9330, 9270, 10013,
            3778, 1348, 2630, 1774,
            3610, 14014, 13981, 53657,
            57094, 904, 2726, 1406],
           [0.0, 0.45, 0.23, 0.99,
            42.79, 47.69, 6.56, 6.52,
            7.04, 2.66, 0.95, 1.85,
            1.25, 2.54, 9.86, 9.83,
            37.74, 40.15, 0.64, 1.92,
            0.99]]

ZERO = [[0, 156, 14, 90275, 240,
         2308, 0, 0, 0, 0, 8612,
         1096, 1, 4, 0, 0, 8587,
         4957, 35, 16, 0],
        [0.0, 0.11, 0.01, 63.49,
         0.17, 1.62, 0.0, 0.0,
         0.0, 0.0, 6.06, 0.77,
         0.0, 0.0, 0.0, 0.0,
         6.04, 3.49, 0.02, 0.01,
         0.0]]

IV_SCORES = [0.1622, 0.0454, 0.1373,
             0.4852, 0.0609, 0.6857,
             0.0664, 0.2641, 0.0943,
             0.0541, 0.0388, 0.0435,
             0.4339, 1.1597, 0.2852,
             0.2354, 0.4306, 0.6258,
             0.0096, 0.2065, 0.5048]


def test_missing_rate(modified_df):
    fa_object = feature_analysis(modified_df,
                                 "RainTomorrow",
                                 "Date",
                                 bad_class="Yes")
    assert fa_object.missing_rate(False) == MISSING


def test_zero_rate(modified_df):
    fa_object = feature_analysis(modified_df,
                                 "RainTomorrow",
                                 "Date",
                                 bad_class="Yes")
    assert fa_object.missing_rate(False) == MISSING


def test_analysis(modified_df, analysis_df):
    fa_object = feature_analysis(modified_df,
                                 "RainTomorrow",
                                 "Date",
                                 bad_class="Yes")
    assert pd.testing.assert_frame_equal(analysis_df,
                                         fa_object.analysis(),
                                         check_dtype=False) is None


def test_iv_scores(modified_df):
    fa_object = feature_analysis(modified_df,
                                 "RainTomorrow",
                                 "Date",
                                 bad_class="Yes")
    assert fa_object.iv_scores("IV") == IV_SCORES


def test_stability(modified_df, stability_df):
    fa_object = feature_analysis(modified_df,
                                 "RainTomorrow",
                                 "Date",
                                 bad_class="Yes")
    assert pd.testing.assert_frame_equal(stability_df,
                                         fa_object.stability(),
                                         check_dtype=False) is None


def test_total(modified_df, total_df):
    fa_object = feature_analysis(modified_df,
                                 "RainTomorrow",
                                 "Date",
                                 bad_class="Yes")
    new_total_df = fa_object.total().reset_index(drop=True)
    assert pd.testing.assert_frame_equal(total_df,
                                         new_total_df,
                                         check_dtype=False,
                                         check_names=False) is None


def test_bad_rate(modified_df, bad_df):
    fa_object = feature_analysis(modified_df,
                                 "RainTomorrow",
                                 "Date",
                                 bad_class="Yes")
    new_bad_df = fa_object.bad_rate().reset_index(drop=True)
    assert pd.testing.assert_frame_equal(bad_df,
                                         new_bad_df,
                                         check_dtype=False,
                                         check_names=False) is None
