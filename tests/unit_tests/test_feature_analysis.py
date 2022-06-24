import pandas as pd

from binary_class import feature_analysis

MISSING = [[0, 2, 0, 5, 224, 235, 30, 30, 31, 12, 6, 11,
            6, 9, 43, 43, 196, 200, 4, 8, 5],
           [0.0, 0.41, 0.0, 1.02, 45.9, 48.16, 6.15, 6.15,
            6.35, 2.46, 1.23, 2.25, 1.23, 1.84, 8.81, 8.81,
            40.16, 40.98, 0.82, 1.64, 1.02]]

ZERO = [[0, 0, 0, 309, 0, 10, 0, 0, 0, 0, 25, 1, 0, 0, 0,
         0, 28, 17, 1, 1, 0],
        [0.0, 0.0, 0.0, 63.32, 0.0, 2.05, 0.0, 0.0, 0.0,
         0.0, 5.12, 0.2, 0.0, 0.0, 0.0, 0.0, 5.74, 3.48,
         0.2, 0.2, 0.0]]

IV_SCORES = [0.6645, 0.0965, 0.1803,
             0.4124, 0.1373, 1.1725,
             0.232, 0.4716, 0.2337,
             0.1986, 0.2184, 0.2617,
             0.5284, 1.4659, 0.4514,
             0.3586, 0.5495, 0.7481,
             0.0407, 0.3072, 0.4635]


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
    assert fa_object.zero_rate(False) == ZERO


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
                                         fa_object.stability(False),
                                         check_dtype=False) is None


def test_total(modified_df, total_df):
    fa_object = feature_analysis(modified_df,
                                 "RainTomorrow",
                                 "Date",
                                 bad_class="Yes")
    new_total_df = fa_object.total(False).reset_index(drop=True)
    assert pd.testing.assert_frame_equal(total_df,
                                         new_total_df,
                                         check_dtype=False,
                                         check_names=False) is None


def test_bad_rate(modified_df, bad_df):
    fa_object = feature_analysis(modified_df,
                                 "RainTomorrow",
                                 "Date",
                                 bad_class="Yes")
    new_bad_df = fa_object.bad_rate(False).reset_index(drop=True)
    assert pd.testing.assert_frame_equal(bad_df,
                                         new_bad_df,
                                         check_dtype=False,
                                         check_names=False) is None


def test_date_binned(og_df, binned_date_df):
    fa_object = feature_analysis(og_df,
                                 "RainTomorrow",
                                 "Date",
                                 bad_class="Yes")
    new_binned_date_df = fa_object.stability(True)
    assert pd.testing.assert_frame_equal(binned_date_df,
                                         new_binned_date_df,
                                         check_dtype=False,
                                         check_names=False) is None
