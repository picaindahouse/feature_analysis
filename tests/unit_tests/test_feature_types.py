import pandas as pd
import pytest

from binary_class import feature_types

C = ["Location",
     "WindGustDir",
     "WindDir9am",
     "WindDir3pm",
     "RainToday",
     "RainTomorrow"]

NC = ['MinTemp',
      'MaxTemp',
      'Rainfall',
      'Evaporation',
      'Sunshine',
      'WindGustSpeed',
      'WindSpeed9am',
      'WindSpeed3pm',
      'Humidity9am',
      'Humidity3pm',
      'Pressure9am',
      'Pressure3pm',
      'Cloud9am',
      'Cloud3pm',
      'Temp9am',
      'Temp3pm']

MISSING = ['MinTemp',
           'Rainfall',
           'Evaporation',
           'Sunshine',
           'WindGustDir',
           'WindGustSpeed',
           'WindDir9am',
           'WindDir3pm',
           'WindSpeed9am',
           'WindSpeed3pm',
           'Humidity9am',
           'Humidity3pm',
           'Pressure9am',
           'Pressure3pm',
           'Cloud9am',
           'Cloud3pm',
           'Temp9am',
           'Temp3pm',
           'RainToday']

PSI1 = [0.6021180254495753,
        0.018236833933908944,
        0.042224567041398126,
        0.02892202942332511,
        0.10183533443234234,
        0.09079400143633055,
        0.0656008439279806,
        0.08468486947193922,
        0.5562147321845746,
        0.21497016024247473,
        0.050293061010997044,
        0.041055737725403205,
        0.07690999651994494,
        0.0824247887507653,
        0.10122201207641006,
        0.08386998618634965,
        0.09914200341249758,
        0.08197315528681638,
        0.01890814959526557,
        0.05833854825048768,
        0.06440087138027063]

PSI2 = [0.6024573756626945,
        0.15809226869173842,
        0.056775737122305256,
        0.013624592156900404,
        0.2345025218921846,
        0.17033273295170107,
        0.13430739281082052,
        0.03464734670419742,
        0.3224620265964764,
        0.09812087671214752,
        0.009674522543387354,
        0.026468739454294933,
        0.03627969465670946,
        0.11139028305901766,
        0.0842814397643875,
        0.03882943099618481,
        0.054284641264211744,
        0.040488456041845144,
        0.06833524924512668,
        0.04451318470561884,
        0.010596661888723668]


@pytest.mark.parametrize(["column", "type1", "type2"],
                         [("Location", "C", "C"),
                          ("MinTemp", "NC", "NC"),
                          ("WindSpeed9am", "NC", "NC"),
                          ("Date", "DC", "DC"),
                          ("Cloud9am", "NC", "N")])
def test_data_type(og_df, column, type1, type2):
    assert feature_types.data_type(og_df, column) == type1
    assert feature_types.data_type(og_df, column, 10) == type2


@pytest.mark.parametrize(["data_type", "list_of_type"],
                         [("C", C),
                          ("NC", NC),
                          ("N", []),
                          ("DC", ["Date"]),
                          ("D", [])])
def test_find_type(og_df, data_type, list_of_type):
    assert feature_types.find_type(og_df, data_type) == list_of_type


def test_labelled_df(label_df, modified_df):
    new_label_df = feature_types.labelled_df(modified_df)
    assert pd.testing.assert_frame_equal(label_df,
                                         new_label_df,
                                         check_dtype=False) is None


def test_cont_to_cat(contocat_df, modified_df):
    new_contocat_df = feature_types.cont_to_cat(modified_df, fillNA=True)
    assert pd.testing.assert_frame_equal(contocat_df,
                                         new_contocat_df,
                                         check_dtype=False) is None


def test_find_missing(modified_df):
    assert feature_types.find_missing(modified_df) == MISSING


def test_fill_missing(modified_df, filled_df):
    new_filled_df = feature_types.fill_missing(modified_df)
    assert pd.testing.assert_frame_equal(filled_df,
                                         new_filled_df,
                                         check_dtype=False) is None


@pytest.mark.parametrize(["val1", "val2", "psi"],
                         [("1. Before 2012",
                           "2. After 2012, Before 2015",
                           PSI1),
                          ("2. After 2012, Before 2015",
                           "3. After 2015",
                           PSI2)])
def test_psi(modified_df, val1, val2, psi):
    df = modified_df
    new_df = feature_types.cont_to_cat(df.drop("Date", axis=1), True)
    new_df["Date"] = df["Date"]
    assert list(feature_types.psi(new_df,
                                  "RainTomorrow",
                                  "Date",
                                  val1,
                                  val2).psi) == psi
