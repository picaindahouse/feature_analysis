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
           'MaxTemp',
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

PSI1 = [0.00907378152623754,
        0.004064174171412495,
        0.003247139573179995,
        0.002680136499003653,
        0.026862179718963354,
        0.04607915340467834,
        0.014171795015291625,
        0.01394898515980616,
        0.0020763702997892964,
        0.00282539100175955,
        0.003753198231597682,
        0.0036186922789531078,
        0.007195500798899057,
        0.010558976712869032,
        0.0059312448468967095,
        0.0034692317116453856,
        0.021060079557903746,
        0.01988207912758279,
        0.002159499575586343,
        0.0040707125579862394,
        0.0029000080606483166]

PSI2 = [0.014696120427005202,
        0.013784677549376362,
        0.01075722073591836,
        0.0004411395221993698,
        0.08531172938985891,
        0.10942349275067836,
        0.002531344733554075,
        0.0026956800524366212,
        0.005732865093223232,
        0.021294171482669735,
        0.0037939972091040833,
        0.038483951717287065,
        0.004225077297072558,
        0.0616243571840802,
        0.003964634646964244,
        0.0050847333735358465,
        0.03193101071031576,
        0.0489247022301791,
        0.01154319729224227,
        0.10721353041886722,
        0.0005935084511908906]


@pytest.mark.parametrize(["column", "type1", "type2"],
                         [("Location", "C", "C"),
                          ("MinTemp", "NC", "NC"),
                          ("WindSpeed9am", "NC", "NC"),
                          ("Date", "DC", "DC"),
                          ("Cloud9am", "NC", "N")])
def test_data_type(og_df, column, type1, type2):
    assert feature_types.data_type(og_df, column) == type1
    assert feature_types.data_type(og_df, column, 10) == type2


@pytest.mark.parametrize(["type", "list"],
                         [("C", C),
                          ("NC", NC),
                          ("N", []),
                          ("DC", ["Date"]),
                          ("D", [])])
def test_find_type(og_df, type, list):
    assert feature_types.find_type(og_df, type) == list


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
