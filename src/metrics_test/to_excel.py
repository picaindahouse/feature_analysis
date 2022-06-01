import pandas as pd

from final import overall


def metrics_excel(df, label, date=[], filename='report.xlsx', bad_class=1, on_bin=False, strat=None, bins=5):
    """
    Saves an excelsheet with all the dataframes available
    in overall having 1 worksheet to themselves.

    Parameters
    ----------
    df : Dataframe
         The dataset whose features we are evaluating

    label : String
            Column containing binary output

    date : String, optional
           Column containing datetime object

    filename: String, optional
              Name of excelsheet saved

    bad_class : int, optional
                Which int is the "bad class"

    on_bin : Boolean, optional
             Whether to find AUC/KS score on
             binned or continuous values of the
             continuous features in df

    strat : String, optional
            A strategy that can be used by SimpleImputer
            from sklearn.impute

    bins : int, optional
          Maximum number of bins each feature should have

    Saves
    -------
    Xslx : xslx file
           Excel file containing all the dataframes from
           overall.
    """
    final = overall(df,
                    label,
                    date=date,
                    bad_class=bad_class,
                    on_bin=on_bin,
                    strat=strat,
                    bins=bins)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    if date == []:
        dfs = {"Performance": final(1),
               "Trend": final(2)}
    else:
        dfs = {"Performance": final(1),
               "Trend": final(2),
               "Stability": final(7),
               "Total": final(3),
               "Ratio": final(4),
               "Bad": final(5),
               "Bad Rate": final(6)}

    for sheetname, df_n in dfs.items():
        df_n.to_excel(writer, sheet_name=sheetname, index=False)
        worksheet = writer.sheets[sheetname]
        for idx, col in enumerate(df_n):
            series = df_n[col]
            max_len = max((
                series.astype(str).map(len).max(),
                len(str(series.name))
                )) + 1
            worksheet.set_column(idx, idx, max_len)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()