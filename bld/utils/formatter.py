import numpy as np
import pandas as pd


class Formatter:
    """
    Format the distance table with colorings, etc.
    """

    @staticmethod
    def color_red_font_minimum_in_a_column(column):
        """
        Assign red color to the minimum value of the column.
        """
        highlight = 'color: red;'
        default = ''
        minimum_in_column = column.min()
        return [highlight if e == minimum_in_column else default for e in column]

    @staticmethod
    def color_green_minimum_value_in_row(row):
        """
        Assign green color to the minimum value of the row.
        """
        highlight = 'background-color: green;'
        default = ''
        minimum_in_row = row.min()
        # must return one string per cell in this column
        return [highlight if v == minimum_in_row else default for v in row]

    @staticmethod
    def rearrange_table(df: pd.DataFrame):
        """
        Rearrange the table based on the FMinD values.
        The closest point to Pref0 will be in the first column, the 2nd closest will be in the second, etc.
        """
  
        row_min_indices = np.argmin(df.values, axis=1)
        column_min_indices = np.argmin(df.values, axis=0)

        df_rearranged = df[[df.columns[i] for i in row_min_indices]]

        diff_row = np.diff(row_min_indices)
        diff_column = np.diff(column_min_indices)

        return df_rearranged, diff_row, diff_column
