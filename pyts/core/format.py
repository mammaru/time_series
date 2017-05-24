#coding: utf-8
from pandas.core.format import DataFrameFormatter

docstring_to_string = """
    Parameters
    ----------
    frame : TimeSeries
        object to render
    buf : StringIO-like, optional
        buffer to write to
    columns : sequence, optional
        the subset of columns to write; default None writes all columns
    col_space : int, optional
        the minimum width of each column
    header : bool, optional
        whether to print column labels, default True
    index : bool, optional
        whether to print index (row) labels, default True
    na_rep : string, optional
        string representation of NAN to use, default 'NaN'
    formatters : list or dict of one-parameter functions, optional
        formatter functions to apply to columns' elements by position or name,
        default None. The result of each function must be a unicode string.
        List must be of length equal to the number of columns.
    float_format : one-parameter function, optional
        formatter function to apply to columns' elements if they are floats,
        default None. The result of this function must be a unicode string.
    sparsify : bool, optional
        Set to False for a DataFrame with a hierarchical index to print every
        multiindex key at each row, default True
    justify : {'left', 'right'}, default None
        Left or right-justify the column labels. If None uses the option from
        the print configuration (controlled by set_option), 'right' out
        of the box.
    index_names : bool, optional
        Prints the names of the indexes, default True
    force_unicode : bool, default False
        Always return a unicode result. Deprecated in v0.10.0 as string
        formatting is now rendered to unicode by default.

    Returns
    -------
    formatted : string (or unicode, depending on data and options)"""


class TimeSeriesFormatter(DataFrameFormatter):
    """
    Renderer of TimeSeries
    Overriding DataFrameFormatter

    self.to_string() : console-friendly tabular output
    self.to_html()   : html table
    self.to_latex()  : LaTeX tabular environment table

    """

    __doc__ = __doc__ if __doc__ else ''
    __doc__ += docstring_to_string

    def to_string(self):
        """
        Overriding DataFrameFormatter.to_string
        Render a TimeSeries to a console-friendly tabular output.
        """

        # render DataFrame into buf 
        super(TimeSeriesFormatter, self).to_string()

        additional_str = "\n" + \
                         self.frame.name + ': ' + \
                         "[%d timepoints x %d features]" \
                         % (len(self.frame.timepoints), len(self.frame.features))
        self.buf.write(additional_str)
