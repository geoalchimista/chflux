"""
PyChamberFlux default settings

(c) Wu Sun <wu.sun@ucla.edu> 2016-2017

"""
default_config = {
    'run_options': {  # Running options
        'chamber_config_filepath': './chamber.yaml',
        # Configuration file that describes chamber settings.

        'baseline_correction': True,
        # If True, a zero-flux baseline correction is applied.

        'median_baseline': True,
        # If True, use medians as the baseline.
        # If False, use means as the baseline.

        'timelag_method': 'nominal',
        # Timelag detection methods: 'nominal', 'manual' or 'optimized'.
        # For the 'manual' method, a fixed timelag must be assigned.
        # For the 'optimized' method, timelag will be optimized based on
        # a nonlinear fitting of the concentration changes.

        'timelag_optimization_species': 'co2',

        'volume_correction': False,
        # If True, optimize the effective volume (V_eff) to fit the curvature.

        'curve_fitting_method': 'all',
        # Curve fitting method: 'nonlinear', 'linear', 'robust_linear', 'all'.

        'process_recent_period': False,
        # If True, process only recent few days' data. This will be useful for
        # online daily processing. If False, process all available data.

        'traceback_in_days': 1,
        # number of days to trace back when processing recent periods
        # only used when 'process_recent_period' is True; must be int type

        'save_fitting_plots': False,
        # If True, save the curve fitting plots for every chamber sampling
        # period.

        'save_daily_plots': False,
        # If True, save daily plots of chamber fluxes and biomet variables.

        'plot_style': 'ggplot',
        # plot style for matplotlib; default is 'ggplot'
        # allowed parameters are:
        # 'bmh', 'classic', 'ggplot', 'fivethirtyeight', 'dark_background',
        # 'presentation', seaborn series, etc.
        # see matplotlib guide on plotting style:
        # <http://matplotlib.org/users/style_sheets.html#style-sheets>
    },
    'data_dir': {  # Input and output directories, and other related settings
        'biomet_data': './biomet/*.csv',
        # Absolute or relative directory to search for biomet data files.

        'conc_data': './conc/*.csv',
        # Absolute or relative directory to search for concentration data
        # files.

        'flow_data': None,
        # Absolute or relative directory to search for flow rate data files.

        'leaf_data': None,
        # Absolute or relative directory to search for leaf area data files.

        'output_dir': './output/',
        # Output directory for the processed flux data.

        'output_filename_prefix': '',
        # A prefix string to append before timestamp for output datafile names.

        'plot_dir': './plots/',
        # Directory for saved plots.

        'separate_conc_data': True,
        # If `True`, concentration measurements are stored on their own, not in
        # the biomet data files.
        # If `False`, search concentration measurements in the biomet data.

        'separate_flow_data': False,
        # If `False` (default), search flow rate variables in the biomet data.
        # If `True`, flow rate measurements are stored on their own, not in
        # the biomet data files (possibly relevant for some MFC instruments).

        'separate_leaf_data': False,
        # If `False` (default), leaf area values are defined with `A_ch` in
        # the chamber schedule configuration.
        # If `True`, leaf area measurements are stored on their own, not in
        # the chamber schedule configuration file.
    },
    'biomet_data_settings': {  # Settings for reading the biomet data
        'delimiter': ',',
        # Supported table delimiters:
        #   - singe space: ' '
        #   - indefinite number of spaces: '\\s+' (works also for single space)
        #   - comma: ','
        #   - tab: '\\t'

        'header': 'infer',
        # Number of rows to skip at the beginning of the file.
        # Default behavior is to infer it with `pandas.read_csv()`.

        'names': None,
        # Define the data table column names.
        # Default is `None`, i.e., to infer with `pandas.read_csv()`.
        # Tip: copy the column names from the data file, and then change names
        # of the variables of interest to the standardized names.

        'usecols': None,
        # Specify a sequence of indices for columns to read into the data
        # structure. Column index starts from 0 in Python.
        # Default behavior (`None`) is to read all columns.

        'dtype': None,
        # If `None`, falls back to the default setting of `pandas.read_csv()`.
        # Its default settings handle data types pretty well without
        # specification. You can also modify the line above to customize column
        # data types. For example, dtype: 'f8, f8, f8, i8' indicates that the
        # first 3 columns are (double) floating numbers and the last column is
        # of integer type.

        'na_values': None,
        # Modify this if you need specify the missing values.
        # Default is `None` that uses the default options of
        # `pandas.read_csv()`.

        'parse_dates': False,
        # if False, do not attempt to parse dates with `pandas.read_csv()`
        # if given a list of column indices or names, parse those columns as
        # dates when parse multiple columns to form a datetime variable, must
        # specify a column name for the parsed result

        'date_parser': None,
        # a date parser for converting date strings stored in multiple columns
        # - 'ymd', YYYY MM DD, date only
        # - 'ymdhm', YYYY MM DD HH MM, down to minute
        # - 'ymdhms', YYYY MM DD HH MM SS, down to second
        # - 'ymdhmsf', YYYY MM DD HH MM SS %f, down to nanosecond
        # note: no need to use this if the date string is in a single column

        'columns_to_save': None,
        # If `None`, save all columns of the 'standardized variables' parsed
        # from the biomet data table

        'time_sec_start': None,
        # If None, the starting year of the time_sec format is 1904 (LabVIEW).
        # This option only takes the year number in four digits (integer).
        # For example, an instrument working on Unix system may give time in
        # seconds since 1 Jan 1970 00:00, then this option must be set to 1970.
        # Check your instrument manual for its specific time format, if it
        # records time in seconds since a date.

        'year_ref': None,
        # must specify a reference year if the time variable is day of year
        # number

        'time_in_UTC': True,
        # Default is `True` to treat the time variable in UTC time.
    },
    'conc_data_settings': {  # Settings for reading the concentration data
        'delimiter': ',',
        # Supported table delimiters:
        #   - singe space: ' '
        #   - indefinite number of spaces: '\\s+' (works also for single space)
        #   - comma: ','
        #   - tab: '\\t'

        'header': 'infer',
        # Number of rows to skip at the beginning of the file.
        # Default behavior is to infer it with `pandas.read_csv()`.

        'names': None,
        # Define the data table column names.
        # Default is `None`, i.e., to infer with `pandas.read_csv()`.
        # Note that for concentration data table, gas species that are not
        # defined in the species settings will be ignored.

        'usecols': None,
        # Specify a sequence of indices for columns to read into the data
        # structure. Column index starts from 0 in Python.
        # Default behavior (`None`) is to read all columns.

        'dtype': None,
        # If `None`, falls back to the default setting of `pandas.read_csv()`.
        # Its default settings handle data types pretty well without
        # specification. You can also modify the line above to customize column
        # data types. For example, dtype: 'f8, f8, f8, i8' indicates that the
        # first 3 columns are (double) floating numbers and the last column is
        # of integer type.

        'na_values': None,
        # Modify this if you need specify the missing values.

        'parse_dates': False,
        # if False, do not attempt to parse dates with `pandas.read_csv()`
        # if given a list of column indices or names, parse those columns as
        # dates when parse multiple columns to form a datetime variable, must
        # specify a column name for the parsed result

        'date_parser': None,
        # a date parser for converting date strings stored in multiple columns
        # - 'ymd', YYYY MM DD, date only
        # - 'ymdhm', YYYY MM DD HH MM, down to minute
        # - 'ymdhms', YYYY MM DD HH MM SS, down to second
        # - 'ymdhmsf', YYYY MM DD HH MM SS %f, down to nanosecond
        # note: no need to use this if the date string is in a single column

        'columns_to_save': None,
        # If `None`, save all columns of the 'standardized variables' parsed
        # from the biomet data table

        'time_sec_start': None,
        # If None, the starting year of the time_sec format is 1904 (LabVIEW).
        # This option only takes the year number in four digits (integer).
        # For example, an instrument working on Unix system may give time in
        # seconds since 1 Jan 1970 00:00, then this option must be set to 1970.
        # Check your instrument manual for its specific time format, if it
        # records time in seconds since a date.

        'year_ref': None,
        # must specify a reference year if the time variable is day of year
        # number

        'time_in_UTC': True,
        # Default is `True` to treat the time variable in UTC time.
    },
    'flow_data_settings': {  # Settings for reading the flow rate data
        'delimiter': ',',
        # Supported table delimiters:
        #   - singe space: ' '
        #   - indefinite number of spaces: '\\s+' (works also for single space)
        #   - comma: ','
        #   - tab: '\\t'

        'header': 'infer',
        # Number of rows to skip at the beginning of the file.
        # Default behavior is to infer it with `pandas.read_csv()`.

        'names': None,
        # Define the data table column names.
        # Default is `None`, i.e., to infer with `pandas.read_csv()`.
        # Tip: copy the column names from the data file, and then change names
        # of the variables of interest to the standardized names.

        'usecols': None,
        # Specify a sequence of indices for columns to read into the data
        # structure. Column index starts from 0 in Python.
        # Default behavior (`None`) is to read all columns.

        'dtype': None,
        # If `None`, falls back to the default setting of `pandas.read_csv()`.
        # Its default settings handle data types pretty well without
        # specification. You can also modify the line above to customize column
        # data types. For example, dtype: 'f8, f8, f8, i8' indicates that the
        # first 3 columns are (double) floating numbers and the last column is
        # of integer type.

        'na_values': None,
        # Modify this if you need specify the missing values.

        'parse_dates': False,
        # if False, do not attempt to parse dates with `pandas.read_csv()`
        # if given a list of column indices or names, parse those columns as
        # dates when parse multiple columns to form a datetime variable, must
        # specify a column name for the parsed result

        'date_parser': None,
        # a date parser for converting date strings stored in multiple columns
        # - 'ymd', YYYY MM DD, date only
        # - 'ymdhm', YYYY MM DD HH MM, down to minute
        # - 'ymdhms', YYYY MM DD HH MM SS, down to second
        # - 'ymdhmsf', YYYY MM DD HH MM SS %f, down to nanosecond
        # note: no need to use this if the date string is in a single column

        'columns_to_save': None,
        # If `None`, save all columns of the 'standardized variables' parsed
        # from the biomet data table

        'time_sec_start': None,
        # If None, the starting year of the time_sec format is 1904 (LabVIEW).
        # This option only takes the year number in four digits (integer).
        # For example, an instrument working on Unix system may give time in
        # seconds since 1 Jan 1970 00:00, then this option must be set to 1970.
        # Check your instrument manual for its specific time format, if it
        # records time in seconds since a date.

        'year_ref': None,
        # must specify a reference year if the time variable is day of year
        # number

        'time_in_UTC': True,
        # Default is `True` to treat the time variable in UTC time.

        'flow_rate_in_STP': True,
        # Default is `True`. The flow rate is referenced to STP condition,
        # as for many commonly available flow sensors. This means that the
        # flow rates need to be corrected for ambient temperature and pressure.
        # If set to False, no such correction on flow rates is applied.
    },
    'leaf_data_settings': {  # Settings for reading the leaf area data
        # must contain a timestamp variable;
        # leaf area variables must be labeled with chamber labels that
        # correspond to those defined in `ch_label` in the chamber schedule
        # configuration

        'delimiter': ',',
        # Supported table delimiters:
        #   - singe space: ' '
        #   - indefinite number of spaces: '\\s+' (works also for single space)
        #   - comma: ','
        #   - tab: '\\t'

        'header': 'infer',
        # Number of rows to skip at the beginning of the file.
        # Default behavior is to infer it with `pandas.read_csv()`.

        'names': None,
        # Define the data table column names.
        # Default is `None`, i.e., to infer with `pandas.read_csv()`.
        # Tip: copy the column names from the data file, and then change names
        # of the variables of interest to the standardized names.

        'usecols': None,
        # Specify a sequence of indices for columns to read into the data
        # structure. Column index starts from 0 in Python.
        # Default behavior (`None`) is to read all columns.

        'dtype': None,
        # If `None`, falls back to the default setting of `pandas.read_csv()`.
        # Its default settings handle data types pretty well without
        # specification. You can also modify the line above to customize column
        # data types. For example, dtype: 'f8, f8, f8, i8' indicates that the
        # first 3 columns are (double) floating numbers and the last column is
        # of integer type.

        'na_values': None,
        # Modify this if you need specify the missing values.

        'parse_dates': False,
        # if False, do not attempt to parse dates with `pandas.read_csv()`
        # if given a list of column indices or names, parse those columns as
        # dates when parse multiple columns to form a datetime variable, must
        # specify a column name for the parsed result

        'date_parser': None,
        # a date parser for converting date strings stored in multiple columns
        # - 'ymd', YYYY MM DD, date only
        # - 'ymdhm', YYYY MM DD HH MM, down to minute
        # - 'ymdhms', YYYY MM DD HH MM SS, down to second
        # - 'ymdhmsf', YYYY MM DD HH MM SS %f, down to nanosecond
        # note: no need to use this if the date string is in a single column

        'columns_to_save': None,
        # If `None`, save all columns of the 'standardized variables' parsed
        # from the biomet data table

        'time_sec_start': None,
        # If None, the starting year of the time_sec format is 1904 (LabVIEW).
        # This option only takes the year number in four digits (integer).
        # For example, an instrument working on Unix system may give time in
        # seconds since 1 Jan 1970 00:00, then this option must be set to 1970.
        # Check your instrument manual for its specific time format, if it
        # records time in seconds since a date.

        'year_ref': None,
        # must specify a reference year if the time variable is day of year
        # number

        'time_in_UTC': True,
        # Default is `True` to treat the time variable in UTC time.
    },
    'site_parameters': {  # Site-specific parameters
        'site_pressure': None,
        # In Pascal. Default behavior (`None`) is to use the standard pressure.

        'time_zone': 0,
        # Time zone with respect to UTC. For example, -8 means UTC-8.
        # Warning: does not support daylight saving transition. Use standard
        # non daylight saving time, or process separately the data before
        # and after daylight saving.
    },
    'species_settings': {
        'species_list': ['co2', 'h2o'],
        # Measured gas species in the concentration data.
        # Note: the order of gas species in the output file will follow
        # the order defined in this sequence

        'species_names': ['CO$_2$', 'H$_2$O'],
        # names of gas species shown in the plot axis labels.
        # LaTeX format is supported by matplotlib.

        # `unit`: the unit of mixing ratio in the concentration data file
        # `output_unit`: the unit of mixing ratio in the output file
        # `multiplier`: the number to multiply to the input values for
        # conversion to the output unit, must equal to `unit / output_unit`.
        # For example, if H2O in the input data file was recorded in percentage
        # (1.0e-2), and the output unit of H2O concentration needs to be parts
        # per thousand (1.0e-3), then the multiplier would be 10.
        # Some commonly used units:
        #     1.0 = mole fraction [0 to 1]
        #     1.0e-2 = percent (%)
        #     1.0e-3 = ppthv (parts per thousand) or mmol mol^-1
        #     1.0e-6 = ppmv or mumol mol^-1
        #     1.0e-9 = ppbv or nmol mol^-1
        #     1.0e-12 = pptv (parts per trillion) or pmol mol^-1

        'h2o': {
            'unit': 1.0e-9,
            'output_unit': 1.0e-3,
            'multiplier': 1.0e-6,
        },

        'co2': {
            'unit': 1.0e-9,
            'output_unit': 1.0e-6,
            'multiplier': 1.0e-3,
        },

        'cos': {
            'unit': 1.0e-9,
            'output_unit': 1.0e-12,
            'multiplier': 1.0e+3,
        },

        'co': {
            'unit': 1.0e-9,
            'output_unit': 1.0e-9,
            'multiplier': 1.0,
        },

        'ch4': {
            'unit': 1.0e-9,
            'output_unit': 1.0e-9,
            'multiplier': 1.0,
        },

        # You may add your own gas species following the same format
        # The name that represents the added gas species is not so important
        # as long as it is used *consistently*. For example, if you define the
        # species name for CO2 to be `CO_2`, you must use the same name `CO_2`
        # in the `species_list` key and the following unit definition.
    },
}
