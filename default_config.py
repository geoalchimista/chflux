"""
PyChamberFlux default settings

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

"""
default_config = {
    'run_options': {  # Running options
        'chamber_config_filepath': './chamber.yaml',
        # Configuration file that describes chamber settings.

        'curve_fitting_method': 'all',
        # NOT IMPLEMENTED YET
        # Curve fitting method: 'nonlinear', 'linear', 'robust_linear', 'all'.

        'load_data_by_day': False,
        # Load and process raw data by daily chunks if `True`. Default is
        # `False` to load all data at once and then process daily chunks.
        # Note: If the total size of the raw data files are larger than
        # the size of computer memory, this should be enabled. Otherwise it
        # may take for ever in reading the data.

        'process_recent_period': False,
        # If True, process only recent few days' data. This will be useful for
        # online daily processing. If False, process all available data.

        'traceback_in_days': 1,
        # Number of days to trace back when processing recent periods
        # only used when 'process_recent_period' is True; must be int type

        'timelag_method': 'none',
        # NOT FULLY IMPLEMENTED YET
        # Timelag detection methods: 'none', 'optimized', 'prescribed'
        # For the 'optimized' method, timelag will be optimized based on
        # a nonlinear fitting of the concentration changes.
        # For the 'prescribed' method, fixed timelag values are assigned from
        # input data table (must enable 'use_timelag_data').

        'timelag_optimization_species': 'co2',
        # The gas species used for timelag optimization. 'co2' is recommended.
        # This option is only effective for timelag method 'optimized'.
        # If the designated species is not found, default to the first one in
        # the species list.

        'volume_correction': False,
        # NOT IMPLEMENTED YET
        # If True, optimize the effective volume (V_eff) to fit the curvature.

        'save_fitting_diagnostics': True,
        # If True, save fitting diagnostics to files.

        'save_config': False,
        # If True, save the configuration files in a subfolder 'config' in the
        # output directory.

        'save_fitting_plots': False,
        # If True, save the curve fitting plots for every chamber sampling
        # period.

        'save_daily_plots': False,
        # If True, save daily plots of chamber fluxes.

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

        'biomet_data.date_format': '%Y%m%d',
        # date format string in the file name (not that in the data table)

        'conc_data': './conc/*.csv',
        # Absolute or relative directory to search for concentration data
        # files.

        'conc_data.date_format': '%Y%m%d',
        # date format string in the file name (not that in the data table)

        'flow_data': None,
        # Absolute or relative directory to search for flow rate data files.

        'flow_data.date_format': '%Y%m%d',
        # date format string in the file name (not that in the data table)

        'leaf_data': None,
        # Absolute or relative directory to search for leaf area data files.

        'timelag_data': None,
        # NOT IMPLEMENTED YET
        # Absolute or relative directory to search for timelag data files.

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

        'use_timelag_data': False,
        # NOT IMPLEMENTED YET
        # Read external data for prescribed timelag values.
    },
    'biomet_data_settings': {  # Settings for reading the biomet data
        'delimiter': ',',
        # Supported table delimiters:
        #   - singe space: ' '
        #   - indefinite number of spaces: '\\s+' (works also for single space)
        #   - comma: ','
        #   - tab: '\\t'

        'header': 'infer',
        # Row number of the last line of the header (starting from 0)
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
        # Row number of the last line of the header (starting from 0)
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
        # Row number of the last line of the header (starting from 0)
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
        # Row number of the last line of the header (starting from 0)
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

        # Options for individual species
        #
        # `unit`: the unit of mixing ratio in the concentration data file
        # Some commonly used units:
        #     1.0 = mole fraction [0 to 1]
        #     1.0e-2 = percent (%)
        #     1.0e-3 = ppthv (parts per thousand) or mmol mol^-1
        #     1.0e-6 = ppmv or mumol mol^-1
        #     1.0e-9 = ppbv or nmol mol^-1
        #     1.0e-12 = pptv (parts per trillion) or pmol mol^-1
        #
        # `output_unit`: the unit of mixing ratio in the output file
        #
        # `multiplier`: the number to multiply to the input values for
        #     conversion to the output unit, must equal to
        #     `unit / output_unit`. For example, if H2O in the input data file
        #     was recorded in percentage (1.0e-2), and the output unit of H2O
        #     concentration needs to be parts per thousand (1.0e-3), then the
        #     multiplier would be 10.
        #
        # `baseline_correction`: str, baseline correction method
        #     'median': use medians as the baseline (default)
        #     'mean': use means as the baseline
        #     'none': do not apply baseline correction
        # Note: 'median' baseline setting is generally recommended, except for
        # water and other adsorptive molecules. Depending on the ambient
        # condition, sometimes 'none' works better for water flux calculation.

        'h2o': {
            'unit': 1.0e-3,
            'output_unit': 1.0e-3,
            'multiplier': 1.0,
            'baseline_correction': 'median',
        },

        'co2': {
            'unit': 1.0e-6,
            'output_unit': 1.0e-6,
            'multiplier': 1.0,
            'baseline_correction': 'median',
        },

        'cos': {
            'unit': 1.0e-9,
            'output_unit': 1.0e-12,
            'multiplier': 1.0e+3,
            'baseline_correction': 'median',
        },

        'co': {
            'unit': 1.0e-9,
            'output_unit': 1.0e-9,
            'multiplier': 1.0,
            'baseline_correction': 'median',
        },

        'ch4': {
            'unit': 1.0e-9,
            'output_unit': 1.0e-9,
            'multiplier': 1.0,
            'baseline_correction': 'median',
        },

        'n2o': {
            'unit': 1.0e-9,
            'output_unit': 1.0e-9,
            'multiplier': 1.0,
            'baseline_correction': 'median',
        },

        # Customize your own gas species following the same format.
        # The name that represents the added gas species is not so important
        # as long as it is used *consistently*. For example, if the species
        # name for CO2 is defined as `CO_2`, the same name `CO_2` must be used
        # in the `species_list` key and in the following unit definition.
    },
}
