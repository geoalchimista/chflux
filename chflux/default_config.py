"""PyChamberFlux default settings"""

default_config = {
    # == General options for the run ===

    # Configuration file that describes chamber schedules.
    'run.chamber_schedule': './chamber.yaml',

    # If True, perform realtime processing of the incoming data. Data files to
    # be processed will be limited to a recent period, with the number of days
    # to be traced back determined by 'run.latency'.
    'run.realtime': False,

    # Number of days to be traced back from the current day when performing
    # realtime processing. Must be a positive integer. Will not have effect if
    # 'run.realtime' is set False.
    'run.latency': 1,

    # Output directory for the processed flux data.
    'run.output.path': './output/',

    # A prefix string to prepend filenames of output data.
    'run.output.prefix': '',

    # If True, save curve fitting diagnostics to files in subfolder './diag'
    # in the output directory.
    'run.save_curvefit_diagnostics': True,

    # If True, save the configuration files in subfolder './config' in the
    # output directory.
    'run.save_config': True,

    # == Plotting settings ==

    # Plot style for matplotlib; default is 'seaborn-darkgrid'.
    # Available styles: 'bmh', 'classic', 'ggplot', 'fivethirtyeight',
    # 'dark_background', 'presentation', seaborn series, etc. See matplotlib
    # guide on styles:
    # <https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html>
    'plot.style': 'seaborn-darkgrid',

    # Directory to save plots to.
    'plot.path': './plots/',

    # If True, save the curve fitting plots for each chamber sampling period.
    'plot.save_curvefit': False,

    # If True, save daily summary of calculated fluxes.
    'plot.save_daily': False,

    # == Biometeorological data settings ==

    # Absolute or relative directory of biomet data files.
    'biomet.files': './biomet/*.csv',

    # Date format in biomet data filenames.
    'biomet.date_format': '%Y%m%d',

    # Options for reading biomet data CSV files with `pandas.read_csv`.
    'biomet.csv.delimiter': ',',
    'biomet.csv.header': 'infer',
    'biomet.csv.names': None,
    'biomet.csv.usecols': None,
    'biomet.csv.dtype': None,
    'biomet.csv.na_values': None,
    'biomet.csv.parse_dates': False,

    # Timestamp parser for converting date strings stored in multiple columns.
    # (Note: no need to use the parser if the timestamp is in one column.)
    # * 'ymd', YYYY MM DD, date only
    # * 'ymdhm', YYYY MM DD HH MM, down to minute
    # * 'ymdhms', YYYY MM DD HH MM SS, down to second
    # * 'ymdhmsf', YYYY MM DD HH MM SS %f, down to nanosecond
    'biomet.timestamp.parser': None,

    # Epoch year for UNIX timestamps (in seconds). Default (None) is 1970.
    # However, some computer system may use a different epoch year, for
    # example, 1904 for LabVIEW.
    'biomet.timestamp.epoch_year': None,

    # Reference year for day-of-year timestamps.
    'biomet.timestamp.day_of_year_ref': None,

    # If True, the timestamp is treated as in UTC.
    'biomet.timestamp.is_UTC': True,

    # == Concentration data settings ==

    # Absolute or relative directory of concentration data files. If None, find
    # concentration data in the biomet datatable.
    'concentration.files': './conc/*.csv',

    # Date format in concentration data filenames.
    'concentration.date_format': '%Y%m%d',

    # Options for reading concentration data CSV files with `pandas.read_csv`.
    'concentration.csv.delimiter': ',',
    'concentration.csv.header': 'infer',
    'concentration.csv.names': None,
    'concentration.csv.usecols': None,
    'concentration.csv.dtype': None,
    'concentration.csv.na_values': None,
    'concentration.csv.parse_dates': False,

    # Timestamp parser for converting date strings stored in multiple columns.
    # (Note: no need to use the parser if the timestamp is in one column.)
    # * 'ymd', YYYY MM DD, date only
    # * 'ymdhm', YYYY MM DD HH MM, down to minute
    # * 'ymdhms', YYYY MM DD HH MM SS, down to second
    # * 'ymdhmsf', YYYY MM DD HH MM SS %f, down to nanosecond
    'concentration.timestamp.parser': None,

    # Epoch year for UNIX timestamps (in seconds). Default (None) is 1970.
    # However, some computer system may use a different epoch year, for
    # example, 1904 for LabVIEW.
    'concentration.timestamp.epoch_year': None,

    # Reference year for day-of-year timestamps.
    'concentration.timestamp.day_of_year_ref': None,

    # If True, the timestamp is treated as in UTC.
    'concentration.timestamp.is_UTC': True,

    # == Chamber flowrate data settings ==

    # Absolute or relative directory of flowrate data files. If None, find
    # flowrate data in the biomet datatable.
    'flow.files': None,

    # Date format in flowrate data filenames.
    'flow.date_format': '%Y%m%d',

    # If True (default), flowrate is referenced to the STP condition, as is the
    # convention used by many commercially available flow sensors. This means
    # that measured flowrates need to be corrected for ambient temperature and
    # pressure conditions. If False, no correction on flow rates is applied.
    'flow.is_STP': True,

    # Options for reading flowrate data CSV files with `pandas.read_csv`.
    'flow.csv.delimiter': ',',
    'flow.csv.header': 'infer',
    'flow.csv.names': None,
    'flow.csv.usecols': None,
    'flow.csv.dtype': None,
    'flow.csv.na_values': None,
    'flow.csv.parse_dates': False,

    # Timestamp parser for converting date strings stored in multiple columns.
    # (Note: no need to use the parser if the timestamp is in one column.)
    # * 'ymd', YYYY MM DD, date only
    # * 'ymdhm', YYYY MM DD HH MM, down to minute
    # * 'ymdhms', YYYY MM DD HH MM SS, down to second
    # * 'ymdhmsf', YYYY MM DD HH MM SS %f, down to nanosecond
    'flow.timestamp.parser': None,

    # Epoch year for UNIX timestamps (in seconds). Default (None) is 1970.
    # However, some computer system may use a different epoch year, for
    # example, 1904 for LabVIEW.
    'flow.timestamp.epoch_year': None,

    # Reference year for day-of-year timestamps.
    'flow.timestamp.day_of_year_ref': None,

    # If True, the timestamp is treated as in UTC.
    'flow.timestamp.is_UTC': True,

    # == Leaf data settings ==

    # Absolute or relative directory of leaf data files. If None, use leaf area
    # values supplied in the chamber schedule files.
    'leaf.files': None,

    # Date format in leaf data filenames.
    'leaf.date_format': '%Y%m%d',

    # Options for reading leaf data CSV files with `pandas.read_csv`.
    'leaf.csv.delimiter': ',',
    'leaf.csv.header': 'infer',
    'leaf.csv.names': None,
    'leaf.csv.usecols': None,
    'leaf.csv.dtype': None,
    'leaf.csv.na_values': None,
    'leaf.csv.parse_dates': False,

    # Timestamp parser for converting date strings stored in multiple columns.
    # (Note: no need to use the parser if the timestamp is in one column.)
    # * 'ymd', YYYY MM DD, date only
    # * 'ymdhm', YYYY MM DD HH MM, down to minute
    # * 'ymdhms', YYYY MM DD HH MM SS, down to second
    # * 'ymdhmsf', YYYY MM DD HH MM SS %f, down to nanosecond
    'leaf.timestamp.parser': None,

    # Epoch year for UNIX timestamps (in seconds). Default (None) is 1970.
    # However, some computer system may use a different epoch year, for
    # example, 1904 for LabVIEW.
    'leaf.timestamp.epoch_year': None,

    # Reference year for day-of-year timestamps.
    'leaf.timestamp.day_of_year_ref': None,

    # If True, the timestamp is treated as in UTC.
    'leaf.timestamp.is_utc': True,

    # == Timelag data settings ==

    # Absolute or relative directory of timelag data files.
    'timelag.files': None,

    # Date format in timelag data filenames.
    'timelag.date_format': '%Y%m%d',

    # Options for reading timelag data CSV files with `pandas.read_csv`.
    'timelag.csv.delimiter': ',',
    'timelag.csv.header': 'infer',
    'timelag.csv.names': None,
    'timelag.csv.usecols': None,
    'timelag.csv.dtype': None,
    'timelag.csv.na_values': None,
    'timelag.csv.parse_dates': False,

    # Timestamp parser for converting date strings stored in multiple columns.
    # (Note: no need to use the parser if the timestamp is in one column.)
    # * 'ymd', YYYY MM DD, date only
    # * 'ymdhm', YYYY MM DD HH MM, down to minute
    # * 'ymdhms', YYYY MM DD HH MM SS, down to second
    # * 'ymdhmsf', YYYY MM DD HH MM SS %f, down to nanosecond
    'timelag.timestamp.parser': None,

    # Epoch year for UNIX timestamps (in seconds). Default (None) is 1970.
    # However, some computer system may use a different epoch year, for
    # example, 1904 for LabVIEW.
    'timelag.timestamp.epoch_year': None,

    # Reference year for day-of-year timestamps.
    'timelag.timestamp.day_of_year_ref': None,

    # If True, the timestamp is treated as in UTC.
    'timelag.timestamp.is_utc': True,

    # == Flux calculator settings ==

    # Enable linear curve fitting method for flux calculation.
    'flux.curvefit.linear': True,

    # Enable robust linear curve fitting method for flux calculation.
    'flux.curvefit.robust_linear': True,

    # Enable nonlinear curve fitting method for flux calculation.
    'flux.curvefit.nonlinear': True,
    # Note: at least one of the curve fitting methods must be enabled.

    # Timelag optimization methods
    # * 'none': No timelag optimization.
    # * 'optimized': Timelags are optimized based on a nonlinear fitting of the
    #   concentration changes.
    # * 'prescribed': Fixed values of timelag are assigned from input data.
    #   Must supply timelag data files in 'timelag.files'.
    'flux.timelag_method': 'none',

    # Gas species used for timelag optimization (only effective for timelag
    # method 'optimized'). 'co2' is recommended. If the designated species is
    # not found, use the first one in 'species.list' by default.
    'flux.timelag_species': 'co2',

    # == Field site parameters ==

    # Surface pressure (Pa) at the site. Default (`None`) is to use the
    # standard atmospheric pressure (101,325 Pa).
    'site.pressure': None,

    # Time zone with respect to UTC. For example, -8 means UTC-8 (aka PST).
    # Warning: does not support daylight saving transition. You may use the
    # standard time consistently, or process separately data files before and
    # after the daylight saving change.
    'site.timezone': 0,

    # == Gas species settings ==

    # Measured gas species -- only those listed here will be calculated from
    # the concentration data. Do not use special characters, for these names
    # will be in the header of the output datatables.
    'species.list': ['co2', 'h2o'],

    # Names of gas species shown in the plot labels. May use LaTeX format.
    'species.names': ['CO$_2$', 'H$_2$O'],

    # Options for individual species
    #
    # `unit`: the unit of mixing ratio in the concentration data `output_unit`:
    # the unit of mixing ratio in the output file
    #
    # Some commonly used units:
    # * 1.0 = mole fraction [0 to 1]
    # * 1.0e-3 = ppthv (parts per thousand) or mmol mol^-1
    # * 1.0e-6 = ppmv or mumol mol^-1
    # * 1.0e-9 = ppbv or nmol mol^-1
    # * 1.0e-12 = pptv (parts per trillion) or pmol mol^-1
    # * Note: percentage (% or 1.0e-2) is not allowed in the output unit
    #
    # `multiplier`: the number to multiply to the input values for conversion
    # to the output unit; must equal to `unit / output_unit`. For example, if
    # H2O in the input data file is in percentage (1.0e-2), and the output H2O
    # concentration needs to be in parts per thousand (1.0e-3), then the
    # multiplier is 10.
    #
    # `baseline_correction`: str, baseline correction method
    # * 'median': use medians as the baseline (default)
    # * 'mean': use means as the baseline
    # * 'none': do not apply baseline correction
    #
    # Note: 'median' baseline setting is generally recommended, except for
    # water and other adsorptive molecules. Depending on the ambient condition,
    # sometimes 'none' works better for water flux calculation.

    'species.h2o': {
        'unit': 1.0e-3,
        'output_unit': 1.0e-3,
        'multiplier': 1.0,
        'baseline_correction': 'median',
    },

    'species.co2': {
        'unit': 1.0e-6,
        'output_unit': 1.0e-6,
        'multiplier': 1.0,
        'baseline_correction': 'median',
    },

    'species.cos': {
        'unit': 1.0e-9,
        'output_unit': 1.0e-12,
        'multiplier': 1.0e+3,
        'baseline_correction': 'median',
    },

    'species.co': {
        'unit': 1.0e-9,
        'output_unit': 1.0e-9,
        'multiplier': 1.0,
        'baseline_correction': 'median',
    },

    'species.ch4': {
        'unit': 1.0e-9,
        'output_unit': 1.0e-9,
        'multiplier': 1.0,
        'baseline_correction': 'median',
    },

    'species.n2o': {
        'unit': 1.0e-9,
        'output_unit': 1.0e-9,
        'multiplier': 1.0,
        'baseline_correction': 'median',
    },

    # You may customize a gas species following the same format. The name
    # representing the added gas species is not so important as long as it is
    # used *consistently*. For example, if the species name for CO2 is defined
    # as 'CO_2', the same name 'CO_2' must be used in the 'species.list' key
    # and in the species definition.
}
