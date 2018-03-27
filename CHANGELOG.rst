=========
Changelog
=========
This file documents all notable changes to the project PyChamberFlux.

The format of this changelog is based on `Keep a Changelog <http://keepachangelog.com/en/1.0.0/>`_.
This project adheres to the `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.
Changes are described in the past tense and the effects of changes are
described in the imperative mood.


Unreleased - 2018-03-27
=======================
Refactoring the main program in progress:

* ``chflux/chflux.py``
* ``chflux/io/parsers.py``
* ``chflux/io/readers.py``
* @TODO: ``iotools.load_tabulated_data`` -> [X] ``io.readers.read_tabulated_data``, [ ] ``io.parsers.parse_timestamp``
* @TODO: rename ``iotools`` to ``_iotools`` for deprecation.
* @TODO: test ``parsers`` and ``readers`` in the new ``chflux`` main program

@TODO:

* ``DataFrame.set_value()`` is deprecated; use ``.at[]`` instead.
* Use double spaces at sentence ending in comments and docstrings; for better readability in a monospace typeface.
    - [X] ``io/readers.py``
    - [ ] ``io/parsers.py``
* Simplify function arguments using OOP.
* Replace the Theil--Sen estimator with a more efficient robust linear
  regression method (perhaps an MM-estimator?).
* Use ``pandas.Timestamp`` as the standard timestamp passed between functions.

Added
-----
* Physical chemistry module (``chflux.physchem``)
  - Added ``convert_flowrate`` function to convert flow rate from STP to
    ambient conditions.

Changed
-------
* Physical chemistry module (``chflux.physchem``)
  - Simplified the saturation vapor pressure (``e_sat``) and the dew
    temperature (``dew_temp``) functions.
* Documentation files reformatted to reStructuredText.


0.1.13.a - 2018-02-17
=====================
Fixed
-----
* The file list returned by ``glob.glob()`` is not ordered by name. The bug has
  been fixed using ``sorted()``.


0.1.13 - 2017-11-29
===================
Changed
-------
* Standardized the style of the ``changelog.md`` file.

Fixed
-----
* A bug concerning the figure legend. Starting from ``matplotlib`` 2.1.0, it
  seems that ``handles`` and ``labels`` must be supplied to ``legend()`` as
  positional arguments not keywords. If they were given as keywords,
  ``legend()`` would have default values for them, and this causes a
  ``TypeError: __init__() got multiple values for argument 'handles'``.


0.1.12 - 2017-04-24
===================
Added
-----
* A quality flag calculated from the spread between the flux values from three
  fitting methods.
* An option to save configuration files in the output folder.
* A timelag method, ``'prescribed'``, which reads external timelag data files
  to determine timelags.


0.1.11 - 2017-04-23
===================
Added
-----
* A bounded timelag optimization method.
* Rules to filter warning messages for array division by zero or by NaN.

Changed
-------
* Improved the daily summary plot of chamber fluxes.
* Changed the output directory of curve fitting diagnostics to the subfolder
  ``diag`` in the data folder.

Fixed
-----
* A bug in the timelag optimization function, when ``closure_period_only`` is
  set to ``False``.
* A bug that crashed the program when skipping days with missing data.


0.1.10 - 2017-04-21
===================
Added
-----
* Baseline correction options: ``'median'``, ``'mean'``, and ``'none'``.
* Timelag optimization method: ``'optimized'``.

Changed
-------
* Default units for predefined species (``h2o``, ``co2``, ``cos``, ``co``,
  ``ch4``, ``n2o``). The user must now define the units explicitly in the
  configuration file, if units in the input data differ from the default.
* Improved the filtering mechanism for warning messages.


0.1.9 - 2017-03-23
==================
Added
-----
* An option ``load_data_by_day`` in the configuration to load and process data
  by daily chunks. This will be useful when dealing with a large number of
  input data files.

Changed
-------
* Replaced ``numpy.isclose`` with ``math.isclose`` for accuracy.

Fixed
-----
* An error in the CO concentration unit.
* A bug in assigning temperature and flow rate sensors to chambers.


0.1.8 - 2017-03-21
==================
Added
-----
* Plots for the daily summary of fluxes.
* An option ``save_daily_plots`` in the configuration file to enable plotting
  the daily summary of fluxes.

Fixed
-----
* Fixed the issue that an empty ``conc_atmb`` array caused the
  concentration-fitting plot to crash.
* Measurement periods with negative flow rates will not be used for flux
  calculation.
* Unreasonably large water concentration will not be used for dew temperature
  calculation.


0.1.7 - 2017-03-16
==================
Added
-----
* An option ``process_recent_period`` in the configuration. If ``True``, this
  lets the program to process only the data over the last few days instead of
  all available data. The number of days to trace back during processing is
  specified with the option ``traceback_in_days``.

Changed
-------
* Default configuration file changed from YAML (``config.yaml``) to Python
  (``default_config.py``).
* Improved data reading performance for a large number of CSV files.

Fixed
-----
* Failure of regression caused by NaN values in the concentration data. NaN
  values are now ignored.


0.1.6 - 2017-02-10
==================
Added
-----
* Parameter error estimates for the nonlinear fitting method.


0.1.5 - 2017-02-09
==================
Changed
-------
* Three times boost in performance compared to version 0.1.4 (not including
  plotting).
* Moved the reading of chamber configuration to the main script
  (``flux_calc.py``).
* Moved the timestamp conversion to the function ``load_tabulated_data()``.
* Output variables, except the day of year number, are now rounded off to 6
  decimal digits.

Deprecated
----------
* The option ``timestamp_format`` in the configuration file.

Removed
-------
* The function that converts timestamps to day of year values (float).
* The function that checks the starting year of the loaded data.


0.1.4 - 2017-02-06
==================
Added
-----
* A general function to parse tabulated data.
* Date parsing options for ``pandas.read_csv()``.

Removed
-------
* Functions to load each type of tabulated data input.

Fixed
-----
* A bug regarding the year number. The year number must now be given explicitly
  when using the day of year number as the time variable.


0.1.3 - 2017-02-05
==================
Added
-----
* An ``is_leaf_chamber`` identifier in the chamber description file.
* Support for leaf area auxiliary data files. Now the program can takes leaf
  area time series in the calculation rather than using fixed values over the
  whole period. If the ``separate_leaf_data`` option is enable, leaf chambers
  specified with ``is_leaf_chamber == True`` will use external leaf area data.
* The ``chamber_config_filepath`` option in the configuration to specify the
  file name of the chamber description file. The default chamber description
  file is ``chamber.yaml``.


0.1.2 - 2017-02-04
==================
Added
-----
* A bash script for the test case.
* Support for separate flow data files using the ``load_flow_data()`` function.

Changed
-------
* Refined the saturation vapor pressure and the dew temperature functions.
* Refined summary statistics functions in ``common_func.py``:
    - ``resist_mean()``: outlier-resistant mean
    - ``resist_std()``: outlier-resistant standard deviation
    - ``IQR_func()``: interquartile range
* List of physical constants moved from ``config.yaml`` to ``common_func.py``.


0.1.1 - 2017-01-18
==================
Added
-----
* A chamber description file ``chamber.yaml``.
* A chamber lookup function that generates a lookup table from the
  configuration file.
* Flow data settings in the configuration file.

Changed
-------
* Now use the ``dict.update()`` method for user customized configuration file.
* Change variable names of the standard errors of fluxes from ``sd_flux_*`` to
  ``se_flux_*``.

Fixed
-----
* A bug regarding the chamber schedule.


0.1.0 - 2017-01-07
==================
Added
-----
* A configuration file.
* Curve fitting plots.

Changed
-------
* The main script was reorganized into functions.
* Reformatted the code to comply with the PEP8 standard.

Fixed
-----
* A bug regarding the year number in ``flux_calc.flux_calc()``.


0.0.1 - 2016-07-18
==================
Added
-----
* The project was created by Wu Sun (wu.sun@ucla.edu).
