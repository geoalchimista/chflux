# Change Log

@TODO:
- Add external timelag file as input.
- Replace Theil-Sen estimator with a more efficient robust linear regression method (perhaps MM-estimator?).
- Use `pandas.Timestamp` as the standard timestamp passed between functions.
- Add options to save configuration files in the output folder.

# 0.1.11


# 0.1.10 (2017-04-21)

- Added baseline correction options: `'median'`, `'mean'`, and `'none'`.
- Timelag optimization (method `'optimized'`) have been implemented.
- Default units for predefined species (`h2o`, `co2`, `cos`, `co`, `ch4`, `n2o`) have been changed. The user must define the units explicitly in the configuration file if units in their data differ from the default.
- Warning filtering mechanism improved.

# 0.1.9 (2017-03-23)

- Added a running option `load_data_by_day` for loading and processing the data by daily chunks. This will be useful when dealing with a large number of data files.
- Fixed the error of CO concentration unit. Replaced `numpy.isclose` with `math.isclose` for accuracy.
- Fixed the error in assigning temperature and flow rate sensors to chambers.

# 0.1.8 (2017-03-21)

- Daily flux summary plot has been implemented. It can be enabled from the key `save_daily_plots` under `run_options` in the configuration file.
- Bug fix: the issue that an empty `conc_atmb` array caused the concentration-fitting plot to crash has been fixed.
- Bug fix: measurement periods with negative flow rates will not be used for flux calculation.
- Bug fix: erroneously large water concentration will not be used for dew temperature calculation.

# 0.1.7 (2017-03-16)

- Bug fix: NaN values of concentration that caused regression procedures to fail are now filtered out.
- Changed the default configuration file from YAML (`config.yaml`) to Python (`default_config.py`).
- Improved data reading performance for large number of CSV files.
- New option: `process_recent_period`. If enabled (`True`), this will let the program to process only the data over the last few days instead of all available data. The number of days to trace back during processing is specified with the option `traceback_in_days`.

# 0.1.6 (2017-02-10)

- Added parameter error estimates for the nonlinear fitting method.

# 0.1.5 (2017-02-09)

- Three times boost in performance compared to version 0.1.4 (not including plotting).
- `timestamp_format` option in the config file has been deprecated.
- Chamber configuration loading moved to the main program.
- Moved time stamp conversion to the function `load_tabulated_data()`. The function that converts timestamps to day of year values (float) has been deprecated and removed. The function that checks the starting year of the loaded data has also been removed.
- Output variables rounded off to 6 decimal digits (not including day of year variables or chamber descriptors).

# 0.1.4 (2017-02-06)

- Added a general function to parse tabulated data. Other data reading functions have been deprecated.
- Added date parsing options for `pandas.read_csv()`.
- Bug fix: year number must be specified when using day of year number as the time variable.

# 0.1.3 (2017-02-05)

- Chamber configuration file can be specified in the general configuration file: `run_options` -> `chamber_config_filepath`. The default is file path is `chamber.yaml`.
- Added `is_leaf_chamber` identifier for the chamber config file.
- Added leaf area auxiliary data support. Now the program can takes leaf area time series in flux calculation, rather than fixed values over the whole period. If the `separate_leaf_data` setting is enable, leaf chambers specified with `is_leaf_chamber == True` will take external leaf area data.

# 0.1.2 (2017-02-04)

- Separate flow data files are supported with `load_flow_data()` function.
- Refined saturation vapor pressure and dew temperature functions.
- Refined summary statistics functions in `common_func.py`:
 * `resist_mean()`: outlier-resistant mean
 * `resist_std()`: outlier-resistant standard deviation
 * `IQR_func()`: interquartile range
- List of physical constants moved from `config.yaml` to `common_func.py`.
- Added a bash script for the test case.

# 0.1.1 (2017-01-18)

- Added a new chamber lookup table function controlled by external config file `chamber.yaml`.
- Bug fix in chamber schedule.
- Added flow data settings to the config.
- Use `dict.update()` method for user custom config.
- Variable fix: standard error of flux estimate, `sd_flux_*` --> `se_flux_*`.

# 0.1.0 (2017-01-07)

- Main program reorganized into functions.
- Configuration file generated.
- Reformatted to comply with PEP8 standard.
- Bug fix: year number in `flux_calc.flux_calc()`.
- Added a procedure to generate curve fitting plots.

# 0.0.1 (2016-07-18)

- Created by Wu Sun (wu.sun@ucla.edu).
