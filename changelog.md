# Change Log

@TODO:

- Use `pandas.Timestamp` as the standard timestamp passed between functions.
- Add timelag control in `chamber.yaml`
- Test and refine the chamber schedule function for timelag optimization.
- Daily summary plots.
- Correct error in RMSE calculation.

# 0.1.7 (2017-03-16)

- Changed the default configuration file from YAML (`config.yaml`) to Python (`default_config.py`).
- Improved data reading performance for large number of CSV files.

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
