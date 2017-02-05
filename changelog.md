# Change Log

# current

- Added `is_leaf_chamber` identifier for the chamber config file `chamber.yaml`
- Added leaf area auxiliary data support. Now the program can takes leaf area time series in flux calculation, rather than fixed values over the whole period. If the `separate_leaf_data` setting is enable, leaf chambers specified with `is_leaf_chamber == True` will take external leaf area data.

@TODO:
- Add timelag control in `chamber.yaml`
- Test and refine the chamber schedule function for timelag optimization.
- Daily summary plots.

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
