# PyChamberFlux Documentation

Wu Sun (wu.sun@ucla.edu)

0.1 alpha (August 23, 2016)

License: [GPL v3](https://www.gnu.org/licenses/gpl-3.0-standalone.html)

A Python software for calculating gas fluxes between land ecosystems and the atmosphere from flow-through chamber measurements.

[TOC]

## Requirements

- Python 3.4 and up (https://www.python.org)

Python libraries

- numpy (http://www.numpy.org)
- scipy (http://scipy.org)
- matplotlib (http://matplotlib.org)
- pandas (http://pandas.pydata.org)


## Configuration







---


## Quick start






## Variable types

### Chamber number

### Temperature

- Chamber temperatures

- Leaf temperatures

- Soil temperatures

- Atmospheric temperatures

### Relative humidity

- Atmospheric relative humidity

- Chamber relative humidity

### Radiation

- Photosynthetically available radiation (PAR)

- Other types of radiation

### Flow rates

- Flow rate into each chamber

### Soil moisture

### Concentrations

### Fluxes

## Configurations

### Running options

`run_options`: *dict*

Options used for doing flux calculation and plotting the data. 

- **`baseline_correction`**: *boolean*

  If `True`, a baseline correction will be applied. If `False`, no such correction. However, this correction requires two chamber opening intervals, one before chamber closure period, and the other after chamber closure period. If a zero-flux baseline is unable to be generated, this correction will not be applied. 

- **`median_baseline`**: *boolean*

  If `True`, use median concentrations of the two chamber opening intervals to construct the baseline; if `False`, use mean concentrations. 


- **`timelag_method`**: allowed methods include `'nominal'`, `'manual'`, and `'optimized'`. 

  For 'nominal' time lags, a time lag function must be assigned in `general_func.py` to calculate time lag from tubing dimensions and flow rates. 

  For 'manual' time lags, a fixed time lag is assigned. 

  For 'optimized' time lags, time lag optimization is enabled and the cost function  for it will be summoned from `general_func.py`.  [==**not available yet**==]

- **`timelag_optimization_species`**: species to use for time lag optimization. Water is not recommended because it may interact with the sampling tube in low-flow condition. [==**not available yet**==]

- **`volume_correction`**: if `True`, optimize the effective volume ($V_\mathrm{eff}$) to fit the curvature of concentration changes. 

  The effectively exchanged volume may be different from the nominal chamber volume. This will change the gas turnover time in the chamber system, and therefore affect the curvature of fitted curve. Use this option with caution. Unless enabling it actually improves the fit, more often than not it suffices to run without this correction. [==**not available yet**==]

- **`curve_fitting_method`**: allowed methods include `'linear'`, `'robust_linear'`, `'nonlinear'`, and `'all'`. 

  The 'linear' method calculates fluxes from simple linear regression (see the Theory section). The 'robust linear' method calculates fluxes from robust linear regression. The 'nonlinear' method fits an exponential function, with a small tolerance range for time lag. To use all three methods at the same time and compare their results, set the method to `'all'` (warning: it costs more computational time). 

- **`process_today`**: *boolean*

  If `True`, process only today's data for online calculation. If `False`, process all available data. [==**not available yet**==]

- **`save_fitting_plots`**: *boolean*

  If `True`, save curve fitting plots. 

- **`save_daily_plots`**: *boolean*

  If `True`, save daily summary plots of fluxes and biomet variables. 

### Data directories

`data_dir`: *dict*

- **`biomet_data`**: (absolute or relative) directory to search for biometeorological data files
- **`conc_data`**: (absolute or relative) directory to search for concentration data files
- **`output_dir`**: output directory for the processed flux data
- **`output_filename_str`**: a prefix string to append before date stamp for output data file names
- **`plot_dir`**: directory for saved plots
- **`separate_conc_data`**: If set to `True`, concentration measurements are stored on their own in files specified by the **`conc_data`** entry. If set to `False`, search concentration measurements in the biometeorological data files. [==**not available yet**==]

### Biometeorological data settings

`biomet_data_settings`: *dict*

- **`delimiter`**: delimiter of the biometeorological data table

- **`skip_header`**: number of rows to skip at the beginning of the file

- **`names`**: column names, must standardizes the names of the variables of interest (see the Variables section)

- **`usecols`**: columns to read from the biometeorological data table. If reading all columns, set this to `None`.

- **`dtype`**: to explicitly define the data types of columns. If set to `None`, this will be handled by `numpy.genfromtxt`. 

- **`missing_values`**: to specify the representation of missing values. If set to `None`, this will be handled by `numpy.genfromtxt`.

- **`columns_to_save`**: to specify the columns to be saved. If set to `None` (default), process all columns that contain standardized variables. [==**not available yet**==]

- **`time_in_UTC`**: *boolean*

  If the time variable is in UTC, set this to `True`. 

- **`flow_rate_in_STP`**: *boolean*

  By default set to `True`. The flow rate is referenced to STP condition, as for many commonly available flow sensors. This means that the flow rates need to be corrected for ambient temperature and pressure. If set `False`, no such correction on flow rates is applied. 

### Concentration data settings

`conc_data_settings`: *dict*

- **`delimiter`**: delimiter of the concentration data table

- **`skip_header`**: number of rows to skip at the beginning of the file

- **`names`**: column names, must standardizes the names of the variables of interest (see the Variables section)

- **`usecols`**: columns to read from the concentration data table. If reading all columns, set this to `None`.

- **`dtype`**: to explicitly define the data types of columns. If set to `None`, this will be handled by `numpy.genfromtxt`. 

- **`missing_values`**: to specify the representation of missing values. If set to `None`, this will be handled by `numpy.genfromtxt`.

- **`columns_to_save`**: to specify the columns to be saved. If set to `None` (default), process all columns that contain standardized variables. [==**not available yet**==]

- **`time_in_UTC`**: *boolean*

  If the time variable is in UTC, set this to `True`. 

### Constants

`consts`: *dict*

Physical constants and site-specific constants. 

- **`p_std`**: standard atmospheric pressure (Pa)
- **`R_gas`**: universal gas constant (J K^-1^ mol^-1^)
- **`T_0`**: zero Celsius in Kelvin
- **`air_conc_std`**: air concentration (mol m^-3^) at standard pressure and temperature condition
- **`site_pressure`**: ambient pressure at the site (Pa)
- **`time_zone`**: with respect to UTC, –12 to +12 (e.g., –8 means UTC–8 time zone). 

### Species settings

`species_settings` : *collections.defaultdict(tree)* 

Settings for the gas species being measured. 

- **`species_list`**: List of gas species being calculated for fluxes. 
- **`species`**: Each species is a nested entry that has these sub-entries: 
  - **`unit`**: unit of mixing ratio in the concentration data file (e.g., 10^–9^ is ppbv, 10^-6^ is ppmv).
  - **`output_unit`**: unit of mixing ratio in the output file. Similar as the **`unit`** entry. 
  - **`multiplier`**: a multiplier to convert the unit of mixing ratio. 


## Functions

### Chamber information lookup table

### Flow rate function

### Time lag optimization function

### Effective volume optimization 



### Concentration function 

## References



## Appendix

### Theory

