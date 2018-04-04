"""
A collection of various helper tools for PyChamberFlux

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

"""
import math


# @TODO: deprecated; use `chflux.tools.parse_unit_names``
def convert_unit_names(output_unit_list):
    """
    A helper function to convert units to text representation.

    Parameters
    ----------
    output_unit_list : list of float
        A list of the output units to parse.

    Returns
    -------
    conc_unit_names : list of str
        A list of concentration unit names.
    flux_unit_names : list of str
        A list of flux unit names.
    """
    conc_unit_names = []
    flux_unit_names = []
    for output_unit in output_unit_list:
        if math.isclose(output_unit, 1e-12):
            conc_unit = 'pmol mol$^{-1}$'
            flux_unit = 'pmol m$^{-2}$ s$^{-1}$'
        elif math.isclose(output_unit, 1e-9):
            conc_unit = 'nmol mol$^{-1}$'
            flux_unit = 'nmol m$^{-2}$ s$^{-1}$'
        elif math.isclose(output_unit, 1e-6):
            conc_unit = '$\mu$mol mol$^{-1}$'
            flux_unit = '$\mu$mol m$^{-2}$ s$^{-1}$'
        elif math.isclose(output_unit, 1e-3):
            conc_unit = 'mmol mol$^{-1}$'
            flux_unit = 'mmol m$^{-2}$ s$^{-1}$'
        elif math.isclose(output_unit, 1e-2):
            conc_unit = '%'
            flux_unit = '% m$^{-2}$ s$^{-1}$'
        elif math.isclose(output_unit, 1.):
            conc_unit = 'mol mol$^{-1}$'
            flux_unit = 'mol m$^{-2}$ s$^{-1}$'
        else:
            conc_unit = 'undefined unit'
            flux_unit = 'undefined unit'

        conc_unit_names.append(conc_unit)
        flux_unit_names.append(flux_unit)

    return conc_unit_names, flux_unit_names


# @TODO: to deprecate
def create_output_header(data_type, species_list, biomet_var_list=[]):
    """
    A helper function to create the header for output data frame.

    Parameters
    ----------
    data_type : str
        The type of output dataframe
        * 'flux' - flux data
        * 'diag' - curve fitting diagnostics
    species_list : list of str
        List of gas species
    biomet_var_list : list of str
        List of biometeorological variable names

    Returns
    -------
    header : list of str
        Table header for the output dataframe. If `data_type` is illegal,
        return a blank list.
    """
    if data_type == 'flux':
        header = ['doy_utc', 'doy_local', 'ch_no', 'ch_label', 'A_ch', 'V_ch']
        for conc_suffix in ['_atmb', '_chb', '_cha', '_atma']:
            header += [s + conc_suffix for s in species_list]
            header += ['sd_' + s + conc_suffix for s in species_list]

        header += [s + '_chc_iqr' for s in species_list]
        for flux_method in ['_lin', '_rlin', '_nonlin']:
            header += ['f' + s + flux_method for s in species_list]
            header += ['se_f' + s + flux_method for s in species_list]

        # add quality flags for fluxes
        header += ['qc_' + s for s in species_list]

        # add number of valid observations of concentrations
        header += ['n_obs_' + s for s in species_list]

        # biomet variables and other auxiliary variables
        header += ['flow_lpm', 't_turnover', 't_lag_nom', 't_lag_optmz',
                   'status_tlag', 'pres', 'T_log', 'T_inst'] + biomet_var_list
    elif data_type == 'diag':
        header = ['doy_utc', 'doy_local', 'ch_no']
        for s in species_list:
            header += ['k_lin_' + s, 'b_lin_' + s, 'r_lin_' + s,
                       'p_lin_' + s, 'rmse_lin_' + s, 'delta_lin_' + s]

        for s in species_list:
            header += ['k_rlin_' + s, 'b_rlin_' + s,
                       'k_lolim_rlin_' + s, 'k_uplim_rlin_' + s,
                       'rmse_rlin_' + s, 'delta_rlin_' + s]

        for s in species_list:
            header += ['p0_nonlin_' + s, 'p1_nonlin_' + s,
                       'se_p0_nonlin_' + s, 'se_p1_nonlin_' + s,
                       'rmse_nonlin_' + s, 'delta_nonlin_' + s]
    else:
        return []

    return header
