"""
===============================================
Data processing (:mod:`chflux.core.processing`)
===============================================

.. currentmodule:: chflux.core.processing

.. autosummary::
   :toctree: generated/

   convert_timestamp
   reduce_dataframe
   find_timestamp
   find_biomet_vars
"""
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

from chflux.core import const, flux, schedule, physchem
from chflux.tools import (filter_str, parse_concentration_units,
                          parse_day_number, parse_unix_time)

__all__ = ['convert_timestamp', 'reduce_dataframe',
           'find_timestamp', 'find_biomet_vars', ]


def convert_timestamp(df: pd.DataFrame, ref_year: int) -> None:
    """
    Convert non-conventional timestamps to a standard format. The converted
    timestamp series is added in place to the dataframe. Note that if a
    ``timestamp`` column already exists and is a valid ``datetime64`` type,
    no conversion will be done.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to append new timestamp to.
    ref_year : int
        The year to which Unix seconds or day-of-year numbers are referenced.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If no valid timestamp exists for the conversion.
    """
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(
            df['timestamp']):
        warnings.warn('Timestamp already exists. Nothing to be done!')
    elif 'timestamp_seconds' in df.columns:
        df['timestamp'] = parse_unix_time(df['timestamp_seconds'], ref_year)
    elif 'timestamp_days' in df.columns:
        df['timestamp'] = parse_day_number(df['timestamp_days'], ref_year)
    else:
        raise KeyError('Found no valid timestamp to convert!')


def reduce_dataframe(df: pd.DataFrame, by: str,
                     upper_bound=None, lower_bound=None) -> pd.DataFrame:
    """Reduce a dataframe by averaging according to a specified label."""
    if upper_bound is None and lower_bound is None:
        df_filtered = df
    elif upper_bound is None:
        df_filtered = df[(df[by] > lower_bound)]
    elif lower_bound is None:
        df_filtered = df[(df[by] < upper_bound)]
    else:
        df_filtered = df[(df[by] < upper_bound) & (df[by] > lower_bound)]

    df_reduced = df_filtered.groupby(by, as_index=False).mean()
    df_reduced = df_reduced.reset_index(drop=True)
    return df_reduced


# TO BE DEPRECATED
def find_timestamp(df: pd.DataFrame, utc: bool = True) -> str:
    """Find the proper timestamp column in a pandas DataFrame."""
    # # replace with this in Python 3.7!
    # if (ts_names := filter_str(df.columns, "timestamp")) == []:
    #     if (ts_names := filter_str(df.columns, "datetime")) == []:
    #         raise KeyError("No valid timestamp column found!")
    ts_names = filter_str(df.columns, "timestamp")
    if ts_names == []:
        ts_names = filter_str(df.columns, "datetime")
        if ts_names == []:
            raise KeyError("No timestamp found!")
    # filter by `_utc` or `_local` suffix
    if utc:
        ts_names = filter_str(ts_names, "_utc") + \
            list(filter(lambda s: s in ["timestamp", "datetime"], ts_names))
        if ts_names == []:
            raise KeyError("No valid UTC timestamp found!")
    else:
        ts_names = filter_str(ts_names, "_local") + \
            list(filter(lambda s: s in ["timestamp", "datetime"], ts_names))
        if ts_names == []:
            raise KeyError("No valid local timestamp found!")
    # # replace with this in Python 3.7!
    # if (ts_names := list(filter(
    #     lambda s: pd.api.types.is_datetime64_any_dtype(
    #         df[s]), ts_names))) == []:
    #     raise ValueError("Timestamp is not a valid datetime64 type!")
    ts_names = list(filter(
        lambda s: pd.api.types.is_datetime64_any_dtype(df[s]), ts_names))
    if ts_names == []:
        raise ValueError("Timestamp is not a valid datetime64 type!")

    return ts_names[0]


def find_biomet_vars(df: pd.DataFrame) -> Dict[str, List[str]]:
    names = {
        "pressure": [s for s in df.columns if "pressure" in s],
        "T_atm": [s for s in df.columns if "T_atm" in s],
        "RH_atm": [s for s in df.columns if "RH_atm" in s],
        "T_ch": [s for s in df.columns if "T_ch" in s],
        "PAR": [s for s in df.columns if "PAR" in s and "PAR_ch" not in s],
        "PAR_ch": [s for s in df.columns if "PAR_ch" in s],
        "T_leaf": [s for s in df.columns if "T_leaf" in s],
        "T_soil": [s for s in df.columns if "T_soil" in s],
        "w_soil": [s for s in df.columns if "w_soil" in s],
    }
    return names


def biomet_processor(df_biomet: pd.DataFrame,
                     chamber_schedule: pd.DataFrame) -> pd.DataFrame:
    df_biomet_labeled = schedule.label_chamber_period(
        df_biomet, chamber_schedule,
        start='schedule.start', end='schedule.end', label='biomet.index')
    df_biomet_reduced = reduce_dataframe(
        df_biomet_labeled, by='biomet.index', lower_bound=-0.5)
    if 'chamber_id' in df_biomet_labeled.columns:
        df_biomet_reduced = df_biomet_reduced.drop(
            columns=['timedelta', 'chamber_id'])
    else:
        df_biomet_reduced = df_biomet_reduced.drop(columns=['timedelta'])
    df_biomet_reduced = df_biomet_reduced.set_index('biomet.index', drop=True)
    df_biomet_reduced.index.name = None
    return df_biomet_reduced


def extract_species_info(config: Dict) -> Dict:
    species_list = config['species.list']
    species_info = {}
    for s in species_list:
        sdict = config[f'species.{s}']
        output_units = parse_concentration_units(sdict['output_unit'])
        species_info[s] = {
            'output_unit.concentration': output_units[0],
            'output_unit.flux': output_units[1],
            'multiplier': sdict['multiplier'],
            'baseline_correction': sdict['baseline_correction'],
        }
    return species_info


def rescale_concentration(df_concentration: pd.DataFrame,
                          species_info: Dict) -> pd.DataFrame:
    df_rescaled = df_concentration.copy()
    for k in species_info.keys():
        df_rescaled[k] *= species_info[k]['multiplier']
    return df_rescaled


def concentration_processor(df_concentration: pd.DataFrame,
                            chamber_schedule: pd.DataFrame,
                            config: Dict) -> pd.DataFrame:
    def _concentration_reducer(label, start, end):
        df_labeled = schedule.label_chamber_period(
            df_concentration, chamber_schedule,
            start=start, end=end, label=f'{label}.index',
            match_chamber_id=False)
        df_reduced = reduce_dataframe(
            df_labeled, by=f'{label}.index', lower_bound=-0.5)
        df_reduced = df_reduced.set_index(f'{label}.index', drop=True)
        df_reduced = df_reduced.drop(
            columns=[col for col in df_reduced.columns
                     if col not in config['species.list']])
        df_reduced.rename(
            columns={s: f'{s}_{label}' for s in df_reduced.columns},
            inplace=True)
        return df_reduced

    # ambient concentrations before the chamber measurement period
    df_reduced_chb = _concentration_reducer(
        'chb', 'schedule.bypass_before.start', 'schedule.bypass_before.end')
    # ambient concentrations after the chamber measurement period
    df_reduced_cha = _concentration_reducer(
        'cha', 'schedule.bypass_after.start', 'schedule.bypass_after.end')

    return pd.concat([df_reduced_chb, df_reduced_cha], axis=1)


def flux_processor(df_concentration: pd.DataFrame,
                   df_concentration_reduced: pd.DataFrame,
                   df_biomet_reduced: pd.DataFrame,
                   chamber_schedule: pd.DataFrame,
                   config: Dict,
                   species_info: Dict,
                   df_flow: Optional[pd.DataFrame] = None,
                   df_leaf: Optional[pd.DataFrame] = None,
                   df_timelag: Optional[pd.DataFrame] = None,
                   plotting: bool = False) -> Tuple[pd.DataFrame,
                                                    pd.DataFrame]:
    # initialize dataframes to store processed fluxes and curvefit diagnostics
    df_flux = pd.DataFrame({}, index=chamber_schedule.index)
    df_diag = pd.DataFrame({}, index=chamber_schedule.index)
    # process the concentration dataframe
    df_labeled = schedule.label_chamber_period(
        df_concentration, chamber_schedule,
        start='schedule.chamber.start', end='schedule.chamber.end',
        label='flux.index', match_chamber_id=False)
    for i in df_flux.index:
        for s in species_info.keys():
            conc = df_labeled.loc[df_labeled['flux.index'] == i, s].values
            # conc must be a numpy array
            t = (df_labeled.loc[df_labeled['flux.index'] == i, 'timedelta'] -
                 chamber_schedule.loc[i, 'schedule.chamber.start']) * 86400.
            t = t.values
            # for now, assume flow rates are given in the biomet dataframe
            # TODO: implement cases when flow rates are given in df_flow
            flow_id = chamber_schedule.loc[i, 'sensors_id.flowmeter']
            temp_id = chamber_schedule.loc[i, 'sensors_id.temperature']
            temp = df_biomet_reduced.loc[i, f'T_ch_{temp_id}']
            if 'pressure' in df_biomet_reduced.columns:
                pressure = df_biomet_reduced.loc[i, 'pressure']
            elif config['site.pressure'] is not None:
                pressure = config['site.pressure']
            else:
                pressure = const.atm
            # calculate the flow rate
            if config['flow.is_STP']:
                flow_STP = df_biomet_reduced.loc[i, f'flow_ch_{flow_id}']
                flow = physchem.convert_flowrate(flow_STP, temp, pressure)
            else:
                flow = df_biomet_reduced.loc[i, f'flow_ch_{flow_id}']
                flow_STP = flow / (temp / const.T_0 + (not kelvin)) * \
                    pressure / const.atm
            flow_molar = physchem.convert_flowrate_molar(flow_STP)
            area = chamber_schedule.loc[i, 'area']
            volume = chamber_schedule.loc[i, 'volume']
            turnover = volume / (flow * 1e-3 / 60.)
            # calculate fluxes using different fitting methods
            res_lin = flux.linfit(conc, t, turnover, area, flow_molar)
            res_rlin = flux.rlinfit(conc, t, turnover, area, flow_molar)
            res_nonlin = flux.nonlinfit(conc, t, turnover, area, flow_molar)
            # unpack results
            # linear regression
            df_flux.loc[i, f'f{s}_lin'] = res_lin.flux
            df_flux.loc[i, f'se_f{s}_lin'] = res_lin.se_flux
            # robust linear regression
            df_flux.loc[i, f'f{s}_rlin'] = res_rlin.flux
            df_flux.loc[i, f'se_f{s}_rlin'] = res_rlin.se_flux
            # nonlinear regression
            df_flux.loc[i, f'f{s}_nonlin'] = res_nonlin.flux
            df_flux.loc[i, f'se_f{s}_nonlin'] = res_nonlin.se_flux
            # others
            df_flux.loc[i, f'n_obs_{s}'] = res_lin.n_obs  # same for all
            # save diagnostics
            for k in ['delta_conc_fitted', 'rmse', 'rvalue', 'slope',
                      'intercept', 'pvalue', 'stderr']:
                df_diag.loc[i, f'{s}.lin.{k}'] = getattr(res_lin, k)
            for k in ['delta_conc_fitted', 'rmse', 'rvalue', 'slope',
                      'intercept', 'lo_slope', 'up_slope']:
                df_diag.loc[i, f'{s}.rlin.{k}'] = getattr(res_rlin, k)
            for k in ['delta_conc_fitted', 'rmse', 'rvalue', 'p0', 'p1',
                      'se_p0', 'se_p1']:
                df_diag.loc[i, f'{s}.nonlin.{k}'] = getattr(res_nonlin, k)

    return df_flux, df_diag

    # # get names of biomet variables
    # biomet_vars = find_biomet_vars(df_biomet)
    # # reduce biomet dataframe
    # df_biomet_labeled = schedule.label_chamber_period(
    #     df_biomet, chamber_schedule,
    #     start="schedule.start", end="schedule.end", label="biomet.index")
    # df_biomet_reduced = reduce_dataframe(
    #     df_biomet_labeled, by="biomet.index", lower_bound=-0.5)
    # df_biomet_reduced = df_biomet_reduced.drop(
    #     columns=["timedelta", "chamber_id"])
    # df_biomet_reduced = df_biomet_reduced.set_index("biomet.index", drop=True)
    # # append to the flux dataframe
    # df_flux = df_flux.join(df_biomet_reduced)

    # # get species list
    # species_list = config['species.list']
    # # reduce concentration dataframe
    # df_concentration_chb_labeled = schedule.label_chamber_period(
    #     df_concentration, chamber_schedule,
    #     start="schedule.bypass_before.start", end="schedule.bypass_before.end",
    #     label="chb.index", match_chamber_id=False)
    # df_concentration_chb_reduced = reduce_dataframe(
    #     df_concentration_chb_labeled, by="chb.index", lower_bound=-0.5)
    # df_concentration_chb_reduced = df_concentration_chb_reduced.drop(
    #     columns=["timedelta"])
    # df_concentration_chb_reduced = df_concentration_chb_reduced.set_index(
    #     "chb.index", drop=True)
    # df_flux = df_flux.join(df_concentration_chb_reduced)

    # df_concentration_cha_labeled = schedule.label_chamber_period(
    #     df_concentration, chamber_schedule,
    #     start="schedule.bypass_after.start", end="schedule.bypass_after.end",
    #     label="cha.index", match_chamber_id=False)
    # df_concentration_cha_reduced = reduce_dataframe(
    #     df_concentration_cha_labeled, by="cha.index", lower_bound=-0.5)
    # df_concentration_cha_reduced = df_concentration_cha_reduced.drop(
    #     columns=["timedelta"])
    # df_concentration_cha_reduced = df_concentration_cha_reduced.set_index(
    #     "cha.index", drop=True)
    # df_flux = df_flux.join(df_concentration_cha_reduced)

    # # label chamber measurement periods
    # df_concentration_chc_labeled = schedule.label_chamber_period(
    #     df_concentration, chamber_schedule,
    #     start="schedule.chamber.start", end="schedule.chamber.end",
    #     label="chc.index", match_chamber_id=False)
    # # calculate fluxes
    # for i in df_concentration_chc_labeled['chc.index'].values.unique():
    #     for s in species_list:
    #         conc = df_concentration_chc_labeled.loc[
    #             df_concentration_chc_labeled['chc.index'] == i, s].values
    #         t = df_concentration_chc_labeled.loc[
    #             df_concentration_chc_labeled['chc.index'] == i,
    #             'timedelta'].values * 86400.0
    #         flow_slpm = df_flux.loc[
    #             i, 'flow_' + chamber_schedule.loc[i, 'sensors_id.flowmeter']]
    #         T_ch = df_flux.loc[
    #             i, 'T_ch_' + chamber_schedule.loc[i, 'sensors_id.temperature']]
    #         pressure = df_flux.loc[i, 'pressure']
    #         flow_lpm = flow_slpm * (1. + T_ch / const.T_0) * \
    #             const.atm / pressure
    #         flow = flow_slpm * 1e-3 / 60. * const.air_concentration
    #         area = df_flux.loc[i, 'area']
    #         volume = df_flux.loc[i, 'volume']
    #         turnover = volume / (flow_lpm * 1e-3 / 60.)

    # TO BE CONTINUED


def process_all_data(config: Dict,
                     chamber_schedule: pd.DataFrame,
                     df_biomet: pd.DataFrame,
                     df_concentration: Optional[pd.DataFrame] = None,
                     df_flow: Optional[pd.DataFrame] = None,
                     df_leaf: Optional[pd.DataFrame] = None,
                     df_timelag: Optional[pd.DataFrame] = None,
                     plotting: Optional[bool] = False) -> Tuple[
                         pd.DataFrame, pd.DataFrame]:
    if df_concentration is None:
        df_concentration = df_biomet  # alias
    if df_flow is None:
        df_flow = df_biomet  # alias

    # perform data reduction & flux calculation
    # biomet data
    df_biomet_reduced = biomet_processor(df_biomet, chamber_schedule)
    # concentration data
    species_info = extract_species_info(config)
    df_concentration = rescale_concentration(df_concentration, species_info)
    df_concentration_reduced = concentration_processor(
        df_concentration, chamber_schedule, config)
    # calculate fluxes
    df_flux, df_diag = flux_processor(
        df_concentration,
        df_concentration_reduced,
        df_biomet_reduced,
        chamber_schedule,
        config,
        species_info,
    )

    # add chamber information
    df_flux_chamber = chamber_schedule[['name', 'id', 'volume', 'area',
                                        'is_leaf_chamber']].copy()
    df_diag_chamber = chamber_schedule[['name', 'id']].copy()

    # combined processed dataframes to df_flux
    df_flux_all = pd.concat([df_flux_chamber,
                             df_biomet_reduced,
                             df_concentration_reduced,
                             df_flux], axis=1)
    df_diag_all = pd.concat([df_flux_chamber, df_diag], axis=1)

    # add timestamps
    ref_ts = df_biomet.loc[df_biomet.shape[0] // 2, 'timestamp'].floor('D')
    df_flux_all.insert(
        0, 'timestamp',
        (chamber_schedule['schedule.chamber.start'] +
         chamber_schedule['schedule.chamber.end']) * 0.5 *
        pd.Timedelta('1 D') + ref_ts)
    df_flux_all['timestamp'] = df_flux_all['timestamp'].dt.round('s')
    df_diag_all.insert(0, 'timestamp', df_flux_all['timestamp'])

    return df_flux_all, df_diag_all
