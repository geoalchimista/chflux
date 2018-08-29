"""
===============================================
Data processing (:mod:`chflux.core.processing`)
===============================================

.. currentmodule:: chflux.core.processing

.. autosummary::
   :toctree: generated/

   convert_timestamp
   find_timestamp
   find_biomet_vars
   reduce_dataframe
"""
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

from chflux.core import const, flux, schedule
from chflux.tools import (filter_str, parse_concentration_units,
                          parse_day_number, parse_unix_time)

__all__ = ['convert_timestamp',
           'find_timestamp', 'find_biomet_vars', 'reduce_dataframe']


def convert_timestamp(df: pd.DataFrame, ref_year: int) -> None:
    """
    Convert non-conventional timestamps to a standard format. The converted
    timestamp is added in place to the dataframe.
    """
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(
            df['timestamp']):
        warnings.warn('Timestamp already exists. Nothing to be done!')
    elif 'timestamp_seconds' in df.columns:
        df['timestamp'] = parse_unix_time(df['timestamp_seconds'], ref_year)
    elif 'timestamp_days' in df.columns:
        df['timestamp'] = parse_day_number(df['timestamp_days'], ref_year)
    else:
        warnings.warn('Found no valid timestamp to convert!')


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


def flux_processor(config: Dict,
                   chamber_schedule: pd.DataFrame,
                   df_biomet: pd.DataFrame,
                   df_concentration: Optional[pd.DataFrame] = None,
                   df_flow: Optional[pd.DataFrame] = None,
                   df_leaf: Optional[pd.DataFrame] = None,
                   df_timelag: Optional[pd.DataFrame] = None,
                   plotting: Optional[bool] = False) -> None:
    # initialize dataframes to store processed fluxes and curvefit diagnostics
    df_flux = chamber_schedule[['name', 'id', 'volume', 'area',
                                'is_leaf_chamber']].copy()
    df_diag = chamber_schedule[['name', 'id']].copy()

    # get names of biomet variables
    biomet_vars = find_biomet_vars(df_biomet)
    # reduce biomet dataframe
    df_biomet_labeled = schedule.label_chamber_period(
        df_biomet, chamber_schedule,
        start="schedule.start", end="schedule.end", label="biomet.index")
    df_biomet_reduced = reduce_dataframe(
        df_biomet_labeled, by="biomet.index", lower_bound=-0.5)
    df_biomet_reduced = df_biomet_reduced.drop(
        columns=["timedelta", "chamber_id"])
    df_biomet_reduced = df_biomet_reduced.set_index("biomet.index", drop=True)
    # append to the flux dataframe
    df_flux = df_flux.join(df_biomet_reduced)

    # get species list
    species_list = config['species.list']
    # reduce concentration dataframe
    df_concentration_chb_labeled = schedule.label_chamber_period(
        df_concentration, chamber_schedule,
        start="schedule.bypass_before.start", end="schedule.bypass_before.end",
        label="chb.index", match_chamber_id=False)
    df_concentration_chb_reduced = reduce_dataframe(
        df_concentration_chb_labeled, by="chb.index", lower_bound=-0.5)
    df_concentration_chb_reduced = df_concentration_chb_reduced.drop(
        columns=["timedelta"])
    df_concentration_chb_reduced = df_concentration_chb_reduced.set_index(
        "chb.index", drop=True)
    df_flux = df_flux.join(df_concentration_chb_reduced)

    df_concentration_cha_labeled = schedule.label_chamber_period(
        df_concentration, chamber_schedule,
        start="schedule.bypass_after.start", end="schedule.bypass_after.end",
        label="cha.index", match_chamber_id=False)
    df_concentration_cha_reduced = reduce_dataframe(
        df_concentration_cha_labeled, by="cha.index", lower_bound=-0.5)
    df_concentration_cha_reduced = df_concentration_cha_reduced.drop(
        columns=["timedelta"])
    df_concentration_cha_reduced = df_concentration_cha_reduced.set_index(
        "cha.index", drop=True)
    df_flux = df_flux.join(df_concentration_cha_reduced)

    # label chamber measurement periods
    df_concentration_chc_labeled = schedule.label_chamber_period(
        df_concentration, chamber_schedule,
        start="schedule.chamber.start", end="schedule.chamber.end",
        label="chc.index", match_chamber_id=False)
    # calculate fluxes
    for i in df_concentration_chc_labeled['chc.index'].values.unique():
        for s in species_list:
            conc = df_concentration_chc_labeled.loc[
                df_concentration_chc_labeled['chc.index'] == i, s].values
            t = df_concentration_chc_labeled.loc[
                df_concentration_chc_labeled['chc.index'] == i,
                'timedelta'].values * 86400.0
            flow_slpm = df_flux.loc[
                i, 'flow_' + chamber_schedule.loc[i, 'sensors_id.flowmeter']]
            T_ch = df_flux.loc[
                i, 'T_ch_' + chamber_schedule.loc[i, 'sensors_id.temperature']]
            pressure = df_flux.loc[i, 'pressure']
            flow_lpm = flow_slpm * (1. + T_ch / const.T_0) * \
                const.atm / pressure
            flow = flow_slpm * 1e-3 / 60. * const.air_concentration
            area = df_flux.loc[i, 'area']
            volume = df_flux.loc[i, 'volume']
            turnover = volume / (flow_lpm * 1e-3 / 60.)

            res_linfit = flux.linfit(conc, t, turnover, area, flow)
            res_rlinfit = flux.rlinfit(conc, t, turnover, area, flow)
            res_nonlinfit = flux.nonlinfit(conc, t, turnover, area, flow)
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

    # initialize dataframes to store processed fluxes and curvefit diagnostics
    df_flux = chamber_schedule[['name', 'id', 'volume', 'area',
                                'is_leaf_chamber']].copy()
    df_diag = chamber_schedule[['name', 'id']].copy()

    df_biomet_reduced = biomet_processor(df_biomet, chamber_schedule)

    species_info = extract_species_info(config)
    df_concentration = rescale_concentration(df_concentration, species_info)
    df_concentration_reduced = concentration_processor(
        df_concentration, chamber_schedule, config)

    # combined processed dataframes to df_flux
    df_flux = pd.concat([df_flux, df_biomet_reduced,
                         df_concentration_reduced], axis=1)

    # TO BE CONTINUED
    return df_flux, df_diag
