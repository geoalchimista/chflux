from typing import Dict, List

import pandas as pd

from chflux.tools.helpers import filter_substr


def find_timestamp(df: pd.DataFrame, utc: bool = True) -> str:
    """Find the proper timestamp column in a pandas DataFrame."""
    # # replace with this in Python 3.7!
    # if (ts_names := filter_substr(df.columns, "timestamp")) == []:
    #     if (ts_names := filter_substr(df.columns, "datetime")) == []:
    #         raise KeyError("No valid timestamp column found!")
    ts_names = filter_substr(df.columns, "timestamp")
    if ts_names == []:
        ts_names = filter_substr(df.columns, "datetime")
        if ts_names == []:
            raise KeyError("No timestamp found!")
    # filter by `_utc` or `_local` suffix
    if utc:
        ts_names = filter_substr(ts_names, "_utc") + \
            list(filter(lambda s: s in ["timestamp", "datetime"], ts_names))
        if ts_names == []:
            raise KeyError("No valid UTC timestamp found!")
    else:
        ts_names = filter_substr(ts_names, "_local") + \
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


def find_biomet_vars(df_bm: pd.DataFrame) -> Dict[str, List[str]]:
    names = {
        "pressure": [s for s in df_bm.columns if "pressure" in s],
        "T_atm": [s for s in df_bm.columns if "T_atm" in s],
        "RH_atm": [s for s in df_bm.columns if "RH_atm" in s],
        "T_ch": [s for s in df_bm.columns if "T_ch" in s],
        "PAR": [s for s in df_bm.columns if "PAR" in s and "PAR_ch" not in s],
        "PAR_ch": [s for s in df_bm.columns if "PAR_ch" in s],
        "T_leaf": [s for s in df_bm.columns if "T_leaf" in s],
        "T_soil": [s for s in df_bm.columns if "T_soil" in s],
        "w_soil": [s for s in df_bm.columns if "w_soil" in s],
    }
    return names


def reduce_dataframe(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """This is a just temporary reminder; will be deprecated."""
    return df.groupby(by, as_index=False).mean()
