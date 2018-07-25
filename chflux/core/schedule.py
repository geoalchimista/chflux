"""Utilities to obtain experimental schedules from chamber specifications."""
import sys
import warnings
from typing import Dict

import numpy as np
import pandas as pd

from chflux.tools.helpers import flatten_dict


def time_unit_conversion_factor(unitname: str) -> float:
    """Return a multiplier that converts a given unit of time to day(s)."""
    if unitname in ["seconds", "second", "sec", "s"]:
        conv_fac = 1. / (60. * 60. * 24.)
    elif unitname in ["minutes", "minute", "min", "m"]:
        conv_fac = 1. / (60. * 24.)
    elif unitname in ["hours", "hour", "hr", "h"]:
        conv_fac = 1. / 24.
    else:
        conv_fac = 1.0
    return conv_fac


def get_schedule(ts: pd.Timestamp, experiments: Dict) -> Dict:
    for ex_id in experiments:
        ex = experiments[ex_id]
        if type(ex["start"]) is str and type(ex["end"]) is str:
            ex_start = pd.Timestamp(ex["start"])
            ex_end = pd.Timestamp(ex["end"])
            if ex_start <= ts < ex_end:
                break
    else:
        warnings.warn(
            "No valid chamber schedule exist for timestamp %s" % str(ts),
            RuntimeWarning)
        return {}

    chamber_specs_flatten = [flatten_dict(d) for d in ex["chambers"]]
    # add schedule.start offset to other schedule variables
    for d in chamber_specs_flatten:
        for k, v in d.items():
            if "schedule." in k and k != "schedule.start":
                d[k] = v + d["schedule.start"]

    df = pd.DataFrame(chamber_specs_flatten)

    # convert all time variables to be in day
    conv_fac = time_unit_conversion_factor(ex["unit_of_time"])
    time_cols = [c for c in df.columns
                 if ("schedule" in c or "timelag" in c) and
                    (c != "timelag.optimize")]
    df[time_cols] *= conv_fac

    return {
        "df": df,
        "cycle_length": ex["cycle_length"] * conv_fac,  # unit of time is day
        "experiment_start": ex_start,
        "experiment_end": ex_end
    }


def make_daily_schedule(date: pd.Timestamp, experiments: Dict) -> pd.DataFrame:
    td_day = pd.Timedelta(1.0, 'D')  # const: timedelta of one day
    date = date.floor("D")  # round down to the start of the day
    timer = 0.0
    df = pd.DataFrame([])  # daily schedule dataframe

    # unpack current experimental schedule
    sch = get_schedule(date, experiments)
    if sch == {}:
        raise RuntimeError("Cannot find a valid schedule on the date given!")
    cycle_length = sch["cycle_length"]
    experiment_end_delta = (sch["experiment_end"] - date) / td_day

    # roll over the day
    while timer < 1.0:
        df_now = sch["df"].copy()
        sch_cols = [c for c in df_now.columns if "schedule." in c]
        df_now[sch_cols] += timer
        df = df.append(df_now, ignore_index=True)
        timer += cycle_length
        # check if the schedule has switched
        if experiment_end_delta < timer < 1.0:
            df = df[df["schedule.start"] < experiment_end_delta]  # truncate
            # switch the schedule
            sch = get_schedule(date + pd.Timedelta(timer), experiments)
            cycle_length = sch["cycle_length"]
            experiment_start_delta = (sch["experiment_start"] - date) / td_day
            experiment_end_delta = (sch["experiment_end"] - date) / td_day
            # roll back the timer by 1 cycle to adjust for the current schedule
            timer = np.floor(timer / cycle_length - 1) * cycle_length
            df_now = sch["df"].copy()
            sch_cols = [c for c in df_now.columns if "schedule." in c]
            df_now[sch_cols] += timer
            # truncate to prevent adding duplicates
            df_now = df_now[df_now["schedule.start"] >= experiment_start_delta]

            df = df.append(df_now, ignore_index=True)
            df = df.reset_index(drop=True)
            timer += cycle_length

    df = df[df["schedule.start"] < 1.0]
    return df


def label_chamber_period(data: pd.DataFrame, sch: pd.DataFrame,
                         start: str, end: str, label: str,
                         match_chamber_id: bool = True,
                         inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        data_orig = data
        data = data_orig.copy()

    if label not in data.columns:
        data[label] = -1
    for i in sch.index:
        slice_index = (data["timedelta"] > sch.loc[i, start]) & \
            (data["timedelta"] < sch.loc[i, end])
        if match_chamber_id:
            slice_index = slice_index & \
                (data["chamber_id"] == sch.loc[i, "id"])
        data.loc[slice_index, label] = i
    if not inplace:
        return data
    else:
        return None
