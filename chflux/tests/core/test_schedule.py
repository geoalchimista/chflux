import json
import math

import numpy as np
import pandas as pd
import pytest

from chflux.core import schedule


def test_get_schedule():
    with open("./chflux/tests/_assets/test-chamber.json") as f:
        chamber_spec_json = json.load(f)

    ts1 = pd.Timestamp("2013-04-01 18:00")
    sch1 = schedule.get_schedule(ts1, chamber_spec_json)
    assert isinstance(sch1["df"], pd.DataFrame)
    assert isinstance(sch1["cycle_length"], float)
    assert isinstance(sch1["experiment_start"], pd.Timestamp)
    assert isinstance(sch1["experiment_end"], pd.Timestamp)

    ts2 = pd.Timestamp("2018-07-25 13:00")
    with pytest.warns(RuntimeWarning):
        sch2 = schedule.get_schedule(ts2, chamber_spec_json)
    assert sch2 == {}


def test_make_daily_schedule():
    with open("./chflux/tests/_assets/test-chamber.json") as f:
        chamber_spec_json = json.load(f)

    ts1 = pd.Timestamp("2013-04-01 18:00")
    schdf1 = schedule.make_daily_schedule(ts1, chamber_spec_json)
    assert isinstance(schdf1, pd.DataFrame)
    assert all(schdf1["schedule.end"] > schdf1["schedule.start"])

    ts2 = pd.Timestamp("2013-04-01 00:00")
    schdf2 = schedule.make_daily_schedule(ts2, chamber_spec_json)
    assert isinstance(schdf2, pd.DataFrame)
    assert schdf1.equals(schdf2)

    ts3 = pd.Timestamp("2018-07-25 13:00")
    with pytest.raises(RuntimeError):
        with pytest.warns(RuntimeWarning):
            schdf3 = schedule.make_daily_schedule(ts3, chamber_spec_json)


def test_label_chamber_period():
    label = "bm.index"
    df_bm = pd.read_csv(
        "./chflux/tests/_assets/test-input-biomet.csv", parse_dates=[0, ])
    # add timedelta
    df_bm["timedelta"] = (df_bm["timestamp"] -
                          df_bm.loc[0, "timestamp"].floor("D")) / \
        pd.Timedelta(1, "D")
    # make a copy for mutation test
    df_bm_cpy = df_bm.copy()

    with open("./chflux/tests/_assets/test-chamber.json") as f:
        chamber_spec_json = json.load(f)

    ts = pd.Timestamp("2013-04-01 00:00")
    sch = schedule.make_daily_schedule(ts, chamber_spec_json)

    df_bm_labeled = schedule.label_chamber_period(
        df_bm, sch, start="schedule.start", end="schedule.end", label=label)

    # the function must be immutable if inplace=False
    assert df_bm.equals(df_bm_cpy)
    assert df_bm_labeled["bm.index"].dtype == np.dtype('int64')

    # test the case when chamber_id does not match
    df_bm_bad_id = df_bm.copy()
    df_bm_bad_id["chamber_id"] += 20
    df_bm_bad_id_labeled = schedule.label_chamber_period(
        df_bm_bad_id, sch, start="schedule.start", end="schedule.end",
        label=label)
    assert all(df_bm_bad_id_labeled["bm.index"] == -1)

    df_bm_bad_id_ignored_labeled = schedule.label_chamber_period(
        df_bm_bad_id, sch, start="schedule.start", end="schedule.end",
        label=label, match_chamber_id=False)
    assert all(df_bm_bad_id_ignored_labeled["bm.index"] == -1) is False

    # test inplace=True mutation
    assert schedule.label_chamber_period(
        df_bm, sch, start="schedule.start", end="schedule.end", label=label,
        inplace=True) is None
    assert df_bm.equals(df_bm_cpy) is False
    assert label in df_bm.columns
