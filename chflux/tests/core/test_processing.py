import pandas as pd

import json
from chflux.core import processing, schedule


def test_find_timestamp():
    df_bm = pd.read_csv("./chflux/tests/_assets/test-input-biomet.csv",
                        parse_dates=[0, ])

    assert processing.find_timestamp(df_bm) == "timestamp"

    assert processing.find_timestamp(
        df_bm.rename(columns={"timestamp": "datetime"})
    ) == "datetime"

    assert processing.find_timestamp(
        df_bm.rename(columns={"timestamp": "timestamp_utc"})
    ) == "timestamp_utc"

    # test utc=False cases
    assert processing.find_timestamp(df_bm, utc=False) == "timestamp"
    assert processing.find_timestamp(
        df_bm.rename(columns={"timestamp": "datetime"}),
        utc=False) == "datetime"

    # test multiple timestamps
    df_bm["timestamp_local"] = df_bm["timestamp"]
    assert processing.find_timestamp(df_bm) == "timestamp"
    assert processing.find_timestamp(df_bm, utc=False) == "timestamp_local"


def test_find_biomet_vars():
    df_bm = pd.read_csv("./chflux/tests/_assets/test-input-biomet.csv",
                        parse_dates=[0, ])

    biomet_vars = processing.find_biomet_vars(df_bm)
    assert isinstance(biomet_vars, dict)
    for _, v in biomet_vars.items():
        assert isinstance(v, list)

    # a null case
    df_null = pd.DataFrame({"A": range(100)})
    null_biomet_vars = processing.find_biomet_vars(df_null)
    assert isinstance(null_biomet_vars, dict)
    for _, v in null_biomet_vars.items():
        assert v == []


def test_reduce_dataframe():
    label = "bm.index"
    df_bm = pd.read_csv(
        "./chflux/tests/_assets/test-input-biomet.csv", parse_dates=[0, ])

    ts_name = processing.find_timestamp(df_bm)
    # add timedelta
    df_bm["timedelta"] = (df_bm[ts_name] -
                          df_bm.loc[0, ts_name].floor("D")) / \
        pd.Timedelta(1, "D")

    with open("./chflux/tests/_assets/test-chamber.json") as f:
        chamber_spec_json = json.load(f)
    date = df_bm.loc[df_bm.shape[0] // 2, ts_name].floor("D")
    sch = schedule.make_daily_schedule(date, chamber_spec_json)

    df_bm_labeled = schedule.label_chamber_period(
        df_bm, sch, start="schedule.start", end="schedule.end", label=label)
    df_bm_reduced = processing.reduce_dataframe(
        df_bm_labeled, by=label, lower_bound=-0.5)
    df_bm_reduced = df_bm_reduced.drop(columns=["timedelta", "chamber_id"])
    df_bm_reduced = df_bm_reduced.set_index(label, drop=True)
    df_bm_processed = sch[["id", "name", "area", "volume"]]
    df_bm_processed = df_bm_processed.join(df_bm_reduced)

    assert label not in df_bm_processed.columns
    assert "timedelta" not in df_bm_processed.columns

    biomet_vars = processing.find_biomet_vars(df_bm)
    for _, namelist in biomet_vars.items():
        for name in namelist:
            assert name in df_bm_processed.columns
