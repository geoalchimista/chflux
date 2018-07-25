import pandas as pd

from chflux.core import biomet


def test_find_timestamp():
    df_bm = pd.read_csv("./chflux/tests/_assets/test-input-biomet.csv",
                        parse_dates=[0, ])

    assert biomet.find_timestamp(df_bm) == "timestamp"

    assert biomet.find_timestamp(
        df_bm.rename(columns={"timestamp": "datetime"})
    ) == "datetime"

    assert biomet.find_timestamp(
        df_bm.rename(columns={"timestamp": "timestamp_utc"})
    ) == "timestamp_utc"

    # test utc=False cases
    assert biomet.find_timestamp(df_bm, utc=False) == "timestamp"
    assert biomet.find_timestamp(
        df_bm.rename(columns={"timestamp": "datetime"}),
        utc=False) == "datetime"

    # test multiple timestamps
    df_bm["timestamp_local"] = df_bm["timestamp"]
    assert biomet.find_timestamp(df_bm) == "timestamp"
    assert biomet.find_timestamp(df_bm, utc=False) == "timestamp_local"


def test_find_biomet_vars():
    df_bm = pd.read_csv("./chflux/tests/_assets/test-input-biomet.csv",
                        parse_dates=[0, ])

    biomet_vars = biomet.find_biomet_vars(df_bm)
    assert isinstance(biomet_vars, dict)
    for _, v in biomet_vars.items():
        assert isinstance(v, list)

    # a null case
    df_null = pd.DataFrame({"A": range(100)})
    null_biomet_vars = biomet.find_biomet_vars(df_null)
    assert isinstance(null_biomet_vars, dict)
    for _, v in null_biomet_vars.items():
        assert v == []
