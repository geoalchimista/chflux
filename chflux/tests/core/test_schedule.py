import json
import math

import pandas as pd
import pytest

from chflux.core import schedule


def test_time_unit_conversion_factor():
    for unitname in ["s", "sec", "second", "seconds"]:
        assert math.isclose(1.0, schedule.time_unit_conversion_factor(
            unitname) * 60. * 60. * 24.)
    for unitname in ["m", "min", "minute", "minutes"]:
        assert math.isclose(1.0, schedule.time_unit_conversion_factor(
            unitname) * 60. * 24.)
    for unitname in ["h", "hr", "hour", "hours"]:
        assert math.isclose(1.0, schedule.time_unit_conversion_factor(
            unitname) * 24.)
    for unitname in ["d", "day", "days"]:
        assert math.isclose(1.0,
                            schedule.time_unit_conversion_factor(unitname))
    # edge test: no conversion for illegal string or input types
    assert math.isclose(1.0, schedule.time_unit_conversion_factor("null"))
    assert math.isclose(1.0, schedule.time_unit_conversion_factor(42))


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


def test_label_chamber_period():  # TODO
    pass
