import math

from chflux import tools


def test_time_conversion_factor():
    for unitname in ["s", "sec", "second", "seconds"]:
        assert math.isclose(1.0, tools.time_conversion_factor(
            unitname) * 60. * 60. * 24.)
    for unitname in ["m", "min", "minute", "minutes"]:
        assert math.isclose(1.0, tools.time_conversion_factor(
            unitname) * 60. * 24.)
    for unitname in ["h", "hr", "hour", "hours"]:
        assert math.isclose(1.0, tools.time_conversion_factor(
            unitname) * 24.)
    for unitname in ["d", "day", "days"]:
        assert math.isclose(1.0,
                            tools.time_conversion_factor(unitname))
    # null cases: no conversion for illegal string or input types
    assert math.isclose(1.0, tools.time_conversion_factor("null"))
    assert math.isclose(1.0, tools.time_conversion_factor(42))
