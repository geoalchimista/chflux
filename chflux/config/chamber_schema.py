"""PyChamberFlux JSON Schema for chamber specifications."""

chamber_schema = {
    "type": "object",
    "properties": {
        # Chamber ID number.
        "id": {"type": "number"},

        # Chamber name.
        "name": {"type": "string"},

        # Is the chamber a leaf chamber?
        "is_leaf_chamber": {"type": "boolean"},

        # Area for gas exchange [m^2]. For a soil chamber, this could be the
        # footprint area. For a leaf chamber, this could be the total leaf area
        # in the chamber.
        "area": {"type": "number"},

        # Standard deviation of the area for gas exchange [m^2].
        "area.sd": {"type": "number"},

        # Chamber headspace volume [m^3].
        "volume": {"type": "number"},

        # Standard deviation of the chamber headspace volume [m^3].
        "volume.sd": {"type": "number"},

        # Sensor IDs to search for in the biometeorological dataframe.
        "sensors_id": {
            "type": "object",
            "properties": {
                # Temperature sensor ID
                "temperature": {"type": "number"},

                # PAR sensor ID
                "PAR": {"type": "number"},

                # Flowmeter ID
                "flowmeter": {"type": "number"}
            }
        },

        # Chamber measurement schedule. The key "start" is the start time of
        # the current _chamber line_ with respect to a _measurement cycle_,
        # which may include more than one chamber. All other keys are offsets
        # with respect to the "start".
        #
        # For example, a 60-minute measurement cycle with three chambers may
        # look like:
        #
        # ch1 [00:00--00:20] --> ch2 [00:20--00:40] --> ch3 [00:40--00:60]
        #
        # The start times of ch1, ch2, and ch3 are 0, 20, 60, respectively, if
        # the `unit_of_time` is defined as "minute" in the experiment
        # description.
        "schedule": {
            "type": "object",
            "properties": {
                # The start time of the current chamber line, referenced to the
                # start time of a measurement cycle.
                "start": {"type": "number"},

                # Note: All timestamps below are offsets with respect to the
                # start of the current chamber line, i.e., the "start" key
                # above.

                # The start and end times of ambient concentration
                # measurements.
                "ambient.start": {"type": "number"},
                "ambient.end": {"type": "number"},

                # The start and end times of the bypass line concentration
                # measurements _before_ the measurement period of chamber
                # headspace concentrations.
                "bypass_before.start": {"type": "number"},
                "bypass_before.end": {"type": "number"},

                # The start and end times of the measurement period of chamber
                # headspace concentrations.
                "chamber.start": {"type": "number"},
                "chamber.end": {"type": "number"},

                # The start and end times of the bypass line concentration
                # measurements _after_ the measurement period of chamber
                # headspace concentrations.
                "bypass_after.start": {"type": "number"},
                "bypass_after.end": {"type": "number"},

                # The end time of the current chamber line.
                "end": {"type": "number"},
            }
        },

        # Timelags between biometeorological variables and concentrations. Note
        # that the unit of the timelag is defined at the parent level of the
        # chamber settings (see `experiment_schema`).
        "timelag": {
            "type": "object",
            "properties": {
                # If `true`, invoke the optimizer for timelag optimization;
                # otherwise, do not attempt a timelag optimization.
                "optimize": {"type": "boolean"},

                # Nominal timelag that serves as the _initial guess_ of the
                # optimization. Note that this setting is ignored when using
                # prescribed timelags from input dataframes.
                "nominal": {"type": "number"},

                # Upper limit of the timelag.
                "upper_limit": {"type": "number"},

                # Lower limit of the timelag.
                "lower_limit": {"type": "number"}
            }
        }
    }
}


experiment_schema = {
    "type": "object",
    "properties": {
        # The start and end datetimes of the current experiment setting. The
        # datetime format must conform to the ISO 8601 standard.
        "start": {"type": "string"},
        "end": {"type": "string"},

        # The unit of time in chamber schedule definitions.
        "unit_of_time": {"type": "string"},

        # The duration of a chamber measurement cycle (which may include
        # multiple chambers). The unit is defined by the "unit_of_time" key.
        "cycle_length": {"type": "number"},

        # The number of chambers in a chamber measurement cycle.
        "n_chambers": {"type": "number"},

        # The number of chamber measurement cycles per day.
        "n_cycles_per_day": {"type": "number"},

        # Specifications of each chamber.
        "chambers": {
            "type": "array",
            "items": chamber_schema,
            "required": ["id", "area", "volume", "sensors_id", "schedule"],
        },
    }
}
