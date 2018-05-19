"""PyChamberFlux JSON Schema for chamber specifications."""

# @TODO: add annotations

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

        "sensors_id": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "PAR": {"type": "number"},
                "flowmeter": {"type": "number"}
            }
        },

        "schedule": {
            "type": "object",
            "properties": {
                "start": {"type": "number"},
                "ambient.start": {"type": "number"},
                "ambient.end": {"type": "number"},
                "bypass_before.start": {"type": "number"},
                "bypass_before.end": {"type": "number"},
                "measurement.start": {"type": "number"},
                "measurement.end": {"type": "number"},
                "bypass_after.start": {"type": "number"},
                "bypass_after.end": {"type": "number"},
                "end": {"type": "number"},
            }
        },

        "timelag": {
            "type": "object",
            "properties": {
                "to_optimize": {"type": "boolean"},
                "nominal": {"type": "number"},
                "upper_limit": {"type": "number"},
                "lower_limit": {"type": "number"}
            }
        }
    }
}

experiment_schema = {
    "type": "object",
    "properties": {
        "start": {"type": "string"},
        "end": {"type": "string"},
        "unit_of_time": {"type": "string"},
        "cycle_length": {"type": "number"},
        "n_chambers": {"type": "number"},
        "n_cycles_per_day": {"type": "number"},
        "chambers": {
            "type": "array",
            "items": chamber_schema,
            "required": ["id", "area", "volume", "sensors_id", "schedule"],
        },
    }
}
