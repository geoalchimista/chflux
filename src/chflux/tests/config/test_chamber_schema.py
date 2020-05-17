import json

import jsonschema
import pytest
from jsonschema.exceptions import ValidationError

from chflux.config import chamber_schema

# set up the test data

chamber_1 = {
    "id": 1,
    "name": "SC1",
    "is_leaf_chamber": False,
    "area": 0.03,
    "volume": 0.01,
    "sensors_id": {
        "temperature": 1,
        "PAR": 2,
        "flowmeter": 1
    },
    "schedule": {
        "start": 0.,
        "bypass_before.start": 0.5,
        "bypass_before.end": 2.5,
        "chamber.start": 3.,
        "chamber.end": 9.,
        "bypass_after.start": 9.5,
        "bypass_after.end": 12.5,
        "end": 12.5
    },
    "timelag": {
        "optimize": False,
        "nominal": 0.5,
        "upper_limit": 1.5,
        "lower_limit": 0.,
    }
}


chamber_2 = {
    "id": 2,
    "name": "SC2",
    "is_leaf_chamber": False,
    "area": 0.04,
    "volume": 0.015,
    "sensors_id": {
        "temperature": 2,
        "PAR": 1,
        "flowmeter": 2
    },
    "schedule": {
        "start": 13.,
        "bypass_before.start": 0.5,
        "bypass_before.end": 2.5,
        "chamber.start": 3.,
        "chamber.end": 9.,
        "bypass_after.start": 9.5,
        "bypass_after.end": 12.5,
        "end": 12.5
    },
    "timelag": {
        "optimize": True,
        "nominal": 0.5,
        "upper_limit": 1.5,
        "lower_limit": 0.,
    }
}


chamber_3 = {
    "id": "3",  # this should NOT raise an exception
    "name": "SC3",
    "is_leaf_chamber": "false",  # this should raise an exception
    "area": 0.04,
    "volume": 0.015,
    "sensors_id": {
        "temperature": 2,
        "PAR": 1,
        "flowmeter": 2
    },
    # "schedule": {  # <- the lack of "schedule" may raise an exception
    #     "start": 13.,
    #     "bypass_before.start": 0.5,
    #     "bypass_before.end": 2.5,
    #     "chamber.start": 3.,
    #     "chamber.end": 9.,
    #     "bypass_after.start": 9.5,
    #     "bypass_after.end": 12.5,
    #     "end": 12.5
    # },
    "timelag": {
        "optimize": True,
        "nominal": 0.5,
        "upper_limit": 1.5,
        "lower_limit": 0.,
    }
}


experiment_1 = {
    "start": "2018-05-17",
    "end": "2018-05-18",
    "unit_of_time": "minute",
    "cycle_length": 30.,
    "n_chambers": 2,
    "n_cycles_per_day": 48,
    "chambers": [chamber_1, chamber_2]
}


experiment_2 = {
    "start": "2018-05-18",
    "end": "2018-05-19",
    "unit_of_time": "minute",
    "cycle_length": 30.,
    "n_chambers": 2,
    "n_cycles_per_day": 48,
    "chambers": [chamber_1, chamber_3]  # this should raise an exception
}


experiment_3 = {
    "start": "2018-05-17",
    "end": "2018-05-18",
    "unit_of_time": "week",  # a deliberate mistake to raise an exception
    "cycle_length": 30.,
    "n_chambers": 2,
    "n_cycles_per_day": 48,
    "chambers": [chamber_1, chamber_2]
}


def test_chamber_schema():
    jsonschema.validate(chamber_1, chamber_schema.chamber_schema)
    jsonschema.validate(chamber_2, chamber_schema.chamber_schema)
    with pytest.raises(ValidationError):
        jsonschema.validate(chamber_3, chamber_schema.chamber_schema)


def test_experiment_schema():
    jsonschema.validate(experiment_1, chamber_schema.experiment_schema)
    with pytest.raises(ValidationError):
        jsonschema.validate(experiment_2, chamber_schema.experiment_schema)
        jsonschema.validate(experiment_3, chamber_schema.experiment_schema)


def test_chamber_schema_json():
    with open("./chflux/tests/_assets/test-chamber.json") as f:
        chamber_spec_json = json.load(f)

    for key in chamber_spec_json:
        jsonschema.validate(chamber_spec_json[key],
                            chamber_schema.experiment_schema)
        for item in chamber_spec_json[key]["chambers"]:
            jsonschema.validate(item, chamber_schema.chamber_schema)
