"""PyChamberFlux JSON Schema for configurations."""

# typedef schema
str_list = {
    "type": "array",
    "items": {"type": "string"}
}

int_list = {
    "type": "array",
    "items": {"type": "integer"}
}

positive_number = {
    "type": "number",
    "minimum": 0,
    "exclusiveMinimum": True
}

species_schema = {
    "type": "object",
    "properties": {
        "unit": positive_number,
        "output_unit": positive_number,
        "multiplier": positive_number,
        "baseline_correction": {
            "type": "string",
            "enum": ["mean", "median"]
        }
    }
}


def make_reader_args_schema(s: str) -> dict:
    """Generate a sub-schema for data-reader arguments."""
    return {
        s + ".files": {
            "type": ["null", "string"],
            "default": ""
        },
        s + ".date_format": {
            "type": "string",
            "default": "%Y%m%d",
            "enum": ["%Y%m%d", "%Y_%m_%d", "%Y-%m-%d",
                     "%y%m%d", "%y_%m_%d", "%y-%m-%d"]
        },
        s + ".csv.delimiter": {
            "type": ["string", "null"],
            "default": ","
        },
        s + ".csv.header": {
            "type": ["string", "integer", "null"],
            "default": "infer"
        },
        s + ".csv.names": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        s + ".csv.usecols": {
            "anyOf": [
                {"type": "null"},
                int_list
            ],
            "default": None
        },
        s + ".csv.dtype": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        s + ".csv.na_values": {
            "anyOf": [
                {"type": ["string", "null"]},
                str_list
            ],
            "default": None
        },
        s + ".csv.parse_dates": {
            "anyOf": [
                {"type": ["boolean", "integer"]},
                int_list
            ],
            "default": False
        },
        s + ".timestamp.parser": {
            "type": ["string", "null"],
            "default": None
        },
        s + ".timestamp.epoch_year": {
            "type": ["integer", "null"],
            "default": None
        },
        s + ".timestamp.day_of_year_ref": {
            "type": ["integer", "null"],
            "default": None
        },
        s + ".timestamp.is_UTC": {
            "type": "boolean",
            "default": True
        },
    }


config_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "definitions": {},
    "properties": {
        "run.chamber": {
            "type": "string",
            "default": "./chamber.yml"
        },
        "run.realtime": {
            "type": "boolean",
            "default": False
        },
        "run.latency": {
            "type": "integer",
            "default": 1,
            "minimum": 0,
            "exclusiveMinimum": True
        },
        "run.output.path": {
            "type": "string",
            "default": "./output/"
        },
        "run.output.prefix": {
            "type": ["string", "null"],
            "default": ""
        },
        "run.save.curvefit": {
            "type": "boolean",
            "default": True
        },
        "run.save.config": {
            "type": "boolean",
            "default": True
        },
        "plot.style": {
            "type": "string",
            "default": "seaborn-darkgrid"
        },
        "plot.path": {
            "type": "string",
            "default": "./plots/"
        },
        "plot.save.curvefit": {
            "type": "boolean",
            "default": False
        },
        "plot.save.daily": {
            "type": "boolean",
            "default": False
        },
        **make_reader_args_schema("biomet"),
        **make_reader_args_schema("concentration"),
        "flow.is_STP": {
            "type": "boolean",
            "default": True
        },
        **make_reader_args_schema("flow"),
        **make_reader_args_schema("leaf"),
        **make_reader_args_schema("timelag"),
        "flux.curvefit.linear": {
            "type": "boolean",
            "default": True
        },
        "flux.curvefit.robust_linear": {
            "type": "boolean",
            "default": True
        },
        "flux.curvefit.nonlinear": {
            "type": "boolean",
            "default": True
        },
        "flux.timelag_method": {
            "anyOf": [
                {"type": "null"},
                {"type": "string",
                 "enum": ["none", "optimize", "prescribed"]}
            ],
            "default": None,
        },
        "flux.timelag_species": {
            "type": "string",
            "default": "co2"
        },
        "site.pressure": {
            "type": ["null", "number"],
            "default": None
        },
        "site.timezone": {
            "type": "integer",
            "default": 0
        },
        "species.list": str_list,
        "species.names": str_list,
        "species.h2o": species_schema,
        "species.co2": species_schema,
        "species.cos": species_schema,
        "species.co": species_schema,
        "species.ch4": species_schema,
        "species.n2o": species_schema
    }
}
