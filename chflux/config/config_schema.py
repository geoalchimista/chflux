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

species_schema = {
    "type": "object",
    "properties": {
        "unit": {"type": "number"},
        "output_unit": {"type": "number"},
        "multiplier": {"type": "number"},
        "baseline_correction": {
            "type": "string",
            "enum": ["mean", "median"]
        }
    }
}


config_schema = {
    "type": "object",
    "definitions": {},
    "$schema": "http://json-schema.org/draft-07/schema#",
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
            "default": 1
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
        "biomet.files": {
            "type": "string",
            "default": ""
        },
        "biomet.date_format": {
            "type": "string",
            "default": "%Y%m%d",
            "enum": ["%Y%m%d", "%Y_%m_%d", "%Y-%m-%d",
                     "%y%m%d", "%y_%m_%d", "%y-%m-%d"]
        },
        "biomet.csv.delimiter": {
            "type": ["string", "null"],
            "default": ","
        },
        "biomet.csv.header": {
            "type": ["string", "integer", "null"],
            "default": "infer"
        },
        "biomet.csv.names": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        "biomet.csv.usecols": {
            "anyOf": [
                {"type": "null"},
                int_list
            ],
            "default": None
        },
        "biomet.csv.dtype": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        "biomet.csv.na_values": {
            "anyOf": [
                {"type": ["string", "null"]},
                str_list
            ],
            "default": None
        },
        "biomet.csv.parse_dates": {
            "anyOf": [
                {"type": ["boolean", "integer"]},
                int_list
            ],
            "default": False
        },
        "biomet.timestamp.parser": {
            "type": ["string", "null"],
            "default": None
        },
        "biomet.timestamp.epoch_year": {
            "type": ["integer", "null"],
            "default": None
        },
        "biomet.timestamp.day_of_year_ref": {
            "type": ["integer", "null"],
            "default": None
        },
        "biomet.timestamp.is_UTC": {
            "type": "boolean",
            "default": True
        },
        "concentration.files": {
            "type": ["null", "string"],
            "default": ""
        },
        "concentration.date_format": {
            "type": "string",
            "default": "%Y%m%d",
            "enum": ["%Y%m%d", "%Y_%m_%d", "%Y-%m-%d",
                     "%y%m%d", "%y_%m_%d", "%y-%m-%d"]
        },
        "concentration.csv.delimiter": {
            "type": ["string", "null"],
            "default": ","
        },
        "concentration.csv.header": {
            "type": ["string", "integer", "null"],
            "default": "infer"
        },
        "concentration.csv.names": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        "concentration.csv.usecols": {
            "anyOf": [
                {"type": "null"},
                int_list
            ],
            "default": None
        },
        "concentration.csv.dtype": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        "concentration.csv.na_values": {
            "anyOf": [
                {"type": ["string", "null"]},
                str_list
            ],
            "default": None
        },
        "concentration.csv.parse_dates": {
            "anyOf": [
                {"type": ["boolean", "integer"]},
                int_list
            ],
            "default": False
        },
        "concentration.timestamp.parser": {
            "type": ["string", "null"],
            "default": None
        },
        "concentration.timestamp.epoch_year": {
            "type": ["integer", "null"],
            "default": None
        },
        "concentration.timestamp.day_of_year_ref": {
            "type": ["integer", "null"],
            "default": None
        },
        "concentration.timestamp.is_UTC": {
            "type": "boolean",
            "default": True
        },
        "flow.files": {
            "type": ["null", "string"],
            "default": ""
        },
        "flow.date_format": {
            "type": "string",
            "default": "%Y%m%d",
            "enum": ["%Y%m%d", "%Y_%m_%d", "%Y-%m-%d",
                     "%y%m%d", "%y_%m_%d", "%y-%m-%d"]
        },
        "flow.is_STP": {
            "type": "boolean",
            "default": True
        },
        "flow.csv.delimiter": {
            "type": ["string", "null"],
            "default": ","
        },
        "flow.csv.header": {
            "type": ["string", "integer", "null"],
            "default": "infer"
        },
        "flow.csv.names": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        "flow.csv.usecols": {
            "anyOf": [
                {"type": "null"},
                int_list
            ],
            "default": None
        },
        "flow.csv.dtype": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        "flow.csv.na_values": {
            "anyOf": [
                {"type": ["string", "null"]},
                str_list
            ],
            "default": None
        },
        "flow.csv.parse_dates": {
            "anyOf": [
                {"type": ["boolean", "integer"]},
                int_list
            ],
            "default": False
        },
        "flow.timestamp.parser": {
            "type": ["string", "null"],
            "default": None
        },
        "flow.timestamp.epoch_year": {
            "type": ["integer", "null"],
            "default": None
        },
        "flow.timestamp.day_of_year_ref": {
            "type": ["integer", "null"],
            "default": None
        },
        "flow.timestamp.is_UTC": {
            "type": "boolean",
            "default": True
        },
        "leaf.files": {
            "type": ["null", "string"],
            "default": ""
        },
        "leaf.date_format": {
            "type": "string",
            "default": "%Y%m%d",
            "enum": ["%Y%m%d", "%Y_%m_%d", "%Y-%m-%d",
                     "%y%m%d", "%y_%m_%d", "%y-%m-%d"]
        },
        "leaf.csv.delimiter": {
            "type": ["string", "null"],
            "default": ","
        },
        "leaf.csv.header": {
            "type": ["string", "integer", "null"],
            "default": "infer"
        },
        "leaf.csv.names": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        "leaf.csv.usecols": {
            "anyOf": [
                {"type": "null"},
                int_list
            ],
            "default": None
        },
        "leaf.csv.dtype": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        "leaf.csv.na_values": {
            "anyOf": [
                {"type": ["string", "null"]},
                str_list
            ],
            "default": None
        },
        "leaf.csv.parse_dates": {
            "anyOf": [
                {"type": ["boolean", "integer"]},
                int_list
            ],
            "default": False
        },
        "leaf.timestamp.parser": {
            "type": ["string", "null"],
            "default": None
        },
        "leaf.timestamp.epoch_year": {
            "type": ["integer", "null"],
            "default": None
        },
        "leaf.timestamp.day_of_year_ref": {
            "type": ["integer", "null"],
            "default": None
        },
        "leaf.timestamp.is_UTC": {
            "type": "boolean",
            "default": True
        },
        "timelag.files": {
            "type": ["null", "string"],
            "default": ""
        },
        "timelag.date_format": {
            "type": "string",
            "default": "%Y%m%d",
            "enum": ["%Y%m%d", "%Y_%m_%d", "%Y-%m-%d",
                     "%y%m%d", "%y_%m_%d", "%y-%m-%d"]
        },
        "timelag.csv.delimiter": {
            "type": ["string", "null"],
            "default": ","
        },
        "timelag.csv.header": {
            "type": ["string", "integer", "null"],
            "default": "infer"
        },
        "timelag.csv.names": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        "timelag.csv.usecols": {
            "anyOf": [
                {"type": "null"},
                int_list
            ],
            "default": None
        },
        "timelag.csv.dtype": {
            "anyOf": [
                {"type": "null"},
                str_list
            ],
            "default": None
        },
        "timelag.csv.na_values": {
            "anyOf": [
                {"type": ["string", "null"]},
                str_list
            ],
            "default": None
        },
        "timelag.csv.parse_dates": {
            "anyOf": [
                {"type": ["boolean", "integer"]},
                int_list
            ],
            "default": False
        },
        "timelag.timestamp.parser": {
            "type": ["string", "null"],
            "default": None
        },
        "timelag.timestamp.epoch_year": {
            "type": ["integer", "null"],
            "default": None
        },
        "timelag.timestamp.day_of_year_ref": {
            "type": ["integer", "null"],
            "default": None
        },
        "timelag.timestamp.is_UTC": {
            "type": "boolean",
            "default": True
        },
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
        "species.list": {
            "type": "array",
            "items": {"type": "string"}
        },
        "species.names": {
            "type": "array",
            "items": {"type": "string"}
        },
        "species.h2o": species_schema,
        "species.co2": species_schema,
        "species.cos": species_schema,
        "species.co": species_schema,
        "species.ch4": species_schema,
        "species.n2o": species_schema
    }
}
