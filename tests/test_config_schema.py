import jsonschema
from chflux.config import config_schema
from chflux.config.default_config import default_config


def test_config_schema():
    jsonschema.validate(default_config, config_schema.config_schema)
