from typing import Dict

import jsonschema

from chflux.config.chamber_schema import chamber_schema
from chflux.config.config_schema import config_schema
from chflux.exceptions import ConfigValidationException

__all__ = ['validate_config', 'validate_chamber']


def validate_config(config: Dict) -> None:
    """Sanity-check the configuration."""
    # note: this validation only checks simple cases. Still need exception
    # handling in the downstream.
    try:
        jsonschema.validate(config, config_schema)
    except jsonschema.ValidationError as err:
        raise ConfigValidationException(
            f'Config file did not pass validation!\nValidationError: {err}')
    # check gas species related keys
    if not config['species.list']:
        raise ConfigValidationException(
            "No gas species found in the key 'species.list'!")
    for s in config['species.list']:
        if f'species.{s}' not in config:
            raise ConfigValidationException(
                f"Missing gas species '{s}' in the key 'species.{s}'!")
    if config['flux.timelag_species'] not in config['species.list']:
        raise ConfigValidationException(
            "Specified 'flux.timelag_species' does not exist"
            "in the key 'species.list'!")


def validate_chamber(chamber: Dict) -> None:
    """Validate the chamber specifications against the schema."""
    try:
        jsonschema.validate(chamber, chamber_schema)
    except jsonschema.ValidationError as err:
        raise ConfigValidationException(
            f'Chamber spec file did not pass validation!\n'
            f'ValidationError: {err}')
