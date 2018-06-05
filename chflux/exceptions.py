"""
====================================
Constants (:mod:`chflux.exceptions`)
====================================

.. currentmodule:: chflux.exceptions

This module contains exceptions raised by chflux.
"""


class ConfigParsingException(ValueError):  # pragma: no cover
    """
    Exception raised when a configuration file cannot be parsed by a JSON
    parser or a YAML parser.
    """
    pass


class ConfigValidationException(ValueError):  # pragma: no cover
    """
    Exception raised when a configuration file does not pass a validation
    against its predefined schema.
    """
    pass


class TimestampException(ValueError):  # pragma: no cover
    """
    Expection raised when the timestamp in a dataframe is missing or cannot be
    parsed.
    """
    pass


class MissingDataFrameError(ValueError):  # pragma: no cover
    """
    Error raised when missing a required dataframe for the calculations.
    """
    pass
