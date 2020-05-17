"""
===========================
Tools (:mod:`chflux.tools`)
===========================

.. currentmodule:: chflux.tools

Miscellaneous tools to deal with datetime and collection types and the package
environment.

Datetime tools
==============
Tools to deal with datetime parsing and conversion.

Parsers
-------
.. data:: chflux.tools.timestamp_parsers

   A dictionary of timestamp parsers for multiple ISO 8601 (year-month-day)
   date and datetime formats; for use with ``pandas.read_csv`` only.

Functions
---------
.. autosummary::
   :toctree: generated/

   extract_date_str -- Extract date substring from a list of file paths.
   parse_day_number -- Parse day number.
   parse_unix_time  -- Parse Unix time in seconds.
   timedelta_days   -- Calculate timedelta w.r.t. a reference timestamp.

Unit tools
==========
Tools for unit conversion and representation.

.. autosummary::
   :toctree: generated/

   parse_concentration_units -- Parse concentration and flux units.
   time_conversion_factor    -- Return a factor to convert time units to days.

Collection tools
================
Helper tools to deal with collection types.

.. autosummary::
   :toctree: generated/

   filter_str   -- Filter a list of strings by a substring.
   flatten_dict -- Flatten a nested dictionary.
   update_dict  -- Recursively updated a nested dictionary.

Miscellaneous tools
===================
.. autosummary::
   :toctree: generated/

   check_pkgreqs -- Check package requirements.
"""
from .datetime import *
from .units import *
from .colle import *
from .misc import *

__all__ = [s for s in dir() if not s.startswith('_')]
