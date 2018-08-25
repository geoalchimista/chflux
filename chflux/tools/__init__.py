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

   extract_date_substr -- Extract date substring from a list of file paths.
   parse_day_number    -- Parse day number.
   parse_unix_time     -- Parse Unix time in seconds.

Unit tools
==========

Tools for unit conversion.

.. autosummary::
   :toctree: generated/

   parse_units -- Parse concentration and flux units from numerical values.

Miscellaneous tools
===================
.. autosummary::
   :toctree: generated/

   check_pkgreqs -- Check package requirements.
"""
from .datetime import *
from .units import *
from .misc import *
from .helpers import *

__all__ = [s for s in dir() if not s.startswith('_')]
