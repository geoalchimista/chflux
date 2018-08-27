"""
======================
I/O (:mod:`chflux.io`)
======================

.. currentmodule:: chflux.io

Input/output module to read and write config and data files.

Readers
=======
.. autosummary::
   :toctree: generated/

   read_json      -- Read JSON file.
   read_yaml      -- Read YAML file.
   read_tabulated -- Read tabulated data files.

Writers
=======
"""
from .readers import *

__all__ = [s for s in dir() if not s.startswith('_')]
