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

   read_yaml      -- Read YAML file.
   read_json      -- Read JSON file.
   read_tabulated -- Read tabulated data files.

Writers
=======
.. autosummary::
   :toctree: generated/

   write_tabulated -- Write tabulated data files.
   write_config    -- Write configuration or chamber specifications to files.
"""
from .readers import read_yaml, read_json, read_tabulated
from .writers import write_tabulated, write_config

__all__ = [s for s in dir() if not s.startswith('_')]
