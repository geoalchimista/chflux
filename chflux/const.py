"""
===============================
Constants (:mod:`chflux.const`)
===============================

.. currentmodule:: chflux.const

Physical constants.

====================  =========================================================
``T_0``               zero Celsius in Kelvin
``p_std``             standard atmospheric pressure [Pa]
``R_gas``             molar gas constant [J mol\ :sup:`-1`\  K\ :sup:`-1`\ ]
``air_conc_stp``      air concentration at STP condition [mol m\ :sup:`-3`\ ]
====================  =========================================================
"""
from scipy.constants import constants as _constants


__all__ = ['T_0', 'p_std', 'R_gas', 'air_conc_stp']


# zero Celsius in Kelvin
T_0 = _constants.zero_Celsius

# standard atmospheric pressure [Pa]
p_std = _constants.atm

# molar gas constant [J mol^-1 K^-1]
R_gas = _constants.R

# air concentration at STP condition [mol m^-3]
air_conc_stp = _constants.atm / (_constants.R * _constants.zero_Celsius)
