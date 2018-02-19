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
from scipy.constants import constants


__all__ = ['T_0', 'p_std', 'R_gas', 'air_conc_stp']


# zero Celsius in Kelvin
T_0 = constants.zero_Celsius

# standard atmospheric pressure [Pa]
p_std = constants.atm

# molar gas constant [J mol^-1 K^-1]
R_gas = constants.R

# air concentration at STP condition [mol m^-3]
air_conc_stp = constants.atm / (constants.R * constants.zero_Celsius)
