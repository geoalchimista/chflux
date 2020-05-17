r"""
====================================
Constants (:mod:`chflux.core.const`)
====================================

.. currentmodule:: chflux.core.const

=====================  ========================================================
``T_0``                zero Celsius in Kelvin
``atm``                standard atmospheric pressure [Pa]
``R_gas``              molar gas constant [J mol\ :sup:`-1`\  K\ :sup:`-1`\ ]
``air_concentration``  air concentration at STP condition [mol m\ :sup:`-3`\ ]
=====================  ========================================================
"""

__all__ = ['T_0', 'atm', 'R_gas', 'air_concentration']

T_0: float = 273.15
atm: float = 101325.0
R_gas: float = 8.3144598
air_concentration: float = atm / (R_gas * T_0)
