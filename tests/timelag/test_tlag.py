"""
Test the timelag optimization function.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from common_func import optimize_timelag

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# create a test case
time = np.arange(900.)  # a 15-minute series with 1 sec sampling interval
conc = np.zeros_like(time)  # concentration

C_atm = 395.
t_turnover = 240.  # turnover time, 240 sec
# e.g., 6 L chamber @ 1.5 L min^-1 flow rate
flux = 0.05  # arbitrary unit
area = 0.03  # arbitrary unit
flow = 1.5 * 1e-3 / 60.  # arbitrary unit
# opening period: 2 minute
conc[0:120] = C_atm + random.random(120) * 5.
# closing period: 9 minute + 75 second timelag
tlag_nom = 75  # an integer, as the index
conc[120:120 + tlag_nom] = C_atm + random.random(tlag_nom) * 5.
conc[120 + tlag_nom:660 + tlag_nom] = \
    flux * area / flow * (1 - np.exp(-np.arange(540.) / t_turnover)) + \
    C_atm + random.random(540) * 3.
conc[660 + tlag_nom:] = C_atm + random.random(conc[660 + tlag_nom:].size) * 5.

# add a linear drift for the instrument (0.1 ppmv per second)
conc += time * 0.1

timelag, status_timelag = \
    optimize_timelag(time, conc, t_turnover,
                     dt_open_before=120., dt_close=540.,
                     dt_open_after=120., closure_period_only=True,
                     bounds=(60., 180.), guess=120.)

print('Nominal timelag assigned to the test case is %d s' % tlag_nom)
print('The optimized timelag is %f s' % timelag)
print('  with optimization status code: %d ' % status_timelag)

plt.figure(figsize=(6, 3))
plt.plot(time, conc, 'k.')
plt.axvline(x=timelag, linestyle='--')
plt.axvline(x=timelag + 120., linestyle='--')
plt.show()
