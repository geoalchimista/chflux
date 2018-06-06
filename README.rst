PyChamberFlux
=============

Wu Sun (wu.sun@ucla.edu)

PyChamberFlux (``chflux``) is a Python package to calculate trace gas fluxes
from chamber enclosure measurements. Chamber enclosure is a technique routinely
used in biosphere--atmosphere gas exchange studies, for example, measuring leaf
photosynthetic carbon uptake or soil respiration.

**WARNING: Refactoring in progress.** Please use the ``master`` branch!


Refactoring tracking
--------------------
1. [ ] Command-line interface . . . [Refactoring]
2. [ ] Configuration . . . [Refactoring]

   * [X] Schema for chamber specifications
   * [X] Schema for configuration
   * [ ] Validation for configuration
   * [ ] Validation for chamber specifications

3. [ ] I/O functions . . . [Refactoring]
4. [ ] Core calculation functions . . . [Refactoring]

   * [X] Physical chemistry functions
   * [X] Stats functions
   * [ ] Flux calculators . . . [Refactoring] [Enhancement]
   * [ ] Cython optimization . . . [Enhancement]

5. [ ] Plotting functions . . . [Refactoring]
6. [ ] Timelag optimization . . . [Refactoring] [Enhancement]


Dependencies
------------
* Python >= 3.6
* NumPy >= 1.13.3
* SciPy >= 1.0.1
* pandas >= 0.22.0
* matplotlib >= 2.2.2
* jsonschema >= 2.5.0
* pyyaml >= 3.12


Installation
------------
1. Clone this git repository.

.. code-block:: bash

   git clone https://github.com/geoalchimista/chflux.git

2. Install the dependencies.
3. In the root directory of this repository, run

.. code-block:: bash

   make develop

.. end

Note: Currently only available in a development version due to refactoring.


License
-------
`BSD 3-Clause License <./LICENSE>`_


Updates
-------
See the `CHANGELOG <./CHANGELOG.rst>`_.


Documentation
-------------
[TO BE ADDED]


Contributing to PyChamberFlux
-----------------------------
[CONTRIBUTING.rst TO BE ADDED]

Please navigate to the `GitHub "issues" tab
<https://github.com/geoalchimista/chflux/issues>`_ for reporting issues and
pull requests.
