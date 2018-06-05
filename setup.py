#!/usr/bin/env python
import os
import sys

from setuptools import find_packages, setup


# package metadata
NAME = 'chflux'
DESCRIPTION = 'Calculate trace gas fluxes from chamber enclosure measurements'
URL = 'https://github.com/wusunlab/chflux'
AUTHOR = 'Wu Sun'
EMAIL = 'wu.sun@ucla.edu'
LICENSE = 'BSD'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

# requirements
REQUIRED = ['numpy', 'scipy', 'pandas', 'matplotlib', 'pyyaml', 'jsonschema']


here = os.path.abspath(os.path.dirname(__file__))

# import README.rst for the long-description; must include it in MANIFEST.in!
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# add version info from _version.py
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '_version.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Atmospheric Science'
    ],
)
