"""PyChamberFlux I/O module for reading YAML files and data tables."""
import collections
import copy

import yaml
import pandas as pd


def read_yaml(filepath):
    """Read a YAML file as a dict. Return empty dict if fail to read."""
    with open(filepath, 'r') as f:
        try:
            ydict = yaml.load(f)
        except yaml.YAMLError as exc_yaml:
            print(exc_yaml)
            ydict = {}  # fall back to an empty dict if fail to read

    return ydict


def update_dict(dct, updater):
    """
    (dict, dict) -> dict

    Return a dict from updating the keys of `dct` with the `updater` dict.
    Both can be nested dicts of arbitrary depth.

    Note: The original dict, `dct`, is unchanged.
    """
    def _update_dict_altered(dct, updater):
        """
        This inner function performs the updating by recursion. It will alter
        the input `dct`.
        """
        for k, v in updater.items():
            if (k in dct and isinstance(dct[k], dict) and
                    isinstance(updater[k], collections.Mapping)):
                _update_dict_altered(dct[k], updater[k])
            else:
                dct[k] = updater[k]

    dct_copy = copy.deepcopy(dct)
    _update_dict_altered(dct_copy, updater)
    return dct_copy
