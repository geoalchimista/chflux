import collections
import copy
from typing import Dict, List

__all__ = ['filter_str', 'flatten_dict', 'update_dict']


def filter_str(strlst: List[str], substr: str) -> List[str]:
    """Filter a list of strings by the existence of a substring."""
    return list(filter(lambda s: substr in s, strlst))


def flatten_dict(d: Dict) -> Dict:
    """Flatten a nested dictionary and concatenate different levels by dots."""
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + "." + subkey, subvalue
            else:
                yield key, value

    return dict(items())


def update_dict(dct: Dict, updater: Dict) -> Dict:
    """
    Return a dict from updating the keys of ``dct`` with the ``updater`` dict.
    Both can be nested dicts of arbitrary depth. The original dict is kept
    unchanged.
    """
    def _update_dict_mut(dct, updater):
        """
        This inner function performs the updating by recursion. It will mutate
        the input `dct`.
        """
        for k, _ in updater.items():
            if (k in dct and isinstance(dct[k], dict) and
                    isinstance(updater[k], collections.Mapping)):
                _update_dict_mut(dct[k], updater[k])
            else:
                dct[k] = updater[k]

    dct_copy = copy.deepcopy(dct)
    _update_dict_mut(dct_copy, updater)
    return dct_copy
