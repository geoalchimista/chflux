import collections
import copy
from typing import Dict, List


def filter_substr(strlst: List[str], substr: str) -> List[str]:
    """Filter a list of string by a substring."""
    return list(filter(lambda s: substr in s, strlst))


def flatten_dict(d: Dict) -> Dict:
    """Flatten a nested dictionary."""
    # Retrived from [1]. Solution by Winston Ewert.
    # [1] https://codereview.stackexchange.com/questions/21033/flatten-dictionary-in-python-functional-style  # noqa
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
    Return a dict from updating the keys of `dct` with the `updater` dict.
    Both can be nested dicts of arbitrary depth. The original dict is kept
    unchanged.
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
