from typing import Dict


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
