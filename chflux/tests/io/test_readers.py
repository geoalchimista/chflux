# import pandas as pd

# import json
import random

import pytest

from chflux.io import readers


def test_read_yaml():
    # normal case
    chamber_spec = readers.read_yaml(
        "./chflux/tests/_assets/test-chamber.yaml")
    assert isinstance(chamber_spec, dict)
    assert len(chamber_spec) > 0

    # raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        chamber_spec_error = readers.read_yaml(
            "./chflux/tests/_assets/file-that-does-not-exist-%032x.yaml" %
            random.randrange(16 ** 32))

    # catch YAMLError
    chamber_spec_noerror = readers.read_yaml(
        "./chflux/tests/_assets/test-bad-yaml.yaml")
    assert chamber_spec_noerror is None


def test_read_json():
    # normal case
    chamber_spec = readers.read_json(
        "./chflux/tests/_assets/test-chamber.json")
    assert isinstance(chamber_spec, dict)
    assert len(chamber_spec) > 0

    # raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        chamber_spec_error = readers.read_json(
            "./chflux/tests/_assets/file-that-does-not-exist-%032x.json" %
            random.randrange(16 ** 32))

    # catch JSONDecodeError
    chamber_spec_noerror = readers.read_json(
        "./chflux/tests/_assets/test-chamber.yaml")
    assert chamber_spec_noerror is None
