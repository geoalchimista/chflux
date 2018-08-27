import random

import numpy as np
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


def test_read_tabulated():
    # make a segment of the config file
    config_segment = {
        "biomet.files": "./chflux/tests/_assets/test-*.csv",
        "biomet.date_format": "%Y%m%d",
        "biomet.csv.delimiter": ",",
        "biomet.csv.header": "infer",
        "biomet.csv.names": None,
        "biomet.csv.usecols": None,
        "biomet.csv.dtype": None,
        "biomet.csv.na_values": None,
        "biomet.csv.parse_dates": [0, ],
        "biomet.timestamp.parser": None,
        "biomet.timestamp.epoch_year": None,
        "biomet.timestamp.day_of_year_ref": None,
        "biomet.timestamp.is_UTC": True,
    }
    df_biomet = readers.read_tabulated('biomet', config_segment)
    assert df_biomet.shape[0] > 0
    assert df_biomet['timestamp'].dtype.type == np.datetime64
