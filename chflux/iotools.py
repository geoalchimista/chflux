"""
A collection of input/output tools for PyChamberFlux

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

"""
import yaml


def load_config(filepath):
    """Load configuration file from a given filepath."""
    with open(filepath, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc_yaml:
            print(exc_yaml)
            config = {}  # return a blank dict if fail to load

    return config
