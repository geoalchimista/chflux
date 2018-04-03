"""
PyChamberFlux main script

A package for calculating trace gas fluxes from chamber measurements
"""
import copy
import argparse
import datetime
# import timeit
import importlib
import pkg_resources

from chflux.default_config import default_config
from chflux.io.readers import read_yaml, update_dict


def check_pkgreqs(echo=True):
    """Check package requirements."""
    # required packages
    pkg_list = ['numpy', 'pandas', 'scipy', 'matplotlib', 'yaml']
    # check existence and print versions (if echo is enabled)
    pkg_specs = {pkg: importlib.util.find_spec(pkg) for pkg in pkg_list}
    if echo:
        print('Checking specifications of required packages:')
    for pkg in pkg_specs:
        if pkg_specs[pkg] is None:
            raise ModuleNotFoundError("Required package '%s' not found" % pkg)
        elif echo and pkg != 'yaml':
            # do not check version on 'yaml'
            print('  %s = %s' %
                  (pkg, pkg_resources.get_distribution(pkg).version))


class ChFluxProcess(object):
    """Start a process to run the PyChamberFlux calculations."""
    # == private properties ==
    _description = 'PyChamberFlux: Main module for flux calculation.'

    # == bind external functions ==
    _check_pkgreqs = staticmethod(check_pkgreqs)

    def __init__(self, args=None):
        """Initialize a PyChamberFlux process."""

        self._init_argparser()
        self._add_arguments()
        self._args = self._argparser.parse_args(args)

    def _init_argparser(self):
        """Initialize a parser for command line arguments."""
        self._argparser = argparse.ArgumentParser(
            description=self._description)

    def _add_arguments(self):
        """Add arguments parsing rules to the arg-parser."""
        self._argparser.add_argument(
            '-c', '--config', dest='config', action='store',
            help='set the configuration file for the run')

    # == methods and properties for timing the session ==

    def _set_time_start(self):
        self._time_start = datetime.datetime.utcnow()

    def _set_time_end(self):
        self._time_end = datetime.datetime.utcnow()

    def _get_time_start(self):
        return self._time_start

    def _get_time_end(self):
        return self._time_end

    time_start = property(_get_time_start, _set_time_start,
                          doc="Start time of the current session")
    time_end = property(_get_time_end, _set_time_end,
                        doc="End time of the current session")

    @property
    def time_lapsed(self):
        return (self._time_end - self._time_start).total_seconds()

    # == methods and properties for the configuration of the run ==

    def _set_config(self, echo=True):
        """Set the configuration of the run."""
        if (isinstance(self._args.config, str) or
                isinstance(self._args.config, bytes)):
            if echo:
                print("Config file is set as '%s'" % self._args.config)
            user_config = read_yaml(self._args.config)
            self._config = update_dict(default_config, user_config)
        else:
            self._config = copy.deepcopy(default_config)

    def _get_config(self):
        """Get the configuration of the run."""
        return self._config

    def _update_config(self, updater):
        """Update the configuration of the run using an updater dict."""
        self._config = update_dict(self._config, updater)

    def _save_config(self, f):
        pass

    def _check_config(self):
        pass

    # clients are allowed to access config, update it by keys, save it to files
    # and do a sanity check on it
    config = property(_get_config, doc="Configuration of the run")
    update_config = _update_config
    check_config = _check_config
    save_config = _save_config

    # == methods and properties for chamber schedules ==

    def _set_chamber_schedule(self, echo=True):
        pass

    def _get_chamber_schedule(self):
        pass

    chamber_schedule = property(_get_chamber_schedule,
                                doc="Chamber schedule")

    def run(self):
        """Run the process."""
        # Note: this function is a wrapper around a series actions on the data.
        # The function itself does not manipulate the data. This is intended
        # for keeping the main program on top of all actions.

        self._set_time_start()  # set the start time of the current session
        print('PyChamberFlux\nStarting data processing at %s ...' %
              datetime.datetime.strftime(self.time_start, '%Y-%m-%d %X UTC'))
        self._check_pkgreqs()  # check versions of required python packages

        self._set_config()
        # @TODO: load chamber descriptions with read_yaml()
        # @TODO: add sanity check for self._config, if not pass, raise Error
        self.make_savedirs()
        self._save_config('filepath to save')
        self.calc()

        self._set_time_end()  # record the end time of the current session
        print('Done. Finished in %.2f seconds.' % self.time_lapsed)

    def make_savedirs(self):
        pass

    def calc(self):
        # a wrapper around more complicated functions in `flux_calc.py`
        pass

    def save_data(self):
        """Save processed data to CSV files."""
        # if no data attached to this 'run'; raise warning and pass
        pass

    def filter_warnings(self):
        pass


def run_instance():
    process = ChFluxProcess()
    process.run()

    # TEST if the user configuration file is set properly
    print(process.config['biomet_data_settings']['usecols'])
    print('try update the config')
    process.update_config({'biomet_data_settings': {'usecols': [0, 2]}})
    print(process.config['biomet_data_settings']['usecols'])


if __name__ == '__main__':
    run_instance()
