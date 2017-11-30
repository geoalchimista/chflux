"""
PyChamberFlux main script

A package for calculating trace gas fluxes from chamber measurements
"""
import copy
import argparse
import datetime
import timeit
import importlib
import pkg_resources

from chflux.default_config import default_config
from chflux.io.readers import read_yaml, update_dict


class ChFluxProcess(object):
    """Start a process to run the PyChamberFlux calculations."""
    description = 'PyChamberFlux: Main program for flux calculation.'
    start_timestamp = None
    config = copy.deepcopy(default_config)

    def __init__(self, args=None):
        """Initialize a PyChamberFlux process."""
        self._init_argparser()
        self._add_arguments()
        self.args = self.argparser.parse_args(args)

    def _init_argparser(self):
        """Initialize a parser for command line arguments."""
        self.argparser = argparse.ArgumentParser(description=self.description)

    def _add_arguments(self):
        """Add arguments parsing rules to the arg-parser."""
        self.argparser.add_argument(
            '-c', '--config', dest='config', action='store',
            help='set the configuration file for the run')

    def run(self):
        """Run the process."""
        # Note: this function is a wrapper around a series actions on the data.
        # The function itself does not manipulate the data. This is intended
        # for keeping the main program on top of all actions.

        t_start = timeit.default_timer()  # start a timer for process time

        print('PyChamberFlux\nStarting data processing...')
        self.start_timestamp = datetime.datetime.utcnow()
        print(datetime.datetime.strftime(self.start_timestamp,
                                         '%Y-%m-%d %X UTC'))
        check_pkgreqs()

        self.set_config()
        # @TODO: load chamber descriptions with read_yaml()
        # @TODO: add sanity check for self.config, if not pass, raise Error
        self.make_savedirs()
        self.save_config()
        self.calc()

        t_end = timeit.default_timer()
        print('Done. Finished in %.2f seconds.' % (t_end - t_start))

    def set_config(self, echo=True):
        """Set the configuration for the run."""
        if (isinstance(self.args.config, str) or
                isinstance(self.args.config, bytes)):
            if echo:
                print("Config file is set as '%s'" % self.args.config)
            user_config = read_yaml(self.args.config)
            self.config = update_dict(self.config, user_config)

    def save_config(self):
        pass

    def make_savedirs(self):
        pass

    def calc(self):
        # a wrapper around more complicated functions in `flux_calc.py`
        pass


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


def run_instance():
    process = ChFluxProcess()
    process.run()


if __name__ == '__main__':
    run_instance()
