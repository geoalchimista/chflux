"""
PyChamberFlux main script

A package for calculating trace gas fluxes from chamber measurements
"""
import argparse
import copy
import datetime
import os

from chflux.config.default_config import default_config
from chflux.io import *
from chflux.tools import check_pkgreqs


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
                          doc="Start time of the current session.")
    time_end = property(_get_time_end, _set_time_end,
                        doc="End time of the current session.")

    @property
    def time_lapsed(self):
        """Time spent on the data processing."""
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

    def _save_config(self, path=None):
        """Save the configuration to a file."""
        pass
        # output_dir = self._config['data_dir']['output_dir']
        # f_usercfg = output_dir + '/config/user_config.yaml'
        # f_chcfg = output_dir + '/config/chamber.yaml'
        # with open(usercfg_filename, 'w') as f:
        #     yaml.dump(config, f, default_flow_style=False,
        #               allow_unicode=True, indent=4)
        #     print('Configuration file saved to %s' % usercfg_filename)
        # with open(run_options['chamber_config_filepath'], 'r') as fsrc, \
        #         open(chcfg_filename, 'w') as fdest:
        #     fdest.write(fsrc.read())
        #     print('Chamber setting file saved to %s' % chcfg_filename)

    def _check_config(self):
        """Sanity-check the configuration."""
        if len(self._config['species.list']) < 1:
            raise ConfigException('No gas species is defined in the config.')
        # @TODO: more rules need to be added

    # clients are allowed to access the config, update it by keys, save it,
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

    # == methods for I/O ==

    def _set_dataframe(self, name):
        pass

    def _get_dataframe(self, name):
        pass

    def _del_dataframe(self, name):
        pass

    def _attach_dataframes(self, date):
        pass

    def _detach_dataframes(self, date):
        pass

    def _make_output_dirs(self):
        output_dir = self._config['run.output.path']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if (self._config['run.save.curvefit'] and
                not os.path.exists(output_dir + '/diag')):
            os.makedirs(output_dir + '/diag')
        if (self._config['run.save.config'] and
                not os.path.exists(output_dir + '/config')):
            os.makedirs(output_dir + '/config')

    def _save_data(self):
        """Save processed data to CSV files."""
        # if no data attached to this 'run'; raise warning and pass
        pass

    save_data = _save_data

    def filter_warnings(self):
        pass

    # == flux calculator ==

    def _calculate_flux(self):
        # a wrapper around more complicated functions in `flux_calc.py`
        pass

    def _calculate(self):
        pass

    def run(self):
        """Run the process."""
        # Note: this function is a wrapper around a series actions on the data.
        # The function itself does not manipulate the data. This is intended
        # for keeping the main program on top of all actions.

        self._set_time_start()  # set the start time of the current session
        print('PyChamberFlux\nStarting data processing at %s ...' %
              datetime.datetime.strftime(self.time_start, '%Y-%m-%d %X UTC'))
        self._check_pkgreqs()  # check versions of required python packages

        self._set_config()  # read config file and set config property
        self._check_config()  # sanity check

        # @TODO: load chamber descriptions with read_yaml()
        # @TODO: add sanity check for self._config, if not pass, raise Error
        self._make_output_dirs()
        self._save_config('filepath to save')
        self._calculate()

        self._set_time_end()  # record the end time of the current session
        print('Done. Finished in %.2f seconds.' % self.time_lapsed)
