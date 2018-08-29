"""The Process class for PyChamberFlux."""
import argparse
import copy
import datetime
import os
import warnings
from typing import Dict, Optional

from chflux.config.default_config import default_config
from chflux.config.utils import validate_config, validate_chamber
from chflux.exceptions import *
from chflux.io import read_json, read_tabulated, read_yaml, write_config
from chflux.tools import check_pkgreqs, timestamp_parsers, update_dict


class ChFluxProcess(object):
    """
    PyChamberFlux Process class that interacts with the client and executes
    calculations.
    """
    _description = 'PyChamberFlux: A command-line tool for dynamic chamber ' \
        'flux calculation.'  # command-line description

    def __init__(self, args=None):
        """Initialize a PyChamberFlux Process."""
        self._init_argparser()
        self._add_arguments()
        self._args = self._argparser.parse_args(args)

    # == arg parser ==

    def _init_argparser(self):
        """Initialize a parser for command line arguments."""
        self._argparser = argparse.ArgumentParser(
            description=self._description)

    def _add_arguments(self):
        """Add arguments parsing rules to the arg-parser."""
        self._argparser.add_argument(
            '-c', '--config', dest='config', action='store',
            help='set the configuration file for the run')

    # == timer ==

    def _set_time_start(self):
        self._time_start = datetime.datetime.utcnow()

    def _set_time_end(self):
        self._time_end = datetime.datetime.utcnow()

    def _get_time_start(self):
        return self._time_start

    def _get_time_end(self):
        return self._time_end

    time_start = property(_get_time_start,  # type: ignore
                          _set_time_start,
                          doc='Start time of the last session (UTC).')
    time_end = property(_get_time_end,  # type: ignore
                        _set_time_end,
                        doc='End time of the last session (UTC).')

    @property
    def time_lapsed(self):
        """Time spent in running the last session (s)."""
        return (self._time_end - self._time_start).total_seconds()

    # == config ==

    def _set_config(self, echo=True):
        """Set the configuration for the run."""
        if isinstance(self._args.config, str):
            if echo:
                print(f'Config file is set to {self._args.config}.')
            ext = os.path.splitext(self._args.config)[1]  # get extension
            read_config = read_json if ext == '.json' else read_yaml
            try:
                user_config = read_config(self._args.config)
            except FileNotFoundError as err:
                print(f'{err}\nConfig file not found! Aborted.')
                exit(1)
            if user_config is not None:
                self._config = update_dict(default_config, user_config)
            else:
                raise ConfigParsingException('Cannot parse the config file!')
        else:
            warnings.warn('No config file is set!', UserWarning)
            self._config = copy.deepcopy(default_config)

    def _get_config(self):
        """Get the configuration for the run."""
        return self._config

    def _update_config(self, updater: Dict):
        """Update the configuration using an updater dict."""
        self._config = update_dict(self._config, updater)

    def _check_config(self):
        """Sanity-check the configuration."""
        validate_config(self._config)

    def _save_config(self, path: Optional[str] = None):
        """Save the configuration to a file."""
        if path is None:
            path = self._config['run.output.path']
        write_config(self._config, path)  # type: ignore
        print(f'Config file saved to {path}/config/config.json.')

    # clients are allowed to access the config, update it by keys, save it,
    # and do a sanity check on it.
    config = property(_get_config, doc="Configuration for the run.")
    update_config = _update_config
    check_config = _check_config
    save_config = _save_config

    # == chamber specifications ==
    # note: these methods can only be invoked after `self._config` is set.

    def _set_chamber(self, echo=True):
        """Set the chamber specifications."""
        if echo:
            print(
                f"Chamber spec file is set to {self._config['run.chamber']}.")
        ext = os.path.splitext(self._config['run.chamber'])[1]  # get extension
        read_chamber = read_json if ext == '.json' else read_yaml
        try:
            chamber = read_chamber(self._config['run.chamber'])
        except FileNotFoundError as err:
            print(f'{err}\nChamber spec file not found! Aborted.')
            exit(1)
        if chamber is not None:
            self._chamber = chamber
        else:
            raise ConfigParsingException('Cannot parse the chamber spec file!')

    def _get_chamber(self):
        """Get the chamber specifications."""
        return self._chamber

    def _check_chamber(self):
        """Validate the chamber specifications."""
        validate_config(self._chamber)

    def _save_chamber(self, path: Optional[str] = None):
        """Save the chamber specifications to a file."""
        if path is None:
            path = self._config['run.output.path']
        write_config(self._chamber, path, is_chamber=True)  # type: ignore
        print(f'Chamber spec file saved to {path}/config/chamber.json.')

    # clients are allowed to access, validate, or save chamber specifications
    chamber = property(_get_chamber, doc='Chamber specifications.')
    check_chamber = _check_chamber
    save_chamber = _save_chamber

    # == logistics ==

    _check_pkgreqs = staticmethod(check_pkgreqs)  # check package requirements

    def _filter_warnings(self):  # NOTE: NOT IMPLEMENTED
        raise NotImplementedError('Filter not implemented!')

    def _make_output_dirs(self):
        output_dir = self._config['run.output.path']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(f'{output_dir}/flux'):
            os.makedirs(f'{output_dir}/flux')
        if (self._config['run.save.curvefit'] and
                not os.path.exists(f'{output_dir}/diag')):
            os.makedirs(f'{output_dir}/diag')
        if (self._config['run.save.config'] and
                not os.path.exists(f'{output_dir}/config')):
            os.makedirs(f'{output_dir}/config')
        if (self._config['plot.save.curvefit'] and
                not os.path.exists(f'{output_dir}/plots/curvefit')):
            os.makedirs(f'{output_dir}/plots/curvefit')
        if (self._config['plot.save.daily'] and
                not os.path.exists(f'{output_dir}/plots/daily')):
            os.makedirs(f'{output_dir}/plots/daily')

    # # == I/O ==  # TODO: rewrite

    # def _set_dataframe(self, name):
    #     pass

    # def _get_dataframe(self, name):
    #     pass

    # def _del_dataframe(self, name):
    #     pass

    # def _attach_dataframes(self, date):
    #     pass

    # def _detach_dataframes(self, date):
    #     pass

    # def _save_data(self):
    #     """Save processed data to CSV files."""
    #     # if no data attached to this 'run'; raise warning and pass
    #     pass

    # save_data = _save_data

    # == the runner ==

    def _calculate(self):
        pass

    def run(self):
        """Run the process."""

        self._set_time_start()  # set the start time of the current session
        print('PyChamberFlux started.\nStarting data processing at %s ...' %
              datetime.datetime.strftime(self.time_start, '%Y-%m-%d %X UTC'))

        # == check pkg environment ==
        try:
            self._check_pkgreqs()
        except ModuleNotFoundError as err:
            print(f'{err}\nMissing required packages. Aborted.')
            exit(1)

        # == config ==
        self._set_config()  # read config file and set config
        self._check_config()  # sanity check

        # == chamber spec ==
        self._set_chamber()
        self._check_config()

        # == logistics ==
        self._make_output_dirs()

        # calculation and output
        self._calculate()
        if self._config['run.save.config']:
            self._save_config()
            self._save_chamber()

        self._set_time_end()  # record the end time of the current session
        print('Done. Finished in %.2f seconds.' % self.time_lapsed)
