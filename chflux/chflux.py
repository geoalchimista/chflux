#!usr/bin/env python
from chflux.base import ChFluxProcess


def run_new_instance():
    process = ChFluxProcess()
    process.run()


if __name__ == '__main__':
    run_new_instance()
