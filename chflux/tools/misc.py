"""Miscellaneous tools."""
import importlib

import pkg_resources

__all__ = ['check_pkgreqs']


def check_pkgreqs(echo: bool = True) -> None:
    """Check package requirements."""
    # required packages
    pkg_list = ['numpy', 'pandas', 'scipy', 'matplotlib', 'yaml', 'jsonschema']
    # check existence and print versions (if echo is enabled)
    pkg_specs = {pkg: importlib.util.find_spec(pkg) for pkg in pkg_list}
    if echo:
        print('Checking specifications of required packages:')
    for pkg in pkg_specs:
        if pkg_specs[pkg] is None:
            raise ModuleNotFoundError("Required package '%s' not found" % pkg)
        elif echo:
            # package and import names differ for pyyaml
            if pkg == 'yaml':
                pkg = 'pyyaml'
            print('  %s = %s' %
                  (pkg, pkg_resources.get_distribution(pkg).version))
