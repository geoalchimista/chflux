import json
from typing import Dict, Optional

import pandas as pd

__all__ = ['write_tabulated', 'write_config']


def write_tabulated(df: pd.DataFrame, config: Dict,
                    is_diag: bool = False,
                    decimals: Optional[int] = None) -> None:
    """
    Write the processed dataframe to tabulated files.

    Parameters
    ---------
    df : pandas.DataFrame
        Dataframe of the processed flux data or the curvefit diagnostics.
    config : dict
        Configuration dictionary parsed from the config file.
    is_diag : bool, optional
        If ``True``, export ``df`` to the curvefit diagnostics folder. Default
        is ``False``.
    decimals : int, optional
        Round all numeric variables to a certain number of decimals. The
        default behavior is no rounding.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``decimals`` is a negative number.
    """
    # perform rounding
    if decimals is not None:
        if decimals < 0:
            raise ValueError(
                'Number of decimals to round to must not be negative!')
        df = df.round({
            k: decimals for k in df.columns
            if k not in ['timestamp', 'name', 'id', 'is_leaf_chamber',
                         'area', 'volume']
            and pd.api.types.is_numeric_dtype(df[k])})
    # set output file name
    date_str = df.loc[df.shape[0] // 2,
                      'timestamp'].floor('D').strftime('%Y%m%d')
    label = 'diag' if is_diag else 'flux'
    if len(config['run.output.prefix']) > 0:
        fo = f"{config['run.output.prefix']}_{label}_{date_str}"
    else:
        fo = f"{label}_{date_str}"
    fo = f"{config['run.output.path']}/{label}/{fo}.csv"
    df.to_csv(fo, sep=',', na_rep='NaN', index=False)


def write_config(config: Dict, path: str, is_chamber: bool = False) -> None:
    """
    Write configuration or chamber specification to files.

    Parameters
    ---------
    config : dict
        Configuration to be saved to a JSON file.
    path : str
        Output path.
    is_chamber : bool, optional
        If ``True``, export ``config`` as the chamber specification file.
        Default is ``False`` to export ``config`` as the configuration file.

    Returns
    -------
    None
    """
    if is_chamber:
        filename = path + '/config/chamber.json'
    else:
        filename = path + '/config/config.json'

    with open(filename, 'w') as fp:
        json.dump(config, fp, indent=4)
        fp.write('\n')
