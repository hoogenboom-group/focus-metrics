# -*- coding: utf-8 -*-
"""
@Author: Ryan Lane
@Date:   21-01-2022
"""

import re
import pandas as pd


def natural_sort(l):
    """A more natural sorting algorithm

    Parameters
    ----------
    l : list
        List of strings in need of sorting

    Returns
    -------
    out : list
        More naturally sorted list

    Examples
    --------
    >>> l = ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']
    >>> sorted(l)
    ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']
    >>> natural_sort(l)
    ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']

    References
    ----------
    [1] https://stackoverflow.com/a/4836734/5285918
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def parse_logs(filepath):
    """Parse autofocus log files for position and focus measurement data

    Parameters
    ----------
    filepath : str
        Filepath to log file

    Returns
    -------
    df : `pd.DataFrame`
        Relevant data from log file as a DataFrame
    """
    # Store relevant data (stage height and focus measurement)
    z_positions = []
    focus_measurements = []
    # Parse file for numerical data
    with open(filepath) as log_data:
        for line in log_data:
            if 'Focus level' in line:
                z, fm = re.findall(r"[-+]?\d*\.\d+|\d+", line)[-2:]
                z_positions.append(float(z))
                focus_measurements.append(float(fm))
    # Create DataFrame
    data = {'Z': z_positions,
            'FM': focus_measurements}
    df = pd.DataFrame(data, columns=['Z', 'FM'])
    return df
