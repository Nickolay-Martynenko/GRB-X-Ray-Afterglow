#!/usr/bin/env python3.11
import os
import pandas as pd
import pickle
import sys
from tqdm import tqdm

LOG10 = 2.302585092994046

def read_SwiftXRT(directory:str,
                  modes:list=['PC_incbad', 'WT_incbad'],
                  only_basic_lightcurve:bool=True,
                  symmetric_errors:bool=True,
                  only_GRB:bool=True
    )->dict:
    """
    Reads Swift-XRT data from the specified directory.
    For each entry, the data points are sorted for the 
    resulting timestamps sequence to be ascending.

    Parameters
    ----------
    directory : str
        Path to the target directory containing
        the raw Swift-XRT dataset.
    modes : list, defatul=['PC_incbad', 'WT_incbad']
        Modes of data collection to be included
        in the output dataset. 
    only_basic_lightcurve : bool, default=True
        If True, only basic lightcurve info will
        be read (i.e. Time, TimePos, TimeNeg,
        Rate, RatePos, RateNeg columns). Otherwise,
        all the columns will be read.
    symmetric_errors : bool, default=True
        Ignored if `only_basic_lightcurve` is False.
        If True, the positive and negative
        errors are symmetrized using their 
        geometric mean. The columns [...]Pos, [...]Neg
        are replaced with a single column [...]Err.
    only_GRB : bool, default=True
        If True, only the events matching
        common-used GRB name pattern (that is,
        confirmed GRB X-Ray afterglows) will
        be processed. Otherwise, the output
        dataset may include X-Ray lightcurves
        not related to any confirmed GRB.

    Returns
    -------
    SwiftXRT : dict
        The resulting dataset.
    """
    if os.path.isdir(directory):
        counter = 0
        events = dict()
        for mode in modes:
            dataset_path = f'{directory}/{mode}'

            if os.path.isdir(dataset_path):
                available_event_names = [event.removesuffix('.json')
                    for event in os.listdir(dataset_path)
                    if (
                        event.endswith('.json') and 
                        (event.startswith('GRB ') or not only_GRB)
                    )
                ]
                print(f'[Processing]: {mode}')
                for event_name in tqdm(available_event_names):
                    dataframe = pd.read_json(
                        f'{dataset_path}/{event_name}.json'
                    )
                    if only_basic_lightcurve:
                        dataframe = dataframe.loc[:,
                            ['Time', 'TimePos', 'TimeNeg',
                             'Rate', 'RatePos', 'RateNeg']
                        ]
                        if symmetric_errors:
                            TimeErr = (
                                -np.prod(
                                    dataframe[
                                        'TimePos', 'TimeNeg'
                                    ].values,
                                    axis=1
                                )
                            )**0.5
                            RateErr = (
                                -np.prod(
                                    dataframe[
                                        'RatePos', 'RateNeg'
                                    ].values,
                                    axis=1
                                )
                            )**0.5
                            dataframe = dataframe.loc[:, 'Time', 'Rate']
                            dataframe.insert(1, 'TimeErr', TimeErr)
                            dataframe.insert(3, 'RateErr', RateErr)
                    counter += 1
                    if event_name not in events.keys():
                        events[event_name] = dataframe
                    else:
                        events[event_name] = pd.concat(
                            [events[event_name], dataframe],
                            axis=0, ignore_index=True
                        ).sort_values(by='Time')
            else:
                print(
                    f'[Warning]: {mode} dataset'+
                    f'not found in {target_directory}'
                )
        print(
            f'Successfully processed {counter} json files.'+
            f'Found {len(events)} unique events' +
            f'(only confirmed GRB).' if only_GRB else f'.'
        )
    else:
        raise FileNotFoundError(
            f'{directory} is not an existing directory!'
        )

def 