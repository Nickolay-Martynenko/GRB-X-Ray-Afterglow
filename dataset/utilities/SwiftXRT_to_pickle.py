#!/usr/bin/env python3.11
import os
import pandas as pd
import pickle
import sys
from tqdm import tqdm

args = sys.argv
assert len(args) == 3, f'Invalid number of arguments.\nUsage: python3 {args[0]} target_directory output_file'
target_directory, output_file = sys.argv[1:]

if os.path.isdir(target_directory):
    counter = 0
    events = dict()

    modes = ['PC_incbad', 'WT_incbad']
    for mode in modes:
        dataset_path = f'{target_directory}/{mode}'

        if os.path.isdir(dataset_path):
            available_event_names = [event.removesuffix('.json') for event in os.listdir(dataset_path) if event.endswith('.json')]
            print(f'[Processing]: {mode}')
            for event_name in tqdm(available_event_names):
                dataframe = pd.read_json(f'{dataset_path}/{event_name}.json')
                counter += 1
                if event_name not in events.keys():
                    events[event_name] = dataframe
                else:
                    events[event_name] = pd.concat([events[event_name], dataframe], axis=0, ignore_index=True).sort_values(by='Time')
        else:
            print(f'[Warning]: {mode} dataset not found in {target_directory}')
    print(f'Successfully processed {counter} json files. Found {len(events)} unique events')
    
    with open(output_file, 'wb') as f:
        pickle.dump(events, f)
    print(f"The result is stored at '{output_file}'")
else:
    raise FileNotFoundError(f'{target_directory} is not an existing directory!')
