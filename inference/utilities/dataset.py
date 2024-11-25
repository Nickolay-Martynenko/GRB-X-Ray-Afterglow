import os
import sys

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from swifttools.ukssdc.data.GRB import GRBNameToTargetID, getLightCurves

module_path = os.path.abspath(os.path.join('../dataset'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utilities.SwiftXRTpreprocessing import complete_lightcurve
from utilities.SwiftXRTpreprocessing import rebin

class LightCurveDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame,
                 data_col:str='lgRate',
                 weight_col:str='weight'):
        
        data = np.array(dataframe.loc[:, data_col].tolist(),
                        dtype=np.float32)
        weight = np.array(dataframe.loc[:, weight_col].tolist(),
                          dtype=np.float32)
        
        self.data = torch.from_numpy(
            data).unsqueeze(dim=1)   # value
        self.weight = torch.from_numpy(
            weight).unsqueeze(dim=1) # weight

        # using dataframe index = event names 
        # as labels
        labels = dataframe.index
        self.label_enc = LabelEncoder()
        self.labels = torch.as_tensor(self.label_enc.fit_transform(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.weight[idx]

def get_SwiftXRTLightCurves(event_names_list:list)->tuple:
    """
    Downloads & preprocesses lightcurves for the
    events listed in event_names_list.

    Parameters
    ----------
    events_names_list : list
        List of GRB ids to be processed.

    Returns
    -------
    lightcurves : pd.DataFrame
        The resulting dataset.

    info : pd.DataFrame
        Dataframe indicating whether the data collecting
        procedure was successfull/failed for each event.

        Description:
        - 'complete_lightcurve' : data collected successfully
        - 'incomplete_lightcurve' : the lightcurve is too short to analyse it
        - 'missing_data' : neither PC_incbad nor WT_incbad dataset is available 
        for this event
        - 'not_found' : the passed event name not found in Swift-XRT repository
    """

    unique_names, counts = np.unique(event_names_list, return_counts=True)
    if (counts > 1).any():
        print(
            '[Warning]: Names passed multiple times found, '+
            f'only {len(unique_names)} unique entries will be processed'
            )

    print('[Processing]: Sending request to the Swift-XRT repository...')
    print('[Processing]: Please be patient, this may be time-consuming')
    lcData = dict()
    for event in unique_names:
        # Here we are trying to avoid a bug in swifttools:
        # the current verison getLightCurves function cannot 
        # easily skip errors caused by unresolved GRBs

        targetID = GRBNameToTargetID(event, silent=True)
        if targetID is not None:
            lc = getLightCurves(
                targetID=targetID,
                saveData=False, returnData=True, silent=True,
                incbad="yes")
            lcData[event] = lc

    lightcurves_info = dict()
    for event in unique_names:
        if event not in lcData.keys():
            lightcurves_info[event] = 'not_found'

    print(f'[Processing]: Found {len(lcData)} out of {len(unique_names)} requested events')
    print(f'[Processing]: Rebinning in progress...')

    lightcurves = dict()
    for event, data in lcData.items():
        modes = [mode for mode in ['PC_incbad', 'WT_incbad']
            if mode in data['Datasets']]
        df = None
        for mode in modes:
            if df is None:
                df = data[mode].loc[:,
                    [
                    'Time', 'TimePos', 'TimeNeg',
                    'Rate', 'RatePos', 'RateNeg'
                    ]
                ]
            else:
                df = pd.concat(
                    (df, data[mode].loc[:,
                            [
                            'Time', 'TimePos', 'TimeNeg',
                            'Rate', 'RatePos', 'RateNeg'
                            ]
                        ]
                    ),
                    axis=0, ignore_index=True
                )
        if df is not None:
            df = df.sort_values(by='Time')
            TimeErr = (
                -np.prod(
                    df[
                        ['TimePos', 'TimeNeg']
                    ].values, axis=1
                )
            )**0.5
            RateErr = (
                -np.prod(
                    df[
                        ['RatePos', 'RateNeg']
                    ].values, axis=1
                )
            )**0.5
            df = df.loc[:, ['Time', 'Rate']]
            df.insert(1, 'TimeErr', TimeErr)
            df.insert(3, 'RateErr', RateErr)

            if complete_lightcurve(df):
                lightcurves_info[event] = 'complete_lightcurve'
                lightcurves[event] = rebin(df, regime='linear_interpolation')
            else:
                lightcurves_info[event] = 'incomplete_lightcurve'
        else:
            lightcurves_info[event] = 'missing_dataset'

    print(f'[Processing]: Rebinning finished')
    info = pd.DataFrame.from_dict(lightcurves_info, orient='index',
        columns=['info']).sort_index(axis=0)
    print(f'[Processing]: Job complete!')
    for key, val in info['info'].value_counts().to_dict().items():
        key = key.ljust(22)
        print(f'    {key} : {val} entries')
    lightcurves = pd.DataFrame.from_dict(lightcurves, orient='index').sort_index(axis=0)

    return (lightcurves, info)

def create_Dataloader(lightcurves:pd.DataFrame)->DataLoader:
    """
    Creates torch Dataloader for the passed lightcurves

    Parameters
    ----------
    lightcurves : pd.DataFrame
       GRB data to be processed (the format is analogous to 
       that of get_SwiftXRTLightCurves first output)

    Returns
    -------
    dataloader : Dataloader
        The resulting Dataloader.
    """

    dataset = LightCurveDataset(lightcurves)
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )
    return dataloader
