import numpy as np
import pandas as pd
from torch import as_tensor, from_numpy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from requests.exceptions import ConnectTimeout
from sklearn.preprocessing import LabelEncoder
from swifttools.ukssdc.data.GRB import GRBNameToTargetID, getLightCurves

LOG10 = 2.302585092994046

def complete_lightcurve(dataframe:pd.DataFrame,
    min_timestamps:int=4, bins:tuple=(1, 7, 64),
    min_bins:int=8)->bool:
    """
    Detects complete/incomplete lightcurves

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame with the raw Swift-XRT lightcurve data.
    min_timestamps : int, default=4
        Minimal required timestamps for 
        the lightcurve to be 'complete'
    bins : tuple, default=(1, 7, 64)
        start, stop, number of bins
        in the decimal log-uniform timescale
    min_bins : int, default=8
        Minimal required non-empty bins for 
        the lightcurve to be 'complete'

    Returns
    -------
    flag : bool
        If True, the lightcurve is complete. 
        Otherwise, the lightcurve is incomplete.
    """

    flag = True
    timeseries = dataframe['Time'].values

    if len(timeseries) < min_timestamps:
        flag = False
        return flag
    else:
        hist, _ = np.histogram(
            timeseries,
            bins=np.logspace(*bins)
        )
        if np.sum( (hist > 0).astype(int) ) < min_bins:
            flag = False
            return flag

    return flag

def rebin(dataframe:pd.DataFrame,
    lgTime_min:float=1.0, lgTime_max:float=7.0,
    lgTime_nbins:int=64, regime:str='padding',
    padding:float=-3.0, subtract_background:bool=True)->dict:
    """
    Applies rebinning to a single Swift-XRT lightcurve.

    If regime is not 'none', the resulting lightcurve will 
    be binned uniformly in the log-Time scale. The rebinning 
    algorithm is designed so that the average count rate in 
    each bin remains unchanged after the rebinning.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame with the raw Swift-XRT lightcurve data.
    lgTime_min : float, default=1.0
        Decimal logarithm of the target time bin edges
        starting point (in seconds). Ignored if `regime`='none'.
    lgTime_max : float, default=7.0
        Decimal logarithm of the target time bin edges
        end point (in seconds). Ignored if `regime`='none'.
    lgTime_nbins : int, default=64
        Number of bins (i.e. the number of bin edges - 1).
        Ignored if `regime`='none'.
    regime : str, default='padding'
        There are three available regimes of rebinning:
        'padding', 'linear_interpolation' and 'none'.
            - 'padding': the rebinned lightcurve is 
            padded with a value controlled 
            by `padding` parameter.
            - 'linear_interpolation': the missing 
            values are interpolated linearly based on 
            the source count rate in the nearest non-empty bins.
            - 'none': only decimal logarithm is applied
            to the original timeseries and timestamps without
            rebinning to a uniform grid.
    padding : float, default=-3.0
        Ignored if `regime` is not 'padding'.
        The value initially assigned to empty bins.
        Default value is approximately a decimal 
        logarithm of the typical X-Ray background 
        count rate. If `subtract_background` is True,
        then the default padded entries would be
        mapped to 0.0 after background subtraction.
    subtract_background : bool, default=True
        Whether to subtract background. If True,
        the preprocessed count rate would be in
        the units of average background count rate,
        that is, 10^{-3} s^{-1}. Otherwise, the 
        original units of s^{-1} are preserved.
        
    Returns
    -------
    rebinned : dict
        rebinned['lgRate'] : np.ndarray
            The (rebinned) source count rate decimal logarithm.
        rebinned['weight'] : np.ndarray
            The estimated inverse squared `lgRate` errors. 
            For the empty bins, a weight of 0.0 is assigned.
        rebinned['lgTime'] : np.ndarray
            The bin centers, in the units of decimal logarithm 
            of time in seconds if `regime` is not 'none'.
            Otherwise, decimal logarithm of the original timestamps.
    """
    assert regime in [
        'padding',
        'linear_interpolation',
        'none'
    ], f"Unknown rebinning regime: '{regime}'"

    lgTimeOrig = dataframe['Time'].apply(np.log10).values

    if regime=='none':
        lgRate = dataframe['Rate'].apply(np.log10).values
        if subtract_background:
            lgRate += 3.0
        lgRateErr = dataframe['RateErr'].values/dataframe['Rate'].values/LOG10
        weight = 1/lgRateErr**2
        # if too many data points, return a sparsed version
        step = len(lgRate) // 1000 + 1
        rebinned = {
            'lgRate': lgRate[::step],
            'weight': weight[::step],
            'lgTime': lgTimeOrig[::step]
        }
        return rebinned

    padding = padding if regime=='padding' else 0.0

    bin_edges = np.linspace(lgTime_min, lgTime_max, lgTime_nbins+1)
    lgTime = 0.5 * (bin_edges[1:] + bin_edges[:-1]) 
    lgRate = np.zeros(lgTime_nbins)
    weight = np.zeros(lgTime_nbins)

    for bin_index in range(lgTime_nbins):
        mask = (
            (lgTimeOrig >= bin_edges[bin_index]) * 
            (lgTimeOrig < bin_edges[bin_index+1])
        )
        dataframe_fragment = dataframe.loc[mask, :].copy()
        
        local_grid = np.exp(
            LOG10 * np.hstack(
                (
                    bin_edges[np.newaxis, bin_index],
                    0.5 * (
                        lgTimeOrig[mask][1:] + lgTimeOrig[mask][:-1]
                    ),
                    bin_edges[np.newaxis, bin_index+1]
                )
            )
        )
        integral = np.sum(
                dataframe_fragment['Rate'].values * np.diff(local_grid)
        ).item()
        flag = integral > 0.0
        lgRate[bin_index] = np.log10(
            integral / np.ptp(local_grid)
        ) if flag else padding
        weight[bin_index] = (LOG10 * integral / 
                             np.sum(
                                dataframe_fragment['RateErr'].values * 
                                np.diff(local_grid)
                             ).item()
        )**2 if flag else 0.0

    if subtract_background:
        lgRate += 3.0
    if regime=='linear_interpolation':
        mask = (weight > 0.0)
        lgRate = np.interp(lgTime, lgTime[mask], lgRate[mask])

    rebinned = {
        'lgRate': lgRate,
        'weight': weight,
        'lgTime': lgTime
    }
    return rebinned

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

    print('[Processing]: Sending request to the Swift-XRT repository...'+
          '\nPlease be patient, this may be time-consuming')

    lcData = dict()

    for event in tqdm(unique_names):
        # Here we are trying to avoid a bug in swifttools:
        # the current verison getLightCurves function cannot 
        # easily skip errors caused by unresolved GRBs

        targetID = GRBNameToTargetID(event, silent=True)
        if targetID is not None:
            try:
                lc = getLightCurves(
                    targetID=targetID,
                    saveData=False, returnData=True, silent=True,
                    incbad="yes", nosys="no")
                lcData[event] = lc
            except ConnectTimeout:
                print(
                    f'[Warning]: Event {event} is resolved as'+
                    f'targetID={targetID}\nbut Swift-XRT repository is not responding'
                )

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

class LightCurveDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame,
                 data_col:str='lgRate',
                 weight_col:str='weight'):
        
        data = np.array(dataframe.loc[:, data_col].tolist(),
                        dtype=np.float32)
        weight = np.array(dataframe.loc[:, weight_col].tolist(),
                          dtype=np.float32)
        
        self.data = from_numpy(
            data).unsqueeze(dim=1)   # value
        self.weight = from_numpy(
            weight).unsqueeze(dim=1) # weight

        # using dataframe index = event names 
        # as labels
        labels = dataframe.index
        self.label_enc = LabelEncoder()
        self.labels = as_tensor(self.label_enc.fit_transform(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.weight[idx]

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
