#!/usr/bin/env python3.11
import numpy as np
import pandas as pd
import pickle
import sys

LOG10 = 2.302585092994046

def extract_raw(dataframe:pd.DataFrame,
                apply_log10_Rate:bool=False,
                full_output:bool=False,
                apply_log10_Time:bool=False)->tuple:
    """
    Extracts relevant lightcurve data from a single Swift-XRT dataframe.
    The positive and negative errors are symmetrized using their geometric mean.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame with the raw Swift-XRT lightcurve data.
    apply_log10_Rate : bool, default=False
        If true, a decimal logarithm is applied to the 
        source rate data points together with their
        uncertainties.
    full_output : bool, default=False
        If True, (Rate, RateErr, Time, TimeErr) is returned.
        Otherwise only (Rate, RateErr) is returned.
    apply_log10_Time : bool, default=False
        Ignored if `full_output`=False. If `full_output`=True
        and `apply_log10_Time`=True, then a decimal logarithm
        is applied to the timestamps together with their
        uncertainties.
    
    Returns
    -------
    Rate : np.ndarray
        1-dimensional array of the source count rate
        in the original or logarithmic units, depending
        on `apply_log10_Rate`.
    RateErr : np.ndarray
        1-dimensional array of the source count rate
        uncertainty in the original or logarithmic units, 
        depending on `apply_log10_Rate`.
    Time : np.ndarray, optional
        Only returned if full_output is True.
        1-dimensional array of the timestamps
        in the original or logarithmic units, 
        depending on `apply_log10_Time`.
    TimeErr: np.ndarray, optional
        Only returned if full_output is True.
        1-dimensional array of the timestamps
        uncertainties in the original or 
        logarithmic units, depending on
        `apply_log10_Time`.
    """
    Rate = dataframe.loc[:, 'Rate'].values
    RateErr = (
        -np.prod(dataframe.loc[:, ['RateNeg', 'RatePos']].values, axis=1)
    )**0.5

    if apply_log10_Rate:
        RateErr = RateErr/Rate/LOG10
        Rate = np.log10(Rate)

    if full_output:
        Time = dataframe.loc[:, 'Time'].values
        TimeErr = (
            -np.prod(dataframe.loc[:, ['TimeNeg', 'TimePos']].values, axis=1)
        )**0.5

        if apply_log10_Time:
            TimeErr = TimeErr/Time/LOG10
            Time = np.log10(Time)

        return (Rate, RateErr, Time, TimeErr)
    else:
        return (Rate, RateErr)
    
def rebin_pad(dataframe:pd.DataFrame,
              lgTime_min:float=1.0, lgTime_max:float=7.0,
              lgTime_nbins:int=64,
              padding:float=-3.0, subtract_background:bool=True,
              full_output:bool=False
             )->tuple:
    """
    Applies padded rebinning to a single Swift-XRT lightcurve.

    The resulting lightcurve will be binned uniformly in the log-Time scale
    The rebinning algorithm is designed so that the average count rate in 
    each bin remains unchanged after the rebinning.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame with the raw Swift-XRT lightcurve data.
    lgTime_min : float, default=1.0
        Decimal logarithm of the target time bin edges
        starting point (in seconds).
    lgTime_max : float, default=7.0
        Decimal logarithm of the target time bin edges
        end point (in seconds).
    lgTime_nbins : int, default=64
        Number of bins (i.e. the number of bin edges - 1).
    padding : float, default=-3.0
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
    full_output : bool, default=False
        If True, return (lgRate, weight, lgTime, num_true_entries),
        where num_true_entries is a number of non-empty bins in the 
        rebinned lightcurve. Otherwise only (lgRate, weight) is returned.
        
    Returns
    -------
    lgRate : np.ndarray
        Array of shape (lgTime_nbins,) of the 
        source count rate rebinned decimal logarithm
    weight : np.ndarray
        Array of shape (lgTime_nbins,) of the
        estimated inverse squared lgRate errors. 
        For the empty bins, a weight of 0.0 is assigned
    lgTime : np.ndarray, optional
        Only returned if full_output is True. Array of
        shape (lgTime_nbins,) of the bin centers, in the
        units of decimal logarithm of time in seconds 
    num_true_entries: int, optional
        Only returned if full_output is True. Number of 
        non-empty bins in the rebinned lightcurve.
        Equivalent to sum(weight.astype(bool)) 
    """
    bin_edges = np.linspace(lgTime_min, lgTime_max, lgTime_nbins+1)
    lgRate = np.zeros(lgTime_nbins)
    weight = np.zeros(lgTime_nbins)
                  
    lgTimeOrig = dataframe['Time'].apply(np.log10).values

    for bin_index in range(lgTime_nbins):
        mask = (lgTimeOrig >= bin_edges[bin_index]) * (lgTimeOrig < bin_edges[bin_index+1])
        dataframe_fragment = dataframe.loc[mask, ['Time', 'Rate', 'RateNeg', 'RatePos']].copy()
        
        local_grid = np.exp(
            LOG10 * np.hstack(
                (
                    bin_edges[np.newaxis, bin_index],
                    0.5 * (lgTimeOrig[mask][1:] + lgTimeOrig[mask][:-1]),
                    bin_edges[np.newaxis, bin_index+1]
                )
            )
        )
        integral = np.sum(
                dataframe_fragment['Rate'].values * np.diff(local_grid)
        ).item()
        flag = integral > 0.0
        lgRate[bin_index] = np.log10(integral / np.ptp(local_grid)) if flag else padding
        weight[bin_index] = (LOG10 * integral / 
                             np.sum(
                                 (-np.prod(
                                     dataframe_fragment.loc[
                                     :, ['RatePos', 'RateNeg']
                                     ].values,
                                     axis=1)
                                 )**0.5 * np.diff(local_grid)
                             ).item()
        )**2 if flag else 0.0

    if subtract_background:
        lgRate += 3.0
    if full_output:
        lgTime = 0.5 * (bin_edges[1:] + bin_edges[:-1]) 
        num_true_entries = sum(weight.astype(bool))
        return (lgRate, weight, lgTime, num_true_entries)
    else:
        return (lgRate, weight)

with open('SwiftXRT_Dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
print(rebin_pad(Dataset['GRB 221009A']))
