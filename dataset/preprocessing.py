#!/usr/bin/env python3.11
import numpy as np
import pandas as pd
import pickle
import sys

LOG10 = 2.302585092994046

def rebin_pad(dataframe:pd.DataFrame,
              lgTime_min:float=1.0, lgTime_max:float=7.0, lgTime_nbins:int=64,
              padding:float=-3.0,
              full_output:bool=False
             )->tuple:
    """
    Applies padded rebinning to a single Swift-XRT lightcurve.

    The rebinning algorithm is designed so that the average count
    rate in each bin remains unchanged after the rebinning.
    
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
        The value assigned to empty bins. Default value is
        approximately a decimal logarithm of the typical 
        X-Ray background count rate.
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
    
    if full_output:
        lgTime = 0.5 * (bin_edges[1:] + bin_edges[:-1]) 
        num_true_entries = sum(weight.astype(bool))
        return (lgRate, weight, lgTime, num_true_entries)
	else:
        return (lgRate, weight)

with open('SwiftXRT_Dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
print(rebin_pad(Dataset['GRB 221009A']))
