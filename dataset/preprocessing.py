#!/usr/bin/env python3.11
import numpy as np
import pandas as pd
import pickle
import sys

LOG10 = 2.302585092994046

def rebin_pad(dataframe:pd.DataFrame, lgTime_min:float=1.0, lgTime_max:float=7.0, lgTime_nbins:int=64, padding:float=-3.0, full_output:bool=False)->tuple:
	"""
	Apply padded rebinning to a single Swift-XRT lightcurve. For each non-empty segment, the total number of counts remains unchanged.
		
		Parameters:
			dataframe: pandas.DataFrame
				DataFrame with the original Swift-XRT lightcurve data
			lgTime_min: float, optional
				Decimal logarithm of the target time bin edges starting 
				point (in seconds). Default is 1.0 (that is, 10 seconds)
			lgTime_max: float, optional
				Decimal logarithm of the target time bin edges end point
				(in seconds). Default is 7.0 (that is, 10^7 seconds ~ 116 days)
			lgTime_nbins: int, optional
				Number of bins (i.e. the number of bin edges - 1). Default is 64
			padding: float, optional
				The value assigned to empty bins. Default is -3.0, which is
				approximately a decimal logarithm of the typical X-Ray background
				count rate
			full_output: bool, optional
				If True, return (lgRate, Weight, num_true_entries), where
				num_true_entries is a number of non-empty (i.e. non-padded)
				bins in the rebinned lightcurve. Otherwise only rebinned_LC
				is returned. Default is False
		
		Returns:
			lgRate: np.ndarray
				(lgTime_nbins,) 1D-array of rebinned decimal logarithm
				of the source count rate
			Weight: np.ndarray
				(lgTime_nbins,) 1D-array of estimated inverse squared
				lgRate errors. For empty bins, a weight of 0.0 is assigned
			num_true_entries: int, optional
				Only returned if full_output is True. Number of non-empty
				(i.e. non-padded) bins in the rebinned lightcurve.
				Equivalent to sum(Weights.astype(bool)) 
	"""
	bin_edges = np.linspace(lgTime_min, lgTime_max, lgTime_nbins+1)
	lgRate = np.zeros(lgTime_nbins)
	Weight = np.zeros(lgTime_nbins)
	
	lgTimeOrig = dataframe['Time'].apply(np.log10).values

	for bin_index in range(lgTime_nbins):
		mask = (lgTimeOrig >= bin_edges[bin_index]) * (lgTimeOrig < bin_edges[bin_index+1])
		dataframe_fragment = dataframe.loc[mask, ['Time', 'Rate', 'RateNeg', 'RatePos']].copy()

		local_grid = np.exp(LOG10 * np.hstack(
				(bin_edges[np.newaxis, bin_index], 0.5 * (lgTimeOrig[mask][1:] + lgTimeOrig[mask][:-1]), bin_edges[np.newaxis, bin_index+1])
				)
		)
		integral = np.sum(
				dataframe_fragment['Rate'].values * np.diff(local_grid)
		).item()
		flag = integral > 0.0
		lgRate[bin_index] = np.log10(integral / np.ptp(local_grid)) if flag else padding
		Weight[bin_index] = (LOG10 * integral / np.sum(
				(-np.prod(dataframe_fragment.loc[:, ['RatePos', 'RateNeg']].values, axis=1))**0.5 * np.diff(local_grid)
			).item()
		)**2 if flag else 0.0
	
	if full_output:
		num_true_entries = sum(Weights.astype(bool))
		return (lgRate, Weight, num_true_entries)
	else:
		return (lgRate, Weight)

with open('SwiftXRT_Dataset.pickle', 'rb') as f:
	Dataset = pickle.load(f)
print(rebin_pad(Dataset['GRB 221009A']))
