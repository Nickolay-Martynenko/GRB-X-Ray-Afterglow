#!/usr/bin/env python3.11
import os
import pandas as pd
import pickle
import sys
from statistics import NormalDist
from sklearn.linear_model import HuberRegressor
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
        return events
    else:
        raise FileNotFoundError(
            f'{directory} is not an existing directory!'
        )

def rebin(dataframe:pd.DataFrame,
          lgTime_min:float=1.0, lgTime_max:float=7.0,
          lgTime_nbins:int=64,
          regime:str='padding', padding:float=-3.0,
          subtract_background:bool=True,
          full_output:bool=False
    )->tuple:
    """
    Applies rebinning to a single Swift-XRT lightcurve.

    The resulting lightcurve will be binned uniformly in 
    the log-Time scale. The rebinning algorithm is designed 
    so that the average count rate in each bin remains unchanged 
    after the rebinning.
    
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
    regime : str, default='padding'
        There are two available regimes of rebinning:
        'padding' and 'linear_interpolation'. In the 'padding'
        regime, the rebinned lightcurve is padded with a value
        controlled by `padding` parameter. In the 
        'linear_interpolation' regime, the missing 
        values are interpolated linearly based on 
        the source count rate in the nearest non-empty bins.
    padding : float, default=-3.0
        Ignored if `regime` is 'linear_interpolation'.
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
    lgRate : np.ndarray
        Array of shape (lgTime_nbins,) of the 
        source count rate rebinned decimal logarithm
    weight : np.ndarray
        Array of shape (lgTime_nbins,) of the
        estimated inverse squared `lgRate` errors. 
        For the empty bins, a weight of 0.0 is assigned
    lgTime : np.ndarray
        Array of shape (lgTime_nbins,) of the bin centers, 
        in the units of decimal logarithm of time in seconds 
    """
    assert regime in ['padding',
        'linear_interpolation'], f"Unknown rebinning regime: '{regime}'"
    padding = padding if regime=='padding' else 0.0

    bin_edges = np.linspace(lgTime_min, lgTime_max, lgTime_nbins+1)
    lgTime = 0.5 * (bin_edges[1:] + bin_edges[:-1]) 
    lgRate = np.zeros(lgTime_nbins)
    weight = np.zeros(lgTime_nbins)
                  
    lgTimeOrig = dataframe['Time'].apply(np.log10).values

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
        mask = weight.astype(bool)
        lgRate = np.interp(lgTime, lgTime[mask], lgRate[mask])
    return (lgRate, weight, lgTime)

class FeatureExtractor:
    """
    Extracts common-used astrophysical lightcurves features.

    Based on light_curve_feature module:
    https://docs.rs/light-curve-feature/latest/light_curve_feature/features/index.html

    See also: 
    - DOI:10.1051/0004-6361/201323252
    - DOI:10.1093/mnras/stw157
    - DOI:10.3847/1538-4357/aa9188
    and references therein.

    Attributes
    ----------
    magnitude : np.ndarray
        Decimal logarithm of the count rate
        Note that it does not match with the
        convention usually adopted in astronomy
    magnitudeErr : np.ndarray
        Standard uncertainty of the magnitude
    timestamps : np.ndarray
        Timestamps at which the magnitude data 
        points were measured.

    Methods
    -------
    """
    def __init__(self, dataframe:pd.DataFrame):
        """
        Constructs necessary attributes for the futrher work.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame with the raw Swift-XRT lightcurve data.
            The timeseries is assumed to be sorted so that 
            time is strictly ascending.

        Returns
        -------
        None
        """
        self.magnitude = dataframe['Rate'].apply(np.log10).values
        self.magnitudeErr = dataframe['RateErr'].values/dataframe['Rate'].values/LOG10
        self.timestamps = dataframe['Time'].values
        assert (np.diff(self.timestamps) > 0).all(), f'Non-ascending timestamps sequence!'

    def Amplitude(self)->float:
        """
        Returns half-amplitude of magnitude.

        Parameters
        ----------
        None

        Returns
        -------
        amplitude : float
            Half-amplitude of magnitude.
        """
        amplitude = np.ptp(self.magnitude).item() / 2.0
        return amplitude

    def AndersonDarlingNormal(self)->float:
        """
        Returns unbiased Anderson–Darling normality test statistic.

        Parameters
        ----------
        None

        Returns
        -------
        statistic : float
            A.-D. normality test statistic.
            [https://en.wikipedia.org/wiki/Anderson–Darling_test]
        """
        N = len(self.magnitude)
        assert N >= 4, 'Not enough data to use Anderson-Darling normality test'
        coef = 1 + 4/N - (5/N)**2

        mu = np.mean(self.magnitude)
        sigma = np.std(self.magnitude, ddof=1)
        distribution = (self.magnitude - mu)/sigma

        cdf = np.vectorize(lambda x: NormalDist().cdf(x))
        Phi = cdf(distribution)
        assert (
            (Phi > 0.0) * (Phi < 1.0)
        ).all(), 'Invalid CDF values found. Anderson-Darling normality test failed'

        statistic = -coef * (N + np.mean(
                np.arange(1, 2*N, 2) * np.log(Phi) + 
                np.arange(2*N-1, 0, -2) * np.log(1.0-Phi)
            )
        ).item()

        return statistic

    def BeyondNStd(self, N:float=1.0)->float:
        """
        Returns the fraction of data points 
        with the values beyond +-n standard
        deviations from the mean magnitude.

        Parameters
        ----------
        N : float, default=1.0
            Number of standard deviations.

        Returns
        -------
        fraction : float
            The fraction of data points 
            with the values beyond +-n standard
            deviations from the mean magnitude.
        """

        mu = np.mean(self.magnitude)
        sigma = np.std(self.magnitude, ddof=1)
        fraction = np.mean(
            (
                np.abs(
                    (self.magnitude - mu)/sigma
                ) >= N
            ).astype(float)
        ).item()

        return fraction

    def Cusum(self)->float:
        """
        Returns a range of cumulative sums.

        Parameters
        ----------
        None

        Returns
        -------
        cusum_range : float
            max(cumulative sum) - min(cumulative sum)
            divided by (magnitude stand. dev. * number of timestamps)
        """
        sigma = np.std(self.magnitude, ddof=1)
        N = len(self.magnitude)
        cusum_range = (
            np.ptp(np.cumsum(self.magnitude)) / sigma / N
        ).item()

        return cusum_range

    def Duration(self)->float:
        """
        Returns the timeseries duration.

        Parameters
        ----------
        None

        Returns
        -------
        duration : float
            last timestamp - first timestamp
        """
        duration = self.timestamps[-1] - self.timestamps[0]

        return duration

    def EtaE(self)->float:
        """
        Returns von Neumann Eta tuned for
        non-uniform timeseries. Probably still
        unreliable for highly non-uniform timeseries.

        Parameters
        ----------
        None

        Returns
        -------
        etaE : float
            von Neumann Eta parameter
        """

        sigma = np.std(self.magnitude, ddof=1)
        N = len(self.magnitude)
        etaE = (
            self.Duration()**2 * 
            np.mean(
                (
                    np.diff(self.magnitude)/np.diff(self.timestamps)
                )**2
            ) / sigma**2 / (N-1)**2
        ).item()

        return etaE

    def ExcessVariance(self)->float:
        """
        Returns the measure of
        the magnitude variability.

        Parameters
        ----------
        None

        Returns
        -------
        excess : float
            Excess variance parameter.
        """
        mu = np.mean(self.magnitude)
        sigma = np.std(self.magnitude, ddof=1)
        mse = np.mean(self.magnitudeErr**2)

        excess = (sigma**2 - mse)/mu**2

        return excess

    def InterQuantileRange(self, q:float=0.25)->float:
        """
        Returns the range between the quantiles q
        and 1-q of the magnitude distribution.
        Quantiles are approximated using the
        closest observation.

        Parameters
        ----------
        q: float, default=0.25
            The parameter determining the 
            desired range. Default is 
            interquartile interval

        Returns
        -------
        iqr : float
            The interquantile range.
        """

        iqr = (
            np.quantile(self.magnitude, q,
                method='closest_observation')-
            np.quantile(self.magnitude, 1-q,
                method='closest_observation')
        ).item()

        return iqr

    def Kurtosis(self)->float:
        """
        Returns the kurtosis of the
        magnitude distribution.

        Parameters
        ----------
        None

        Returns
        -------
        G2 : float
            The unbiased kurtosis statistic 
            of the magnitude distribution.
        """

        N = len(self.magnitude)
        assert N >= 4, 'Not enough data to calculate kurtosis'

        mu = np.mean(self.magnitude)
        sigma = np.std(self.magnitude, ddof=1)

        G2 = (
            N * (N+1) / (N-1) / (N-2) / (N-3) * 
            np.sum(
                (self.magnitude-mu)**4
            ) / sigma**4
            - 3 * (N-1)**2 / (N-2) / (N-3)
        ).item()

        return G2

    def HuberLinearFit(self)->tuple:
        """
        Returns the intercept and slope
        of the magnitude - time decimal logarithm
        trend. Uses a default configuration
        of the sklearn HuberRegressor model 
        with sample_weight inverse proportional
        to the squared magnitude errors to 
        estimate the trend.

        Parameters
        ----------
        None

        Returns
        -------
        intercept : float
            The estimated intercept of the trend.
        slope : float
            The estimated slope of the trend.
        """
        X = np.log10(self.timestamps).reshape(-1, 1)
        y = self.magnitude
        sample_weight = self.magnitudeErr**(-2)
        fitted = HuberRegressor().fit(X, y, sample_weight=sample_weight)
        intercept, slope = fitted.intercept_, fitted.coef_.item()
        return (intercept, slope)

    def Mean(self)->float:
        """
        Returns the mean magnitude.

        Parameters
        ----------
        None

        Returns
        -------
        mu : float
            The mean magnitude.
        """
        mu = np.mean(self.magnitude)
        return mu

    def MeanVariance(self)->float:
        """
        Returns the standard deviation-to-mean 
        ratio for the magnitude.

        Parameters
        ----------
        None

        Returns
        -------
        ratio : float
            Standard deviation-to-mean ratio 
            for the magnitude.
        """
        mu = np.mean(self.magnitude)
        sigma = np.std(self.magnitude, ddof=1)
        return sigma/mu

    




















