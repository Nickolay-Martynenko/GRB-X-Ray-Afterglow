import os
import re
import numpy as np
import pandas as pd
from statistics import NormalDist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor
from tqdm import tqdm

LOG10 = 2.302585092994046
PATTERN = r'GRB [0-9]{2}[0-1]{1}[0-9]{1}[0-3]{1}[0-9]{1}[A-Z]?'

def read_SwiftXRT(directory:str,
                  modes:list=['PC_incbad', 'WT_incbad'], metadata_file:str=None,
                  only_basic_lightcurve:bool=True,
                  symmetric_errors:bool=True,
                  only_GRB:bool=True)->dict:
    """
    Reads Swift-XRT data from the specified directory.
    For each entry, the data points are sorted for the 
    resulting timestamps sequence to be ascending.

    Parameters
    ----------
    directory : str
        Path to the target directory containing
        the raw Swift-XRT dataset.
    modes : list, default=['PC_incbad', 'WT_incbad']
        Modes of data collection to be included
        in the output dataset. 
    metadata_file : str, default=None
        File to read metadata from (only csv format is supported).
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

    Raises
    ------
    FileNotFoundError
        If `directory` is not an existing directory.
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
                        (
                            bool(
                                re.fullmatch(PATTERN, event.removesuffix('.json'))
                                ) or not only_GRB
                        )
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
                                        ['TimePos', 'TimeNeg']
                                    ].values,
                                    axis=1
                                )
                            )**0.5
                            RateErr = (
                                -np.prod(
                                    dataframe[
                                        ['RatePos', 'RateNeg']
                                    ].values,
                                    axis=1
                                )
                            )**0.5
                            dataframe = dataframe.loc[:, ['Time', 'Rate']]
                            dataframe.insert(1, 'TimeErr', TimeErr)
                            dataframe.insert(3, 'RateErr', RateErr)
                    counter += 1
                    if event_name not in events.keys():
                        events[event_name] = {'data': dataframe}
                    else:
                        events[event_name]['data'] = pd.concat(
                            [events[event_name]['data'], dataframe],
                            axis=0, ignore_index=True
                        ).sort_values(by='Time')
            else:
                print(
                    f'[Warning]: {mode} dataset'+
                    f'not found in {target_directory}'
                )
        print(
            f'Successfully processed {counter} json files. '+
            f'Found {len(events)} unique events' +
            f' (only confirmed GRB).' if only_GRB else f'.'
        )
        if os.path.isfile(f'{directory}/{metadata_file}'):
            metadata = pd.read_csv(
                f'{directory}/{metadata_file}',
                index_col=0
            )
            for event in events.keys():
                if event in metadata.index:
                    events[event].update(
                        metadata.loc[event, :].to_dict()
                    )
        return events
    else:
        raise FileNotFoundError(
            f'{directory} is not an existing directory!'
        )

def complete_lightcurve(dataframe:pd.DataFrame,
                        min_timestamps:int=4,
                        bins:tuple=(1, 7, 64), min_bins:int=8)->bool:
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


def get_year(event_name:str)->int:
    """
    Reads a year from the common-used GRB name pattern

    Parameters
    ----------
    event_name : str
        Confirmed Gamma-Ray Burst event name,
        e.g. 'GRB 221009A'

    Returns
    -------
    year : int
        The year of detection.

    Raises
    ------
    ValueError
        If the string `event_name` does not
        match the expected pattern.

    Examples
    --------
    >>> get_year('GRB 221009A')
    2022
    """

    match = re.fullmatch(PATTERN, event_name)
    if match:
        year = 2000 + int(event_name.removeprefix('GRB ')[:2])
        return year
    else:
        raise ValueError(f'{event_name} does not match the expected pattern')

def rebin(dataframe:pd.DataFrame,
          lgTime_min:float=1.0, lgTime_max:float=7.0,
          lgTime_nbins:int=64,
          regime:str='padding', padding:float=-3.0,
          subtract_background:bool=True,
          masked_flares:bool=False, flares_list:list=[],
          **kwargs
    )->dict:
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
    masked_flares : bool, deafult=False
        Indicates whether to mask X-Ray flares.
    flares_list : list, default=[]
        List of tuples indicating start and end timestamps
        of X-ray flares. Ignored if masked_flares is False.
        
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
        rebinned['flares'] : np.ndarray
            Flags indicating flares in the X-ray lightcurve timeseries.
            Only returned if masked_flares is True.
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

        if masked_flares:
            flares = np.full_like(lgTimeOrig, fill_value=False)
            for (start, stop) in flares_list:
                flares += (
                    (start <= 10**lgTimeOrig) * (10**lgTimeOrig <= stop)
                ).astype(bool)

        # if too many data points, return a sparsed version
        step = len(lgRate) // 1000 + 1
        rebinned = {
            'lgRate': lgRate[::step],
            'weight': weight[::step],
            'lgTime': lgTimeOrig[::step]
        }

        if masked_flares:
            rebinned.update({'flares': flares[::step]})

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

    if masked_flares:
            flares = np.full_like(lgTime, fill_value=False)
            for (start, stop) in flares_list:
                flares += (
                    (start <= 10**lgTime) * (10**lgTime <= stop)
                ).astype(bool)
            rebinned.update({'flares': flares})

    return rebinned

class FeatureExtractor:
    """
    Extracts common-used astrophysical lightcurves features.

    Based on light_curve_feature module:
    https://docs.rs/light-curve-feature/latest/light_curve_feature/features/index.html

    See also: 
    - DOI:10.1051/0004-6361/201323252
    - DOI:10.1086/133808
    - DOI:10.1093/mnras/stw157
    - DOI:10.1109/tsmc.1979.4310076
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
    Amplitude()->float:
        Half-amplitude of magnitude.
    AndersonDarlingNormal()->float:
        Unbiased Anderson???Darling normality test statistic.
    BeyondNStd(N:float=1.0)->float:
        The fraction of data points with the 
        values beyond +-n standard deviations
        from the mean magnitude.
    Cusum()->float:
        Range of cumulative sums.
    Duration()->float:
        Timeseries duration.
    EtaE()->float:
        Von Neumann Eta tuned for non-uniform timeseries. 
    ExcessVariance()->float:
        The measure of magnitude variability.
    HuberLinearFit()->tuple:
        Huber Linear Regression in the log-time scale.
    InterQuantileRange(q:float=0.25)->float:
        The range between the quantiles q
        and 1-q from the magnitude distribution.
    Kurtosis()->float:
        The kurtosis of the magnitude distribution.
    Mean()->float:
        The mean magnitude.
    MeanVariance()->float:
        The standard deviation-to-mean 
        ratio for the magnitude distribution.
    Median()->float:
        The median magnitude.
    MedianAbsoluteDeviation()->float:
        Median of the absolute value of the difference 
        between magnitude and the median magnitude.
    MedianBufferRangePercentage(q:float=0.5)->float:
        The fraction of data points with the values inside a
        symmetric interval of +-q/2 * (max(magnitude)-min(magnitude))
        around the median magnitude.
    OtsuSplit()->tuple:
        Otsu threshholding algorithm.
    PercentAmplitude()->float:
        Maximum absolute deviation of the magnitude 
        from its median.
    PercentDifferenceMagnitudeQuantile(q:float=0.25)->float:
        The interquantile range-to-median ratio
        of the magnitude distribution.
    ReducedChi2()->float:
        Reduced Chi-Squared test statistic
        for the magnitude measurements.
    RobustMedianStatistic()->float:
        Robust median statistic of the
        magnitude distribution.
    Skew()->float:
        Skewness of magnitude.
    StandardDeviation()->float:
        Standard deviation of magnitude.
    StetsonK()->float:
        Stetson K coefficient describing lightcurve shape.
    WeightedMean()->float:
        Weighted mean magnitude.
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
        Half-amplitude of magnitude.

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
        Unbiased Anderson???Darling normality test statistic.

        Parameters
        ----------
        None

        Returns
        -------
        statistic : float
            A.-D. normality test statistic
            [https://en.wikipedia.org/wiki/Anderson???Darling_test]
            If A.-D. test statistic for some reason can not be
            calculated, np.NaN is returned instead.
        """

        N = len(self.magnitude)
        assert N >= 4, 'Not enough data to use Anderson-Darling normality test'
        coef = 1 + 4/N - (5/N)**2

        mu = self.Mean()
        sigma = self.StandardDeviation()
        distribution = (self.magnitude - mu)/sigma

        cdf = np.vectorize(lambda x: NormalDist().cdf(x))
        Phi = cdf(distribution)

        flag = ( (Phi > 0.0) * (Phi < 1.0) ).all()
        if not flag:
            # Anderson-Darling normality test failed because of invalid CDF values
            return np.NaN

        statistic = -coef * (N + np.mean(
                np.arange(1, 2*N, 2) * np.log(Phi) + 
                np.arange(2*N-1, 0, -2) * np.log(1.0-Phi)
            )
        ).item()

        return statistic

    def BeyondNStd(self, N:float=1.0)->float:
        """
        The fraction of data points with the 
        values beyond +-n standard deviations
        from the mean magnitude.

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

        mu = self.Mean()
        sigma = self.StandardDeviation()
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
        Range of cumulative sums.

        Parameters
        ----------
        None

        Returns
        -------
        cusum_range : float
            max(cumulative sum) - min(cumulative sum)
            divided by (magnitude stand. dev. * number of timestamps)
        """

        sigma = self.StandardDeviation()
        N = len(self.magnitude)
        cusum_range = (
            np.ptp(np.cumsum(self.magnitude)) / sigma / N
        ).item()

        return cusum_range

    def Duration(self)->float:
        """
        Timeseries duration.

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
        Von Neumann Eta tuned for non-uniform timeseries. 
        Probably still unreliable for highly non-uniform
        timeseries.

        Parameters
        ----------
        None

        Returns
        -------
        etaE : float
            von Neumann Eta parameter
        """

        sigma = self.StandardDeviation()
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
        The measure of magnitude variability.

        Parameters
        ----------
        None

        Returns
        -------
        excess : float
            Excess variance parameter.
        """

        mu = self.Mean()
        sigma = self.StandardDeviation()
        mse = np.mean(self.magnitudeErr**2)

        excess = (sigma**2 - mse)/mu**2

        return excess

    def HuberLinearFit(self)->tuple:
        """
        Huber Linear Regression in the log-time scale.

        Uses a default configuration
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
        estimator = HuberRegressor().fit(X, y, sample_weight=sample_weight)
        intercept, slope = estimator.intercept_, estimator.coef_.item()

        return (intercept, slope)
    HuberLinearFit.suffix = ['intercept', 'slope']

    def InterQuantileRange(self, q:float=0.25)->float:
        """
        The range between the quantiles q
        and 1-q from the magnitude distribution.
        Quantiles are approximated using the
        closest observation.

        Parameters
        ----------
        q : float, default=0.25
            The parameter determining the 
            desired range. Default is 
            interquartile interval

        Returns
        -------
        iqr : float
            The interquantile range.
        """

        iqr = (
            np.quantile(self.magnitude, 1-q,
                method='closest_observation')-
            np.quantile(self.magnitude, q,
                method='closest_observation')
        ).item()

        return iqr

    def Kurtosis(self)->float:
        """
        The kurtosis of the magnitude distribution.

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

        mu = self.Mean()
        sigma = self.StandardDeviation()

        G2 = (
            N * (N+1) / (N-1) / (N-2) / (N-3) * 
            np.sum(
                (self.magnitude-mu)**4
            ) / sigma**4
            - 3 * (N-1)**2 / (N-2) / (N-3)
        ).item()

        return G2

    def Mean(self)->float:
        """
        The mean magnitude.

        Parameters
        ----------
        None

        Returns
        -------
        mu : float
            The mean magnitude.
        """

        mu = np.mean(self.magnitude).item()

        return mu

    def MeanVariance(self)->float:
        """
        The standard deviation-to-mean 
        ratio for the magnitude distribution.

        Parameters
        ----------
        None

        Returns
        -------
        ratio : float
            Standard deviation-to-mean ratio 
            for the magnitude.
        """

        mu = self.Mean()
        sigma = self.StandardDeviation()

        return sigma/mu

    def Median(self)->float:
        """
        The median magnitude.

        Parameters
        ----------
        None

        Returns
        -------
        med : float
            The median magnitude.
        """
        med = np.median(self.magnitude).item()

        return med

    def MedianAbsoluteDeviation(self)->float:
        """
        Median of the absolute value of the difference between
        magnitude and the median magnitude.

        Parameters
        ----------
        None

        Returns
        -------
        med_abs_dev : float
            Median of the absolute value of 
            the difference between magnitude 
            and its median.
        """

        med = self.Median()
        med_abs_dev = np.median(
            np.abs(self.magnitude - med)
        ).item()

        return med_abs_dev

    def MedianBufferRangePercentage(self, q:float=0.5)->float:
        """
        The fraction of data points with the values inside a
        symmetric interval of +-q/2 * (max(magnitude)-min(magnitude))
        around the median magnitude.

        Parameters
        ----------
        q : float, default=0.5
            The width of the interquantile interval.

        Returns
        -------
        fraction : float
            The fraction of data points with
            the values inside the symmetric interval 
            of +-q/2 * (max(magnitude)-min(magnitude))
            around the median magnitude.
        """

        med = self.Median()
        width = q * np.ptp(self.magnitude) / 2

        fraction = np.mean(
            (
                np.abs(self.magnitude-med) < width
            ).astype(float)
        ).item()

        return fraction

    def OtsuSplit(self)->tuple:
        """
        Otsu threshholding algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        opt_threshold : float
            The optimal Otsu threshold 
            maximizing w0 * w1 * (mu1 - mu0)**2,
            where w0 and w1 are the fractions
            of observations belonging to 
            subsets 0 and 1, respectively,
            mu0 and mu1 are the subset means.

        mu_dif : float
            Difference of subset means.

        std0 : float
            Standard deviation of the lower subset.

        std1 : float
            Standard deviation of the upper subset.

        w0 : float
            Lower-to-all observation count ratio.
        """

        N = len(self.magnitude)
        assert N >= 2, 'Not enough data for Otsu Split'

        frac = lambda threshold: (
            np.mean(
                self.magnitude <= threshold
            ).astype(float)
        ).item()

        dif = lambda threshold: (
            np.mean(self.magnitude[self.magnitude > threshold]) - 
            np.mean(self.magnitude[self.magnitude <= threshold])
        ).item()

        goodness = lambda threshold: (
            frac(threshold) * (1-frac(threshold)) * dif(threshold)**2
        )

        thresholds = np.sort(self.magnitude)[1:-2]
        opt_threshold = thresholds[
            np.argmax(
                [goodness(threshold) for threshold in thresholds]
            )
        ]

        mu_dif = dif(opt_threshold)
        std0 = np.std(self.magnitude[self.magnitude <= opt_threshold])
        std1 = np.std(self.magnitude[self.magnitude > opt_threshold])
        w0 = frac(opt_threshold)

        return (opt_threshold, mu_dif, std0, std1, w0)
    OtsuSplit.suffix = ['opt_threshold', 'mu_dif', 'std0', 'std1', 'w0']

    def PercentAmplitude(self)->float:
        """
        Maximum absolute deviation 
        of the magnitude from its median.

        Parameters
        ----------
        None

        Returns
        -------
        percent_ampli : float
            Maximum absolute deviation 
            of the magnitude from its median.
        """

        percent_ampli = np.max(
            np.abs(self.magnitude-self.Median())
        )

        return percent_ampli

    def PercentDifferenceMagnitudeQuantile(self,q:float=0.25)->float:
        """
        The interquantile range-to-median ratio
        of the magnitude distribution.

        Parameters
        ----------
        q : float, default=0.25
            The parameter determining the 
            desired range. Default is 
            interquartile interval

        Returns
        -------
        ratio : float
            The interquantile range-to-median ratio.
        """

        iqr = self.InterQuantileRange(q)
        med = self.Median()
        ratio = iqr/med

        return ratio

    def ReducedChi2(self)->float:
        """
        Reduced Chi-Squared test statistic
        for the magnitude measurements.

        Parameters
        ----------
        None

        Returns
        -------
        red_chi2 : float
            The reduced Chi-Squared statistic.
        """

        N = len(self.magnitude)
        average = self.WeightedMean()
        red_chi2 = np.sum(
            (self.magnitude-average)**2 * self.magnitudeErr**(-2)
        ).item() / (N-1)

        return red_chi2

    def RobustMedianStatistic(self)->float:
        """
        Robust median statistic of the
        magnitude distribution.

        Parameters
        ----------
        None

        Returns
        -------
        roms : float
            The robust median statistic.
        """

        N = len(self.magnitude)
        roms = np.sum(
            np.abs(self.magnitude - self.Median()) / self.magnitudeErr
        ).item() / (N-1)

        return roms

    def Skew(self)->float:
        """
        Skewness of magnitude.

        Parameters
        ----------
        None

        Returns
        -------
        G1 : float
            The sample skewness statistic.
        """

        N = len(self.magnitude)
        mu = self.Mean()
        sigma = self.StandardDeviation()

        G1 = N / (N-1) / (N-2) * np.sum(
            (self.magnitude-mu)**3 / sigma**3
        ).item()

        return G1

    def StandardDeviation(self)->float:
        """
        Standard deviation of magnitude.

        Parameters
        ----------
        None

        Returns
        -------
        sigma : float
            The sample (unbiased) standard deviation.
        """

        sigma = np.std(self.magnitude, ddof=1)

        return sigma

    def StetsonK(self)->float:
        """
        Stetson K coefficient describing lightcurve shape.

        Parameters
        ----------
        None

        Returns
        -------
        K : float
            Stetson K coefficient.
        """

        average = self.WeightedMean()
        numerator = np.mean(
            np.abs(self.magnitude-average) / self.magnitudeErr
        )
        denominator = np.mean(
            (self.magnitude-average)**2 / self.magnitudeErr**2
        )**0.5

        K = (numerator/denominator).item()

        return K

    def WeightedMean(self)->float:
        """
        Weighted mean magnitude.

        Parameters
        ----------
        None

        Returns
        -------
        average : float
            The weighted mean magnitude.
        """

        average = np.average(
            self.magnitude, weights=self.magnitudeErr**(-2)
        ).item()

        return average

def extract_features(dataframe:pd.DataFrame, **kwargs)->dict:
    """
    Extract all available features using FeatureExtractor class.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame with the raw Swift-XRT lightcurve data.
    Returns
    -------
    features : dict
        The dictionary of extracted features.
    """

    obj = FeatureExtractor(dataframe)
    features = dict()
    for func_name in dir(obj):
        func = getattr(obj, func_name)
        if callable(func) and not func_name.startswith("__"):
            output = func()
            if hasattr(func, 'suffix'):
                for feature, suffix in zip(output, func.suffix):
                    features['_'.join([func_name, suffix])] = feature
            else:
                features[func_name] = output
    return features

def make_dataset(SwiftXRTdict:dict,
                 criterion:callable=complete_lightcurve,
                 preprocesser:callable=extract_features,
                 preprocesser_kwargs:dict=None,
                 random_state:int=20041120,
    )->tuple:
    """
    Produces dataset from the raw SwiftXRT data utilizing preprocesser.
    The train-val-test split procedure is designed in order to preserve
    the original distribution by the year of detection. This allows to
    avoid some undesired instrumental and local signatures in the data.

    Parameters
    ----------
    SwiftXRTdict : dict
        A dictionary in the format returned by read_SwiftXRT.
    criterion : callable, default=complete_lightcurve
        A function that returns True iff the passed dataframe 
        satisfies the user-defined requirements and False otherwise.
        The entries not satisfying the `criterion` will be excluded
        from the resulting dataset.
    preprocesser : callable, default=extract_features
        A function that preprocesses individual pandas dataframes 
        from `SwiftXRTdict` and returns a dict with the extracted
        information.
    preprocesser_kwargs : dict, default=None
        Optional keyword arguments to be passed to preprocesser.
    random_state : int, default=20041120
        Random state used in the train-val-test split.
        Default value is the Swift Observatory 
        launch date (20 Nov 2004), it guarantees that the
        outlier GRB 221009A is sent to the testing fragment.

    Returns
    -------
    train : pd.DataFrame
        Training fragment of the dataset.
    val : pd.DataFrame
        Validation fragment of the dataset.
    test : pd.DataFrame
        Test fragment of the dataset.
    """

    dataset = dict()
    if preprocesser_kwargs is None:
        preprocesser_kwargs = {}
    print('[Creating Dataset]: All available data collection modes')
    for event_name, datadict in tqdm(SwiftXRTdict.items()):
        dataframe = datadict['data']
        if (
            'masked_flares' in preprocesser_kwargs.keys()
            ):

            preprocesser_kwargs.update(
                'flares_list': datadict.get('Flares', [])
            )
            
        if criterion(dataframe):
            year = get_year(event_name)
            preprocessed = preprocesser(dataframe, **preprocesser_kwargs)
            preprocessed.update({'Year': year})
            dataset[event_name] = preprocessed
        else:
            continue
    dataset = pd.DataFrame.from_dict(dataset, orient='index').sort_index(axis=0)
    train, val_test = train_test_split(
        dataset,
        random_state=random_state,
        test_size=0.3,
        stratify=dataset['Year']
    )
    val, test = train_test_split(
        val_test,
        random_state=random_state,
        test_size=0.5,
        stratify=val_test['Year']
    )

    train.name, val.name, test.name = 'train', 'val', 'test'
    print(f'Successfully preprocessed {len(SwiftXRTdict)} lightcurves.')
    print(
        f'Found {len(dataset)} lightcurves satisfying the requirements.\n'+
        f'Preprocessing algorithm used: `{preprocesser.__name__}(...)`'
    )
    return (train, val, test)
