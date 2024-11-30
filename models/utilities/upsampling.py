import numpy as np
import pandas as pd


def upsampling(
    dataframe:pd.DataFrame,
    ratio:int=100,
    noise_amplitude:float=0.1,
    shift_amplitude:int=3,
    random_state:int=42)->pd.DataFrame:
    """
    Augmented upsampling of 
    the lightcurves dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to be upsampled.
    ratio : int, default=100
        The upsampling ratio.
    noise_amplitude : float, default=0.1
        The gaussian noise amplitude.
        Noise is applied to the decimal
        logarithm of the count rate.
        The default value is approximately
        equal to a typical systematic error.
    shift_amplitude : int, default=3
        The timeseries shift augmentation
        amplitude. The edges are truncated.

    Returns
    -------
    upsampled : pd.DataFrame
        The upsampled dataframe.
    """

    np.random.seed(random_state)

    upsampled = dataframe.sample(
        frac=ratio,
        replace=True,
        random_state=random_state)

    rate, weight, flares = (
        np.array(upsampled.loc[:, col].tolist(), dtype='float32')
        for col in ['lgRate', 'weight', 'flares']
    )

    n_samples, n_bins = rate.shape
    pseudo_t = np.arange(n_bins)

    # random gaussian noise
    noise = np.random.normal(loc=0,
                             scale=noise_amplitude,
                             size=(n_samples, n_bins)
                             ) * weight.astype(bool).astype(int)
    rate += noise

    # random time series shift
    roll = np.random.randint(
        low=-shift_amplitude,
        high=shift_amplitude+1,
        size=n_samples
        )

    for shift in range(-shift_amplitude, shift_amplitude+1):
        if shift == 0:
            continue
        else:
            rows = (roll==shift)

            rate[rows, :] = np.roll(rate[rows, :], shift, axis=1)
            weight[rows, :] = np.roll(weight[rows, :], shift, axis=1)
            flares[rows, :] = np.roll(flares[rows, :], shift, axis=1)

            insert = slice(0,shift) if shift>0 else slice(n_bins+shift,n_bins)

            weight[rows, insert] = 0.0
            flares[rows, insert] = 0.0

            for r in np.argwhere(rows).ravel():
                non_empty = weight[r, :] > 0
                rate[r, insert] = np.interp(
                    pseudo_t[insert],
                    pseudo_t[non_empty],
                    rate[r, non_empty]
                )

    upsampled['lgRate'] = rate.tolist()
    upsampled['weight'] = weight.tolist()
    upsampled['flares'] = flares.tolist()
    upsampled['weight_masked_flares'] = ( weight * (1.0-flares) ).tolist()

    return upsampled