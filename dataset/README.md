# *Swift*-XRT Light Curves
The lightcurves data are downloaded from the *Swift*-XRT GRB lightcurve [[repository]](https://www.swift.ac.uk/xrt_curves/) and are stored in [`./SwiftXRT`](./SwiftXRT) directory.

- [`./SwiftXRT/PC_incbad`](./SwiftXRT/PC_incbad): data points collected in a [`P`]hoton [`C`]ounting mode

- [`./SwiftXRT/WT_incbad`](./SwiftXRT/WT_incbad): data points collected in a [`W`]indow [`T`]iming settling mode

 [`inc`]luding [`bad`] entries (i.e. unreliable data points according to *Swift* analysis)

Each file `./$MODE/$EVENT_NAME.json` contains lightcurve data points measured for the event `$EVENT_NAME` in the mode `$MODE`. 

If the event is a confirmed GRB, `$EVENT_NAME` matches the common-used pattern:
> GRBs are named after the date on which they are discovered: the first two digits being the year, followed by the two-digit month and two-digit day and a letter with the order they were detected during that day. The letter 'A' is appended to the name for the first burst identified, 'B' for the second, and so on. For bursts before the year 2010, this letter was only appended if more than one burst occurred that day.
>
> -- <cite>[*[wiki]*](https://en.wikipedia.org/wiki/Gamma-ray_burst)</cite>

For instance, `./PC_incbad/GRB 221009A.json` contains lightcurve data points in the Photon Counting mode including unreliable entries for the Gamma-Ray Burst [[GRB 221009A]](https://en.wikipedia.org/wiki/GRB_221009A) detected on October 09, 2022.

For the further analysis within the present study, only these columns are used:
- `Time`: Trigger time, s
- `TimePos`, `TimeNeg`:  Positive and negative error in the trigger time, respectively
- `Rate`: Source count rate, 1/s
- `RatePos`, `RateNeg`: Positive and negative error in the source count rate (1-σ), respectively

See also a detailed lightcurve documentation: [[docs]](https://www.swift.ac.uk/xrt_curves/docs.php)

# Preprocessing
In our analysis, we develop four methods of the lightcurves preprocessing:
1. ***Original*** method: Applies decimal logarithm to both timestamps and count rates and subtract a typical stationary background of about 1 count per 1000 seconds. The source count rate +/- errors are symmetrized using their mean in the logarithmic scale (equivalent to a geometric mean in the original scale).
2. ***FeatureExtraction*** method: Extracts relevant statistical features[^1] from the raw timeseries and thus converts lightcurves dataset to tabular data. The source count rate +/- errors are symmetrized using their mean in the logarithmic scale (equivalent to a geometric mean in the original scale).
3. ***Padding*** method: *Original* lightcurves are rebinned to a uniform time grid in the decimal logarithm scale; missing values are padded with zeros.
4. ***Interpolation*** method: *Original* lightcurves are rebinned to a uniform time grid in the decimal logarithm scale; missing values are interpolated linearly using their closest non-missing neighbor entries.

Additionally, we filter all incomplete and non-GRB events, and split our datasets into training (`train`), validation (`val`), and testing (`test`) fragments (70% $\div$ 15% $\div$ 15%, respectively), preserving the original nearly-uniform proportion of each year of observations in the resulting subsets. The split does not depend on the preprocessing method.

All the preprocessing scripts are stored in [`./utilities`](utilities) directory.

# Prepared Dataset
[`./Data`](./Data) directory contains preprocessed [`train`](./Data/train), [`val`](./Data/val), and [`test`](./Data/test) samples for each preprocessing method listed above: `original.csv`, `features.csv`, `padded.csv`, and `interp.csv`, respectively. 

Additionally, a `GRBtable.csv` dataset is provided. It contains *Swift*-XRT analysis results semi-manually collected from the *Swift*-XRT repository.

# Usage
Before you run the preprocessing script, please read the <a href="/README.md/#setup"> [setup instructions]</a> and install the requirements listed in [`requirements.txt`](requirements.txt)

To see instructions:
```
(GRB_env) $ python ./utilities/make_dataset.py -h
```

Example usage:[^2]
```
(GRB_env) $ python ./utilities/make_dataset.py -s SwiftXRT -n <name> -p <preprocesser>    
```
The output would be:
```
[Processing]: PC_incbad
100%|███████████████████████| 1576/1576 [00:06<00:00, 233.78it/s]
[Processing]: WT_incbad
100%|█████████████████████████| 867/867 [00:04<00:00, 178.47it/s]
Successfully processed 2443 json files. Found 1579 unique events
(only confirmed GRB).

[Creating Dataset]: All available data collection modes
100%|███████████████████████| 1579/1579 [00:08<00:00, 190.06it/s]
Successfully preprocessed 1579 lightcurves.
Found 1158 lightcurves satisfying the requirements.
Preprocessing algorithm used: `<preprocesser>(...)`

Dataset created:
.
└── Data
    └── train
        └── <name>.csv        (810 entries)
    └── val
        └── <name>.csv        (174 entries)
    └─test
        └── <name>.csv        (174 entries)
```

[^1]: The feature exctraction procedure is close to that proposed by analogous Rust package [[docs]](https://docs.rs/light-curve-feature/latest/light_curve_feature/features/index.html) but has been tuned significantly to be appropriate for the GRB X-Ray afterglows analysis. See `FeatureExtractor` class in [`./utilities/SwiftXRTpreprocessing.py`](./utilities/SwiftXRTpreprocessing.py) for details

[^2]: Replace `<name>` and `<preprocesser>` with a desired filename and a valid preprocessing function name, respectively, e.g. `features` and `extract_features`. Add an optional argument via `-r` to set the rebinning method
