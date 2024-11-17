# Dataset & Related Utilities

## `./SwiftXRT`
Raw *Swift*-XRT GRB lightcurves repository data. See `./SwiftXRT/README.md` for a detailed discussion.

## `./utilities`
*Swift*-XRT data preprocessing utilities. Unless a specific user-defined preprocessing is needed, please use `./utilities/make_dataset.py`. Note that this is optional, since all the datasets are already created and can be found in `./Data` directory. 

To see instructions:
```
$ python3 ./utilities/make_dataset.py -h
```

Example usage:
```
$ python3 ./utilities/make_dataset.py -s SwiftXRT -n features -p extract_features    
```
The output would be:
```
[Processing]: PC_incbad
100%|███████████████████████████████| 1576/1576 [00:06<00:00, 233.78it/s]
[Processing]: WT_incbad
100%|█████████████████████████████████| 867/867 [00:04<00:00, 178.47it/s]
Successfully processed 2443 json files. Found 1579 unique events (only confirmed GRB).

[Creating Dataset]: All available data collection modes
100%|███████████████████████████████| 1579/1579 [00:08<00:00, 190.06it/s]
Successfully preprocessed 1579 lightcurves.
Found 1158 lightcurves satisfying the requirements.
Preprocessing algorithm used: `extract_features(...)`

Dataset created:
.
└── Data
    └── train
        └── features.csv        (810 entries)
    └── val
        └── features.csv        (174 entries)
    └── test
        └── features.csv        (174 entries)
```

## `./Data`
The train, val and test samples created by `./utilities/make_dataset.py` script in various regimes.
