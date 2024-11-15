# Dataset & Basic Preprocessing Utility

`./SwiftXRT`: Raw *Swift*-XRT GRB lightcurves repository data. See `./SwiftXRT/README.md` for a detailed discussion.

---
`./SwiftXRT_lightcurves_to_pickle.py`: basic preprocessing script. Example usage:
```
$ python3 SwiftXRT_lightcurves_to_pickle.py SwiftXRT SwiftXRT_Dataset.pickle
```
The output would be:
```
[Processing]: PC_incbad
100%|██████████████████████████████████████████████████| 1577/1577 [00:05<00:00, 302.38it/s]
[Processing]: WT_incbad
100%|████████████████████████████████████████████████████| 867/867 [00:04<00:00, 205.28it/s]
Successfully processed 2444 json files. Found 1580 unique events
The result is stored at 'SwiftXRT_Dataset.pickle'
```

---
`./SwiftXRT_Dataset.pickle`: the basic preprocessing output file. A dictionary, where the keys are the event names, and the values contain the corresponding lightcurves encoded as pandas.DataFrame instances.
