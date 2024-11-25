# Usage
**Step 0.**
Before you start working with the scripts, please read the <a href="/README.md/#setup"> [setup instructions]</a> and install the requirements listed in [`requirements.txt`](requirements.txt)

**Step 1.** Prepare your `<input-file>`. It should be GRB names you want to score, listed line-by-line. 

If some of the passed events:
- listed more than once, they will be processed only once
- do not correspond to any confirmed GRB in the *Swift*-XRT lightcurves repository, they will not be further processed
- have incomplete lightcurve (that is, include less than 4 data points or span less than 8 bins on the rebinned uniform log-time grid), they also will not be further processed

Note that for technical reasons, the *Swift*-XRT repository may sometimes be unavailable, even if your internet connection is stable. Whenever `ConnectTimeout` exception arises while requesting event, a corresponding warning will be shown. The time-outed request also will not be further processed.

Please look at an example input file [`example_input`](example_input) (a number of non-existing and non-GRB events are intentionally added for illustration).

**Step 2.** Once the requirements are installed and the `<input-file>` is prepared, you can run `score_samples` script. To see instructions:
```
(GRB_env) $ python ./utilities/score_samples.py -h
```
To use a default configuration (that is, to load the recommended model directly from this repository via curl), just run:
```
(GRB_env) $ python ./utilities/score_samples.py <input-file>
```
You can set the name of the output csv file via `-o` option.

Instead of using default configuration, it is possible to use non-default architectures available in [`/models/AutoEncoder/Architectures`](/models/AutoEncoder/Architectures). You can also load your own custom pre-trained AutoEncoder model checkpoint file `best.ckpt` together with a scoring function[^1] `scoring.joblib` from a local directory.

Assuming that your `<input-file>` includes 100 events in total, of which 20 have invalid names and another 30 have incomplete lightcurves, the output would be:
```
[Processing]: Sending request to the Swift-XRT repository...
Please be patient, this may be time-consuming
100%|████████████████████████████████████████████| 100/100 [01:02<00:00,  1.61it/s]
[Processing]: Found 75 out of 100 requested events
[Processing]: Rebinning in progress...
[Processing]: Rebinning finished
[Processing]: Job complete!
    complete_lightcurve    : 50 entries
    incomplete_lightcurve  : 30 entries
    not_found              : 20 entries
[Loading model]: In progress...
[Loading model]: Job complete!
[Predictions]: In progress...
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Testing DataLoader 0: 100%|████████████████████████| 1/1 [00:06<00:00,  8.33it/s]
[Predictions]: Job complete!
The output table is saved in the './output.csv' file
```
**Step 3.** The output is saved in the csv file (through the link [`output.csv`](output.csv) you can access the output produced for the [`example_input`](example_input) file). The event names are sorted alphabetically. The columns are: 
- **info**: the message generated while processing the event
  - `complete_lightcurve` : data collected and processed successfully
  - `incomplete_lightcurve` : the passed event name is valid, but the lightcurve is too short to analyse it
  - `missing_data` : the passed event name is valid, but neither PC_incbad nor WT_incbad dataset is available for this event, so nothing to process
  - `not_found` : the passed event name not found in the *Swift*-XRT repository (due to either invalid name or a connection timeout issue)
- **p-value**: for complete lightcurves, the estimated $p$-value; otherwise, empty cell

[^1]: Scoring function maps the decimal logarithm of a weighted mean squared reconstruction error to the natural logarithm of $p$-value