# Usage
**Step 0.**
Before you start working with the scripts, please read the <a href="/README.md/#setup"> [setup instructions]</a> and install the requirements listed in [`requirements.txt`](requirements.txt).

**Step 1.** Prepare `<input-file`. It should be GRB names you want to score, listed line-by-line. If some names do not correspond to any GRB in the *Swift*-XRT lightcurves repository, they will not be further processed. If some names are listed more than once, they will be processed only once. If some of the found lightcurves are incomplete (that is, include less than 4 data points or span less than 8 bins on the rebinned grid), they also will not be further processed.

Please look at an example input file [`example_input`](example_input). Non-existing and non-GRB events are intentionally added for illustration.

**Step 2.** Once the `<input-file>` is prepared and the requirements are installed, run:
```
(GRB_env) $ python ./utilities/score_samples.py <input-file>
```
Assuming that your `<input-file>` includes 100 events in total, of which 25 have invalid names and another 25 have incomplete lightcurves:
```
[Processing]: Sending request to the Swift-XRT repository...
[Processing]: Please be patient, this may be time-consuming
[Processing]: Found 75 out of 100 requested events
[Processing]: Rebinning in progress...
[Processing]: Rebinning finished
[Processing]: Job complete!
    complete_lightcurve   : 50 entries
    incomplete_lightcurve : 25 entries
    not_found             : 25 entries
[Loading model]: In progress...
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
[Loading model]: Job complete!
[Predictions]: In progress...
Testing DataLoader 0: 100%|██████████████████████| 1/1 [00:00<00:00,  7.37it/s]
[Predictions]: Job complete!
The output table is available at './scored_samples.csv'
```
**Step 3.** Look at `./scored_samples.csv`: in this 
