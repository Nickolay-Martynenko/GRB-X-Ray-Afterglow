# *Swift*-XRT Light Curves
This directory contains data downloaded from the *Swift*-XRT GRB lightcurve repository: [[link]](https://www.swift.ac.uk/xrt_curves/)

`./PC_incbad`: data points collected in a [`P`]hoton [`C`]ounting mode

`./WT_incbad`: data points collected in a [`W`]indow [`T`]iming settling mode

 [`inc`]luding possibly [`bad`] entries (i.e. unreliable data points according to *Swift* Analysis)

Each file `./$MODE/$EVENT_NAME.json` contains lightcurve data points measured for the event `$EVENT_NAME` in the mode `$MODE`. As a rule, if the event is classified as a GRB, `$EVENT_NAME` matches the following pattern:
`GRB YYMMDDX`, where `YYMMDD` denotes the date when the GRB was detected and `X` is an optional capital latin letter. 

For instance, `./PC_incbad/GRB 221009A.json` contains lightcurve data points in the Photon Counting mode including unreliable entries for the Gamma-Ray Burst GRB 221009A [[wiki]](https://en.wikipedia.org/wiki/GRB_221009A) detected on October 09, 2022.

For the further analysis within the present study, only these columns are used:
- `Time`: Trigger time, s
- `TimePos`, `TimeNeg`:  Positive and negative error in the trigger time, respectively
- `Rate`: Source count rate, 1/s
- `RatePos`, `RateNeg`: Positive and negative error in the source count rate (1-σ), respectively

See also a detailed lightcurve documentation: [[docs]](https://www.swift.ac.uk/xrt_curves/docs.php)
