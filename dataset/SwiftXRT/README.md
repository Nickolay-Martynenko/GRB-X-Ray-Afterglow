# Swift XRT Light Curves
This directory contains data downloaded from the *Swift*-XRT GRB lightcurve repository: [[link]](https://www.swift.ac.uk/xrt_curves/)

`./PC_incbad`: data points collected in a [`P`]hoton [`C`]ounting mode

`./WT_incbad`: data points collected in a [`W`]indow [`T`]iming settling mode

 [`inc`]luding possibly [`bad`] entries (i.e. unreliable data points according to *Swift* Analysis)

Each file `./$MODE/$EVENT.json` contains lightcurve data points measured for the event `$EVENT` in the mode `$MODE`.

For the further analysis within the present study, only these columns are used:
- `Time`: Trigger time, s
- `TimePos`, `TimeNeg`:  Positive and negative error in the trigger time, respectively
- `Rate`: Source count rate, 1/s
- `RatePos`, `RateNeg`: Positive and negative error in the source count rate (1-σ), respectively

See also a detailed documentation: [[docs]](https://www.swift.ac.uk/xrt_curves/docs.php)
