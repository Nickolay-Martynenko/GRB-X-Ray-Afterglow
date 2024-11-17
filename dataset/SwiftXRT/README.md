# *Swift*-XRT Light Curves
Data downloaded from the *Swift*-XRT GRB lightcurve repository: [[link]](https://www.swift.ac.uk/xrt_curves/)

- `./PC_incbad`: data points collected in a [`P`]hoton [`C`]ounting mode

- `./WT_incbad`: data points collected in a [`W`]indow [`T`]iming settling mode

 [`inc`]luding [`bad`] entries (i.e. unreliable data points according to *Swift* analysis)

Each file `./$MODE/$EVENT_NAME.json` contains lightcurve data points measured for the event `$EVENT_NAME` in the mode `$MODE`. 

If the event is a confirmed GRB, `$EVENT_NAME` matches the common-used pattern:
> GRBs are named after the date on which they are discovered: the first two digits being the year, followed by the two-digit month and two-digit day and a letter with the order they were detected during that day. The letter 'A' is appended to the name for the first burst identified, 'B' for the second, and so on. For bursts before the year 2010, this letter was only appended if more than one burst occurred that day.
>
> -- <cite>[*[wiki]*](https://en.wikipedia.org/wiki/Gamma-ray_burst)</cite>

For instance, `./PC_incbad/GRB 221009A.json` contains lightcurve data points in the Photon Counting mode including unreliable entries for the Gamma-Ray Burst GRB 221009A [[wiki]](https://en.wikipedia.org/wiki/GRB_221009A) detected on October 09, 2022.

For the further analysis within the present study, only these columns are used:
- `Time`: Trigger time, s
- `TimePos`, `TimeNeg`:  Positive and negative error in the trigger time, respectively
- `Rate`: Source count rate, 1/s
- `RatePos`, `RateNeg`: Positive and negative error in the source count rate (1-σ), respectively

See also a detailed lightcurve documentation: [[docs]](https://www.swift.ac.uk/xrt_curves/docs.php)
