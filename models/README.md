# Models
We compare the AutoEncoder model with topological-based (DBSCAN, KernelPCA) and ensemble-based (IsolationForest) unsupervised Machine Learning algorithms. A detailed review of scoring and feature extraction pipeline is given in the corresponding notebooks.

# Metrics
To compare between the models, we score samples and measure a connection between the scores and the presence of flares in the lightcurve using ROC AUC score. Although the latter is not exactly what one must call an *anomaly*, such a metric is a robust and well-defined  measure of how well the model captures real physical properties of the lightcurve morphology. 

Additionally, we look for a linear relationship between the extracted features and number of breaks in the lightcurve trend estimated by *Swift*-XRT analysis, fitting a simple linear regression to the extracted features – number-of-breaks relation and calculating the coefficient of determination (CoD). 

A summary of these experiments is given in the table below.

Feature extraction + Scoring | None + IsolationForest | KernelPCA + KDE | Isomap + DBSCAN | AutoEncoder + KDE |
--- | --- | --- | --- | --- |
**Dataset** | lightcurves rebinned with padding | extracted [[light curve features]](https://github.com/light-curve/light-curve-feature) | [[dynamic time warping]](https://en.wikipedia.org/wiki/Dynamic_time_warping) distance matrix for raw lightcurves | lightcurves rebinned with linear interpolation |
**ROC AUC/test** | 0.69 | 0.55 | 0.89 | 0.89 |
**ROC AUC/bins** | - | - | - | 0.69[^1] |
**CoD/test** | - | 0.20 | 0.35 | 0.45 |

Finally, we manually analyse top anomalous-scored samples, and try to argue what physical, morphological, or instrumental lightcurve features did the model look at, if possible.

We conclude that the AutoEncoder-based model is best-suitable for anomaly detection.

# Usage
To make it easier to reporduce all the experiments without thinking of data loading and data preprocessing (e.g. in Google Colab), the data sets are downloaded directly from the online version of this repository using [[curl]](https://curl.se). If one prefers to use local paths instead of URLs, it can be achieved manually (see [`dataset`](/dataset) directory in this repository).

Before you start working with the notebooks, please read the <a href="/README.md/#setup"> [setup instructions]</a> and install the requirements listed in either [`requirements.txt`](requirements.txt) (for ML models) or [`AutoEncoder/requirements.txt`](AutoEncoder/requirements.txt) (for AutoEncoder).

[^1]: An individual-bin analysis is possible only for the AutoEncoder+KDE model.
