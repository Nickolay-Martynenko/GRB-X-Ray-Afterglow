# Models
To make it easier to reporduce all the experiments without thinking of data loading and data preprocessing (e.g. in Google Colab), the data sets are downloaded directly from this repository using `curl`. If one prefers to use local paths instead of URLs, it can be achieved manually (see `dataset` directory in this repository).

## Models
We compare the AutoEncoder model with a number of Machine Learning algorithms, namely, DBSCAN, IsolationForest and KernelPCA.

## Metrics
To compare between the models, we score samples and measure a connection between the scores and the presence of flares in the lightcurve. Although the latter is not exactly what one must call an *anomaly*, such a metric is a robust and well-defined  measure of how well the model captures real physical properties of the lightcurve morphology.
We also manually look at top-10 anomalous-scored samples, and try to argue what physical/morphological/instrumental lightcurve features did the model look at, if possible.
