# Uncovering Anomalies in Gamma-Ray Bursts X-Ray Afterglows
<img src="figures/GRB221009Awide.gif" />

A great diversity of the Gamma-Ray Bursts (GRBs) *prompt emission* together with a relatively small number of detected events (several thousand) significantly limit the use of statistical tools in distinguishing between the “ordinary” and “anomalous” objects.  In contrast, the X-Ray *afterglow emission* exhibits a high universality, which makes the statistical approach possible.

However, the present-day analysis of GRB X-Ray afterglow lightcurves is highly model-dependent and probably oversimplified, which obstructs a robust anomaly detection. Here we address this issue and develop a number of model-independent anomaly detection techniques utilizing Machine Learning and Deep Learning models. 

In a few words, we are looking for such model that:
- encodes an X-Ray lightcurve in terms of 2-3 features, which correlate with the lightcurve morphology (e.g. with the number of breaks)
- estimates the anomaly measure for each lightcurve and, using this measure, detects at least conventionally anomalous GRBs such as GRB 221009A (its X-Ray afterglow lightcurve together with the corresponding pseudo-color timelapse animation, both measured by *Swift*-XRT, are shown in the figure above)

We demonstrate that AutoEncoder is a promising model satisfying these criteria.

## Inference
The main product presented by this project is a pre-trained AutoEncoder-based GRB anomaly detector employing `swifttools` API. The detector works in a user-friendly “black-box” manner, without *requiring* the user to understand what is going on, but still being flexible enough for those who wants to manage the parameters.

[`inference/README.md`](inference/README.md) presents a complete User's Guide for this detector.

## Dataset
This study utilizes the *Swift*-XRT GRB lightcurve [[repository]](https://www.swift.ac.uk/xrt_curves/) data. The dataset is briefly described in [`dataset/README.md`](dataset/README.md)

We focus only on the basic lightcurve entries: the timestamps and source count rate, together with the errors of the latter. We develop and use four preprocessing methods, see [`dataset/README.md`](dataset/README.md)

## Models
We analyse lightcurves using Density-based spatial clustering of applications with noise (DBSCAN), IsolaitonForest, Kernel principal component analysis (KernelPCA), and AutoEncoder. Each model is accompanied by  its own pipeline of pre-/postprocessing steps. See  [`models/README.md`](models/README.md) for details.

## Tested Environment

| Operating System  | Python Version | IDE/Editor | Hardware |
| --- | --- | --- | --- |
| macOS Sequoia 15.0.1 | 3.11.5 | Jupyter Notebook 6.4.13 | MacBook Air (M1, 8GB RAM) |

Before trying to run anything on your device, please pay attention to the fact that:
- directories in this repository contain their individual  `requirements.txt` files,
- list of packages provided in each `requirements.txt` file assumes that you have already installed python and Jupyter Notebook, as well as the corresponding dependencies on your device.

<h2 id="setup">Setup with Conda (recommended)</h2>

Please consider installing [[miniconda3]](https://docs.anaconda.com/miniconda/install/) and working within individual [[environments]](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Following the instructions given below will ensure reproducibility, assuming that you work with code locally on your device.

**Step 1.** Create a new `python 3.11.5` environment:[^*]
```
$ conda create -n GRB_env python==3.11.5 -y
```
**Step 2.** Next, activate the environment:
```
$ conda activate GRB_env
```
**Step 3.** From the current directory (where you read this guide), switch to the folder `<directory>`:
- [`inference`](inference) — to work with a pre-trained anomaly detector based on AutoEncoder model,
- [`dataset`](dataset) — to create/preprocess dataset,
- [`models`](models) — to work with Machine Learning models,
- [`models/AutoEncoder`](models/AutoEncoder) — to work with AutoEncoder model,

   
It is highly recommended to have an individual environment for each of these options. Then install the required packages and Jupyter Notebook 6.4.13 (the latter is optional if  `<directory>` is `inference` or `dataset`)
```
(GRB_env) $ cd ./<directory>
(GRB_env) $ python -m pip install -r requirements.txt [notebook==6.4.13]
```
You can now work with:
- [`inference/utilities`](inference/utilities) python scripts,
- [`dataset/utilities`](dataset/utilities) python scripts,
- [`DBSCAN.ipynb`](models/DBSCAN/DBSCAN.ipynb), [`IsolationForest.ipynb`](models/IsolationForest/IsolationForest.ipynb), and [`KernelPCA.ipynb`](models/KernelPCA/KernelPCA.ipynb) notebooks,
- [`AutoEncoder.ipynb`](models/AutoEncoder/AutoEncoder.ipynb) notebook,
  
respectively. For details, please read `<directory>/README.md`. 

**Step 4.** When your work is finished, deactivate the environment:
```
(GRB_env) $ conda deactivate
```
If you do not need the created environment anymore, you can permanently remove it by:
```
$ conda remove -n GRB_env --all -y
```

## Contacts
If you have any comments, suggestions, or questions, please send them to [martynenko.ns18@physics.msu.ru](mailto:martynenko.ns18@physics.msu.ru?subject=GRB-X-Ray-Afterglow)

## Acknowledgements
This work is greatly supported by Non-commercial Foundation for the Advancement of Science and Education INTELLECT.
The author is indebted to G.I. Rubtsov, V.A. Nemchenko, and A.V. Ivchenko for helpful discussions.

[^*]: Of course, istead of `GRB_env` you can assign the environment names at your discretion
