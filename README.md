# Uncovering Anomalies in Gamma-Ray Bursts X-Ray Afterglows

# General Information and Instructions
All the scripts and notebooks in this repository have been tested by the author with python 3.11.5 and Jupyter Notebook 6.4.13 on macOS Sequoia 15.0.1 (MacBook Air, 8GB RAM, M1 chip). The AutoEncoder model was trained on CPU, but the code is flexible enough to use GPU if available.

Before trying to run anything on your device, please pay attention to the fact that:
- directories in this repository contain their individual  `requirements.txt` files,
- list of packages provided in each `requirements.txt` file assumes that you have already installed python and Jupyter Notebook, as well as the corresponding dependencies on your device.

# Setup with Conda (recommended)
Please consider installing [[miniconda3]](https://docs.anaconda.com/miniconda/install/) and working within individual [[environments]](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Following the instructions given below will ensure reproducibility, assuming that you work with code locally on your device.

**Step 1.** Create a new `python 3.11.5` environment:[^1]
```
$ conda create -n GRB_env python==3.11.5 -y
```
**Step 2.** Next, activate the environment:
```
$ conda activate GRB_env
```
**Step 3.** From the current directory (where you read this guide), switch to the folder `<directory>`:
- [`dataset`](dataset) — to create/preprocess dataset,
- [`models`](models) — to work with Machine Learning models,
- [`models/AutoEncoder`](models/AutoEncoder) — to work with AutoEncoder model.
   
It is highly recommended to have an individual environment for each of these options. Then install the required packages and Jupyter Notebook 6.4.13 (the latter is optional if  `<directory>` is  `dataset`)
```
(GRB_env) $ cd ./<directory>
(GRB_env) $ python -m pip install -r requirements.txt [notebook==6.4.13]
```
You can now work with:
- [`dataset/utilities`](dataset/utilities) python scripts,
- [`DBSCAN.ipynb`](models/DBSCAN/DBSCAN.ipynb), [`IsolationForest.ipynb`](models/IsolationForest/IsolationForest.ipynb), and [`KernelPCA.ipynb`](models/KernelPCA/KernelPCA.ipynb) notebooks,
- [`AutoEncoder.ipynb`](models/AutoEncoder/AutoEncoder.ipynb) notebook,
  
respectively. For details, please read `<directory>/README.md`. When your work is finished, deactivate the environment:
```
(GRB_env) $ conda deactivate
```

[^1]: Of course, you can assign the environment names at your discretion
