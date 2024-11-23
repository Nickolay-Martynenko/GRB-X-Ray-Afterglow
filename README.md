# Uncovering Anomalies in Gamma-Ray Bursts X-Ray Afterglows

All the scripts and notebooks in this repository have been tested with python 3.11.5 and Jupyter Notebook 6.4.13 on macOS 15.0.1 (24A348). Before trying to run anything on your device, please pay attention to the fact that directories in this repository contain their individual  `requirements.txt` files. List of packages provided in each `requirements.txt` file assumes that you have already installed python and Jupyter Notebook and the corresponding dependencies on your device.
 

# Setup 
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
- [`dataset`](dataset) — to create/preprocess dataset
- [`models`](models) — to work with Machine Learning analysis
- [`models/AutoEncoder`](models/AutoEncoder) — to work with AutoEncoder model
   
Then install the required packages and Jupyter Notebook 6.4.13 (the latter is optional for creating/preprocessing dataset)
```
(GRB_env) $ cd ./<directory>
(GRB_env) $ python -m pip install -r requirements.txt [notebook==6.4.13]
```
You can now work with the scripts and notebooks in the corresponding folder. For details, please read README.md in there.

## Machine Learning Models 
Create a new `python 3.11.5` environment:
```
$ conda create -n GRB_ML_env python==3.11.5 -y
```
Next, activate the environment:
```
$ conda activate GRB_ML_env
```
From the current directory (where you read this guide), switch to the `models` folder, and then install and the required packages:
```
(GRB_ML_env) $ cd ./models
(GRB_ML_env) $ python -m pip install -r requirements.txt 
```
You can now work with [[DBSCAN]](models/DBSCAN/DBSCAN.ipynb), [[IsolationForest]](models/DBSCAN/IsolationForest.ipynb), and [[KernelPCA]](models/KernelPCA/KernelPCA.ipynb) notebooks. For details, please read [[models/README.md]](models/README.md)

When your work with these notebooks is finished, deactivate the environment:
```
(GRB_ML_env) $ conda deactivate
```

## AutoEncoder Model
Create a new `python 3.11.5` environment:
```
$ conda create -n GRB_AE_env python==3.11.5 -y
```
Next, activate the environment:
```
$ conda activate GRB_AE_env
```
From the current directory (where you read this guide), switch to the `models/AutoEncoder` folder, and then install Jupyter Notebook 6.4.13 and the required packages:
```
(GRB_AE_env) $ cd ./models/AutoEncoder
(GRB_AE_env) $ python -m pip install -r requirements.txt notebook==6.4.13
```
You can now work with [[AutoEncoder]](models/AutoEncoder/AutoEncoder.ipynb) notebook. For details, please read [[models/README.md]](models/README.md)

When your work with these notebooks is finished, deactivate the environment:
```
(GRB_AE_env) $ conda deactivate
```


[^1]: Of course, you can assign the environment names at your discretion
