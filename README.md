# Uncovering Anomalies in Gamma-Ray Bursts X-Ray Afterglows

All the scripts and notebooks in this repository have been tested with python 3.11.5 and Jupyter Notebook 6.4.13 on macOS 15.0.1 (24A348). Before trying to run anything on your device, please pay attention to the fact that directories in this repository contain their individual  `requirements.txt` files. List of packages provided in each `requirements.txt` file assumes that you have already installed python and Jupyter Notebook and the corresponding dependencies on your device.
 

# Setup 
Please consider installing [[miniconda3]](https://docs.anaconda.com/miniconda/install/) and working within individual [[environments]](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Following the instructions given below will ensure reproducibility. 

## Creating Dataset
Create a new `python 3.11.5` environment:[^1]
```
$ conda create -n GRB_dataset_env python==3.11.5 -y
```
Next, activate the environment:
```
$ conda activate GRB_dataset_env
```
From the current directory (where you read this guide), switch to the `dataset` folder:
```
(GRB_dataset_env) $ cd ./dataset
```
Install the required packages:
```
(GRB_dataset_env) $ python -m pip install -r requirements.txt
```
You can now work with dataset creating/preprocessing utilities. For details, please read [[README.md]](dataset/README.md)

### Machine Learning Algorithms 
Install `Jupyter Notebook 6.4.13`:
```
(GRB_dataset_env) $ python -m pip install notebook==6.4.13
```

[^1]: Of course, you can assign the environment name at your discretion
