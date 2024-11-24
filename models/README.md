# Models
We compare the AutoEncoder model with topological-based (DBSCAN, KernelPCA) and ensemble-based (IsolationForest) unsupervised Machine Learning algorithms. A detailed review of scoring and feature extraction pipeline is given in the corresponding notebooks.

# Metrics
To compare between the models, we score samples and measure a connection between the scores and the presence of flares in the lightcurve. Although the latter is not exactly what one must call an *anomaly*, such a metric is a robust and well-defined  measure of how well the model captures real physical properties of the lightcurve morphology. 

Additionally, we look for a linear relationship between the extracted features and number of breaks in the lightcurve trend estimated by *Swift*-XRT analysis. 

Finally, we manually analyse top anomalous-scored samples, and try to argue what physical, morphological, or instrumental lightcurve features did the model look at, if possible.

# Usage
To make it easier to reporduce all the experiments without thinking of data loading and data preprocessing (e.g. in Google Colab), the data sets are downloaded directly from this repository using curl. If one prefers to use local paths instead of URLs, it can be achieved manually (see [`dataset`](/dataset) directory in this repository).

Before you start working with the notebooks, please read the <a href="/README.md/#setup"> [setup instructions]</a> and install the requirements listed in either [`requirements.txt`](requirements.txt) (for ML models) or [`AutoEncoder/requirements.txt`](AutoEncoder/requirements.txt) (for AutoEncoder).

# Load Pretrained AutoEncoder from Checkpoint
Training of the AutoEncoder model on a local device is remarkably time-consuming. 

To use the model without reproducing the entire training loop, the reader is encouraged to load a pre-trained model from a checkpoint, see [[docs]](https://lightning.ai/docs/pytorch/stable//common/checkpointing_basic.html). It is recommended to load AutoEncoder with *«Stadnard»* Architecture: [`best.ckpt`](./AutoEncoder/Architectures/AE_dim=3_archi=32_4/best.ckpt) together with the scoring function [`scoring.joblib`](./AutoEncoder/Architectures/AE_dim=3_archi=32_4/scoring.joblib). 

The following pseudo-code reflects the general steps:
```
# Step 0. Declare directory with a pretrained model
path = "./AutoEncoder/Architectures/AE_dim=3_archi=32_4"

# Step 1. Create LitAE model
model = LitAE(
    Encoder(latent_dim=3, architecture=(32, 4)),
    Decoder(latent_dim=3, architecture=(32, 4))
)

# Step 2. Create Trainer
trainer = lightning.Trainer()

# Step 3. Get predictions on your data
trainer.test(
    model,
    dataloader,
    ckpt_path=f"{path}/best.ckpt"
)

# Step 4. Get labels
labels = dataloader.dataset.label_enc.inverse_transform(
    model.test_result["labels"].astype(int)
)

# Step 5. Calculate weighted MSE on your data

real = model.test_result['real'].squeeze()
recon = model.test_result['recon'].squeeze()
weight = model.test_result['weight'].squeeze()

weightedMSE = (real-recon)**2 * weight
weightedMSE = np.ma.masked_array(
    data=weightedMSE,
    mask=~(weight.astype(bool))
)
weightedMSE = weightedMSE.mean(axis=1, keepdims=True)

# Step 6. Load scoring function
scoring = joblib.load(f"{path}/scoring.joblib")

# Step 7. Calculate the p-values
pvalues = np.exp(
    scoring(np.log10(weightedMSE))
)

# Step 8. Save result in a pandas DataFrame:
result = pd.DataFrame(
    index=labels,
    data=pvalues.reshape(-1, 1),
    columns=["p-value"]
)
```
