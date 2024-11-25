import fileinput
import os
import shutil

import pandas as pd
import numpy as np
from torch import no_grad

from warnings import simplefilter, filterwarnings
simplefilter("ignore", category=RuntimeWarning)
filterwarnings("ignore", ".*does not have many workers.*")

from dataset import get_SwiftXRTLightCurves, create_Dataloader
from model import load_model, load_scoring

event_names = []
for line in fileinput.input(encoding='utf-8'):
    event_names.append(line.removesuffix('\n'))

lightcurves, info = get_SwiftXRTLightCurves(event_names)
dataloader = create_Dataloader(lightcurves)

print(f'[Loading model]: In progress...')
model, trainer = load_model()
model.eval()

scoring = load_scoring()
print(f'[Loading model]: Job complete!')

print(f'[Predictions]: In progress...')
with no_grad():
    trainer.test(model, dataloader)

labels = dataloader.dataset.label_enc.inverse_transform(
    model.test_result['labels'].astype(int)
)
real = model.test_result['real'].squeeze()
recon = model.test_result['recon'].squeeze()
weight = model.test_result['weight'].squeeze()

weightedMSE = (real-recon)**2 * weight
weightedMSE = np.ma.masked_array(
    data=weightedMSE, mask=~(weight.astype(bool))
).mean(axis=1)

p_value = np.exp(scoring(np.log10(weightedMSE)))
info['p-value'] = np.nan
info.loc[labels, 'p-value'] = p_value
info.to_csv('scored_samples.csv')

print(f'[Predictions]: Job complete!')
print("The full output table is stored at './scored_samples.csv'")

shutil.rmtree(f"{os.path.dirname(__file__)}/__pycache__")
shutil.rmtree(f"{os.path.dirname(__file__)}/../../dataset/utilities/__pycache__")

