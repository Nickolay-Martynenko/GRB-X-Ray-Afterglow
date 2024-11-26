import argparse
import inspect
import os
import shutil
import pandas as pd
import numpy as np

from torch import no_grad
from lightning import Trainer
from dataset import rebin, get_SwiftXRTLightCurves, create_Dataloader
from model import load_model_scoring
from plots import plot_lightcurves

from warnings import simplefilter, filterwarnings
simplefilter("ignore", category=RuntimeWarning)
filterwarnings("ignore", ".*does not have many workers.*")

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str,
    help="your <input-file> with the list of events to be scored"
)
parser.add_argument("-o", "--output_file", type=str,
    help="the name of the <output-file> ('.csv' extension will be appended)",
    default="output"
)
parser.add_argument("--plot_lightcurves",  action=argparse.BooleanOptionalAction,
    help="whether to plot the real and reconstructed lightcurves"
)
parser.add_argument("-s", "--source", type=str,
    default="from_repo",
    help=(
        "local path where to search for the 'best.ckpt' and 'scoring.joblib' files "+
        "(by default, they are downloaded from repository via curl)"
    )
)
parser.add_argument("-d", "--dim", type=int,
    default=3,
    help=(
        "latent_dim of the AutoEncoder model"
    )
)
parser.add_argument("-a", "--archi", type=int, nargs=2,
    default=(32, 4),
    help=(
        "architecture of the AutoEncoder model"
    )
)
args = parser.parse_args()

# read event names
with open(args.filename, 'r') as inputfile:
    event_names = [line.rstrip('\n') for line in inputfile]

# request data from the Swift-XRT repository and create a DataLoader
lightcurves, info = get_SwiftXRTLightCurves(event_names)
dataloader = create_Dataloader(lightcurves)

# load model
print(f'[Loading model]: In progress...')
kwargs = {
    'latent_dim': args.dim,
    'architecture': args.archi
}
if args.source=='from_repo':
    kwargs.update({'use_local_path': False})
elif args.source is not None:
    kwargs.update({'use_local_path': True, 'local_path': args.source})
model, scoring = load_model_scoring(**kwargs)
model.eval()
print(f'[Loading model]: Job complete!')

# get predictions
print(f'[Predictions]: In progress...')
with no_grad():
    Trainer(logger=False).test(model, dataloader)
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

# plot lightcurves
if args.plot_lightcurves:

    # get the rebinning procedure info 
    signature = inspect.signature(rebin)
    bin_edges = np.linspace(
        signature.parameters['lgTime_min'].default,
        signature.parameters['lgTime_max'].default,
        signature.parameters['lgTime_nbins'].default+1)
    time_grid = (bin_edges[1:] + bin_edges[:-1])/2
    offset = (
        +3.0 if signature.parameters['subtract_background'].default 
        else 0.0
    )

    # plot
    if not os.path.isdir('./Figures'):
        os.mkdir('./Figures')
    plot_lightcurves(labels, real, recon, weight, time_grid, offset)

# save predictions
info['p-value'] = np.nan
info.loc[labels, 'p-value'] = p_value
info.to_csv(f'{args.output_file}.csv')

print(f'[Predictions]: Job complete!')
print(f"The output table is saved in the './{args.output_file}.csv' file")

shutil.rmtree(f"{os.path.dirname(__file__)}/__pycache__")

