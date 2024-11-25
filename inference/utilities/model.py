import joblib
import os
import subprocess 
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from lightning import Trainer

PARENT_DIR_URL = (
    'https://raw.githubusercontent.com/'+
    'Nickolay-Martynenko/GRB-X-Ray-Afterglow/'+
    'main/models/AutoEncoder/Architectures'
)


class Encoder(nn.Module):
    def __init__(self, latent_dim:int,
                 architecture:tuple=(32, 4),
                 tseries_length:int=64):
        super().__init__()

        self.hidden_dims = [
            architecture[0]* 2**pow for pow in range(architecture[1])
            ]                                       # num of filters in layers
        self.tseries_length = tseries_length

        modules = []
        in_channels = 1                             # initial num of channels
        for h_dim in self.hidden_dims:              # conv layers
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,    # num of input channels
                        out_channels=h_dim,         # num of output channels
                        kernel_size=3,
                        stride=2,                   # convolution kernel step
                        padding=1,                  # save shape
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim                     # changing num of 
                                                    # input channels for 
                                                    # next iteration

        modules.append(nn.Flatten())                # to vector
        intermediate_dim = (
            self.hidden_dims[-1] * 
            self.tseries_length // (2**len(self.hidden_dims))
        )
        modules.append(nn.Linear(in_features=intermediate_dim,
                                 out_features=latent_dim))

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim:int,
                 architecture:tuple=(32, 4),
                 tseries_length:int=64):
        super().__init__()
        self.hidden_dims = [
            architecture[0]* 2**pow for pow in range(architecture[1]-1, 0, -1)
            ]                                       # num of filters in layers
        self.tseries_length = tseries_length

        intermediate_dim = (
            self.hidden_dims[0] * 
            self.tseries_length // (2**len(self.hidden_dims))
        )
        self.linear = nn.Linear(in_features=latent_dim,
                                out_features=intermediate_dim)

        modules = []
        for i in range(len(self.hidden_dims) - 1):  # define upsample layers
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(
                        in_channels=self.hidden_dims[i],
                        out_channels=self.hidden_dims[i + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm1d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(in_channels=self.hidden_dims[-1],
                          out_channels=1,
                          kernel_size=3, padding=1)
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self.linear(x)        # from latents space to Linear
        x = x.view(
            -1, self.hidden_dims[0],
            self.tseries_length // (2**len(self.hidden_dims))
            )                     # reshape
        x = self.decoder(x)       # reconstruction
        return x

class LitAE(L.LightningModule):
    def __init__(self, encoder, decoder, derivative_weight=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.derivative_weight = derivative_weight

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward_handler(self, data,
                        *args, **kwargs):
        # here is the logic how data is moved through AE
        latent = self.encoder(data)
        recon = self.decoder(latent)
        return latent, recon

    def loss_handler(self, recon, data, weight, latent,
                     *args, **kwargs):
        # here is the loss function computing
        recon_loss = torch.masked_select(
            input = F.mse_loss(
                recon, data, reduction='none'
            ) * weight,
            mask = weight.ge(0.0)
        )
        recon_loss = recon_loss.mean()

        # derivative penalty = 
        # L1-regularization of the output timeseries
        derivative_loss = torch.abs(
            torch.diff(recon, dim=-1)
        ).mean()

        # total loss
        loss = recon_loss + self.derivative_weight * derivative_loss

        return loss

    def training_step(self, batch, batch_idx):
        data, labels, weight = batch

        latent, recon = self.forward_handler(data, labels)
        loss = self.loss_handler(recon, data, weight, latent)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels, weight = batch

        latent, recon = self.forward_handler(data, labels)
        loss = self.loss_handler(recon, data, weight, latent)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_start(self):
        # create dict with empty tensors for further accumulating over batches
        self.test_result = defaultdict(torch.Tensor)

    def test_step(self, batch, batch_idx):
        data, labels, weight = batch

        latent, recon = self.forward_handler(data, labels)
        self.update_test_result(data, weight, recon, latent, labels)

    def update_test_result(self, data, weight, recon, latent, labels):
        # accumulating results every batch
        self.test_result['real'] = torch.cat(
            [self.test_result['real'], data.cpu()]
        )
        self.test_result['weight'] = torch.cat(
            [self.test_result['weight'], weight.cpu()]
        )
        self.test_result['recon'] = torch.cat(
            [self.test_result['recon'], recon.cpu()]
        )
        self.test_result['latent'] = torch.cat(
            [self.test_result['latent'], latent.cpu()]
        )
        self.test_result['labels'] = torch.cat(
            [self.test_result['labels'], labels.cpu()]
        )

    def on_test_epoch_end(self):
        # simply change type from torch tensor to numpy array
        # for every item in test_result dictionary
        for key in self.test_result:
            self.test_result[key] = self.test_result[key].numpy()
'''
def create_model(latent_dim:int=3, architecture:tuple=(32, 4)):
    """
    Creates autoencoder model instance
    """
    encoder, decoder = (
        Encoder(latent_dim=latent_dim, architecture=architecture),
        Decoder(latent_dim=latent_dim, architecture=architecture)
    )
    autoencoder = LitAE(encoder, decoder)
    return encoder, decoder, autoencoder
'''

def load_model(
    latent_dim=3, architecture=(32, 4),
    save_downloaded_checkpoint=False,
    use_local_path:bool=False,
    local_path:str='./best.ckpt'):
    """
    Loads model from checkpoint. By default, 
    checkpoint from GitHub is used. However user
    can load local checkpoint passing the 
    corresponding optional arguments.
    """
    if not use_local_path:
        exp_name = f'AE_dim={latent_dim}_archi=' + '%d_%d' % architecture
        url = PARENT_DIR_URL+f'/{exp_name}/best.ckpt'
        subprocess.run(['curl', '-o', './best_checkpoint_loaded_from_GitHub.ckpt', '-s',
                    '--show-error', f'{url}'])
        model = LitAE.load_from_checkpoint(
            './best_checkpoint_loaded_from_GitHub.ckpt',
            encoder=Encoder(latent_dim=latent_dim, architecture=architecture),
            decoder=Decoder(latent_dim=latent_dim, architecture=architecture)
        )

        if not save_downloaded_checkpoint:
            os.remove('./best_checkpoint_loaded_from_GitHub.ckpt')
    else:
        LitAE.load_from_checkpoint(
            local_path,
            encoder=Encoder(latent_dim=latent_dim, architecture=architecture),
            decoder=Decoder(latent_dim=latent_dim, architecture=architecture)
        )
        
    return model, Trainer(logger=False)

def load_scoring(
    latent_dim=3, architecture=(32, 4),
    save_downloaded_scoring=False,
    use_local_path:bool=False,
    local_path:str='./scoring.joblib'):
    
    if not use_local_path:
        exp_name = f'AE_dim={latent_dim}_archi=' + '%d_%d' % architecture
        url = PARENT_DIR_URL+f'/{exp_name}/scoring.joblib'
        subprocess.run(['curl', '-o', './scoring_loaded_from_GitHub.joblib', '-s',
                    '--show-error', f'{url}'])
        scoring = joblib.load('./scoring_loaded_from_GitHub.joblib')

        if not save_downloaded_scoring:
            os.remove('./scoring_loaded_from_GitHub.joblib')
    else:
        scoring = joblib.load(local_path)

    return scoring
    


