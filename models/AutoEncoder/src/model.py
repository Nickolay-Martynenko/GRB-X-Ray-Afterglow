import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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