from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np


class Encoder(nn.Module):

    def __init__(self, input_shape: Tuple[int, int], hidden_dim: int,
                 latent_dim: int):

        super(Encoder, self).__init__()
        self._input_shape = input_shape
        self.linear1 = nn.Linear(input_shape[0] * input_shape[1], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: 'torch.tensor') -> 'torch.tensor':

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int,
                 output_shape: Tuple[int, int]):

        super(Decoder, self).__init__()
        self._output_shape = output_shape
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_shape[0] * output_shape[1])

    def forward(self, z: 'torch.tensor') -> 'torch.tensor':

        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        z = z.reshape((-1, self._output_shape[0], self._output_shape[1]))
        return z


class Autoencoder(nn.Module):

    def __init__(self, input_shape: Tuple[int, int], hidden_dim: int,
                 latent_dim: int):

        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_shape, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_shape)

    def forward(self, x: 'torch.tensor') -> 'torch.tensor':

        z = self.encoder(x)
        z = self.decoder(z)
        return z


class VariationalEncoder(nn.Module):

    def __init__(self, input_shape: Tuple[int, int], hidden_dim: int,
                 latent_dim: int):

        super(VariationalEncoder, self).__init__()
        self._input_shape = input_shape
        self.linear1 = nn.Linear(input_shape[0] * input_shape[1], hidden_dim)
        self.linear2a = nn.Linear(hidden_dim, latent_dim)
        self.linear2b = nn.Linear(hidden_dim, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0.0

    def forward(self, x: 'torch.tensor') -> 'torch.tensor':

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2a(x)
        sigma = torch.exp(self.linear2b(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z


class VAE(nn.Module):

    def __init__(self, input_shape: Tuple[int, int], hidden_dim: int,
                 latent_dim: int):

        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_shape, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_shape)

    def forward(self, x: 'torch.tensor') -> 'torch.tensor':

        z = self.encoder(x)
        z = self.decoder(z)
        return z


def train_vae(model: 'VAE', dataset: 'torch.utils.data.Dataset',
              epochs: int = 16, batch_size=16,
              verbose: int = 0) -> Tuple['torch.tensor', 'torch.tensor']:

    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    dataloader_train = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        train_loss = 0.0
        for x in dataloader_train:
            opt.zero_grad()
            x_hat = model(x)
            loss = ((x - x_hat)**2).sum() + model.encoder.kl
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(dataloader_train)
        if epoch % verbose == 0:
            print(f'Epoch: {epoch} | Train loss: {train_loss}')
    return model
