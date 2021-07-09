from typing import Tuple
import torch
import torch as T
import torch.nn as nn

class ChemVAE(nn.Module):

    def __init__(self, input_dim: int = 13, latent_dim: int = 4):

        super(ChemVAE, self).__init__()
        self._input_dim = input_dim
        self._latent_dim = latent_dim

        self.fc1 = nn.Linear(self._input_dim, 32)
        self.fc2a = nn.Linear(32, self._latent_dim)
        self.fc2b = nn.Linear(32, self._latent_dim)
        self.fc3 = nn.Linear(self._latent_dim, 32)
        self.fc4 = nn.Linear(32, self._input_dim)

    def encode(self, x: 'torch.tensor') -> Tuple['torch.tensor', 'torch.tensor']:

        z = T.sigmoid(self.fc1(x))
        mean = T.sigmoid(self.fc2a(z))
        logvar = T.sigmoid(self.fc2b(z))
        return (mean, logvar)

    def decode(self, z: 'torch.tensor') -> 'torch.tensor':

        z = T.sigmoid(self.fc3(z))
        z = T.sigmoid(self.fc4(z))
        return z

    @staticmethod
    def generate(mean: 'torch.tensor', logvar: 'torch.tensor') -> 'torch.tensor':

        stdev = T.exp(0.5 * logvar)
        noise = T.randn_like(stdev)
        return mean + (noise * stdev)

    def forward(self, x: 'torch.tensor') -> Tuple['torch.tensor', 'torch.tensor', 'torch.tensor']:

        mean, logvar = self.encode(x)
        inpt = self.generate(mean, logvar)
        recon_x = self.decode(inpt)
        return (recon_x, mean, logvar)


def elbo_loss(recon_x: 'torch.tensor', x: 'torch.tensor', mean: 'torch.tensor', logvar: 'torch.tensor', beta: float = 1.0):
    # https://arxiv.org/abs/1312.6114

    bce = T.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * T.sum(1 + logvar - T.pow(mean, 2) - T.exp(logvar))
    return bce + (beta * kld)
