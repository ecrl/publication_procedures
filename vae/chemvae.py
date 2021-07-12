from typing import Tuple
import torch
import torch as T
import torch.nn as nn


class ChemVAE(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 4, hidden_dim: dict = 8):
        """
        ChemVAE: variational auto-encoder leveraging ELBO loss, designed to
        encode/decode SMILES strings into/out of a probabilistic latent space;
        child of torch.nn.Module

        Args:
            input_dim (int): number of input features; also acts as the number
                of outputs when decoding
            latent_dim (int, optional): number of features/probabilitic
                distributions in the encoder's latent space (default: 4)
            hidden_dim (int, optional): number of neurons between input/latent
                and latent/output (default: 8)
        """

        super(ChemVAE, self).__init__()
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self._input_dim, self._hidden_dim)
        self.fc2a = nn.Linear(self._hidden_dim, self._latent_dim)
        self.fc2b = nn.Linear(self._hidden_dim, self._latent_dim)
        self.fc3 = nn.Linear(self._latent_dim, self._hidden_dim)
        self.fc4 = nn.Linear(self._hidden_dim, self._input_dim)

    def encode(self, x: 'torch.tensor') -> Tuple['torch.tensor', 'torch.tensor']:
        """
        Encodes input data to latent distributions

        Args:
            x (torch.tensor): input/unencoded data, shape [n_samples,
                n_features]

        Returns:
            Tuple[torch.tensor, torch.tensor]: (latent distr. means, latent
                distr. logvar i.e. log(sigma^2)); shape of means and logvars is
                [n_samples, latent_dim]
        """

        z = T.sigmoid(self.fc1(x))
        mean = T.sigmoid(self.fc2a(z))
        logvar = T.sigmoid(self.fc2b(z))
        return (mean, logvar)

    def decode(self, z: 'torch.tensor') -> 'torch.tensor':
        """
        Given samples (assumed from mean/logvar distribution), decode to
        original dimensionality

        Args:
            z (torch.tensor): samples, shape [n_samples, latent_dim]

        Returns:
            torch.tensor: decoded data, shape [n_samples, n_features]
        """

        z = T.sigmoid(self.fc3(z))
        z = T.sigmoid(self.fc4(z))
        return z

    @staticmethod
    def generate(mean: 'torch.tensor', logvar: 'torch.tensor') -> 'torch.tensor':
        """
        Given latent means and logvars, generate data in latent dimension;
        generation follows:

        $ mu + (N(0,1) * e^{0.5 * logvar})$

        Args:
            mean (torch.tensor): latent distribution mean values, shape
                [n_samples, latent_dim]
            logvar (torch.tensor): latent distribution logvar values, shape
                [n_samples, latent_dim]

        Returns:
            torch.tensor: generated data, shape [n_samples, latent_dim]
        """

        stdev = T.exp(0.5 * logvar)
        noise = T.randn_like(stdev)
        return mean + (noise * stdev)

    def forward(self, x: 'torch.tensor') -> Tuple['torch.tensor', 'torch.tensor', 'torch.tensor']:
        """
        torch.nn.Module forward operation; given input data, return
        reconstructed data, mean/logvar in latent space

        Args:
            x (torch.tensor): input data, shape [n_samples, n_features]

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]: (reconstructed
                data, latent mean, latent logvar); reconstructed data of shape
                [n_samples, n_features], latent mean/logvar of shape
                [n_samples, latent_dim]
        """

        mean, logvar = self.encode(x)
        inpt = self.generate(mean, logvar)
        recon_x = self.decode(inpt)
        return (recon_x, mean, logvar)


def elbo_loss(recon_x: 'torch.tensor', x: 'torch.tensor', mean: 'torch.tensor', logvar: 'torch.tensor', beta: float = 1.0):
    """
    Evidence lower bound (ELBO) loss function, from
    https://arxiv.org/abs/1312.6114

    Args:
        recon_x (torch.tensor): reconstructed/decoded data, shape [n_samples,
            n_features]
        x (torch.tensor): input data, shape [n_samples, n_features]
        mean (torch.tensor): latent mean, shape [n_samples, latent_dim]
        logvar (torch.tensor): latent logvar, shape [n_samples, latent_dim]
        beta (float, optional): scalar for Kullback-Leibler divergence portion
            of loss (default: 1.0)
    """
    # https://arxiv.org/abs/1312.6114

    bce = T.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * T.sum(1 + logvar - T.pow(mean, 2) - T.exp(logvar))
    return bce + (beta * kld)
