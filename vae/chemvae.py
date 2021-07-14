from typing import Tuple, List
import sys

import torch
import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from callbacks import CallbackOperator, Callback


class Validator(Callback):

    def __init__(self, loader, model, patience: int, beta: float):
        """
        Periodic validation using training data subset

        Args:
            loader (torch.utils.data.DataLoader): validation set
            model (ChemVAE): model being trained
            patience (int): if new lowest validation loss not found after `this` many epochs,
                terminate training, set model parameters to those observed @ lowest validation loss
        """

        super().__init__()
        self.loader = loader
        self.model = model
        self._best_loss = sys.maxsize
        self._most_recent_loss = sys.maxsize
        self._epoch_since_best = 0
        self.best_state = model.state_dict()
        self._patience = patience
        self._beta = beta

    def on_epoch_end(self, epoch: int) -> bool:
        """
        Training halted if:
            number of epochs since last lowest valid. MAE > specified patience
        """

        valid_loss = 0.0
        for batch in self.loader:
            recon_x, mean, logvar = self.model(batch)
            loss = elbo_loss(recon_x, batch, mean, logvar, self._beta)
            valid_loss += loss.item()
        valid_loss /= len(self.loader)
        self._most_recent_loss = valid_loss
        if valid_loss < self._best_loss:
            self._best_loss = valid_loss
            self.best_state = self.model.state_dict()
            self._epoch_since_best = 0
            return True
        self._epoch_since_best += 1
        if self._epoch_since_best > self._patience:
            return False
        return True

    def on_train_end(self) -> bool:
        """
        After training, recall weights when lowest valid. MAE occurred
        """

        self.model.load_state_dict(self.best_state)
        return True


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
        self.fc2 = nn.Linear(self._hidden_dim, self._hidden_dim)
        self.fc3a = nn.Linear(self._hidden_dim, self._latent_dim)
        self.fc3b = nn.Linear(self._hidden_dim, self._latent_dim)
        self.fc4 = nn.Linear(self._latent_dim, self._hidden_dim)
        self.fc5 = nn.Linear(self._hidden_dim, self._hidden_dim)
        self.fc6 = nn.Linear(self._hidden_dim, self._input_dim)
        # self.out = nn.Softmax(dim=1)

    def fit(self, dataset: 'torch.utils.data.Dataset',
            valid_size: float = 0.0,
            patience: int = 8,
            random_state: int = None,
            batch_size: int = 32,
            epochs: int = 64,
            shuffle: bool = False,
            beta: float = 1.0,
            verbose: int = 0,
            **kwargs) -> Tuple['torch.tensor', 'torch.tensor']:

        CBO = CallbackOperator()

        if valid_size > 0.0:
            _validate = True
            index_train, index_valid = train_test_split(
                [i for i in range(len(dataset))], test_size=valid_size,
                random_state=random_state
            )
            dataloader_train = DataLoader(
                Subset(dataset, index_train), batch_size=batch_size,
                shuffle=True
            )
            dataloader_valid = DataLoader(
                Subset(dataset, index_valid), batch_size=batch_size,
                shuffle=True
            )
            _validator = Validator(dataloader_valid, self, 1, patience)
            CBO.add_cb(_validator)
        else:
            _validate = False
            dataloader_train = DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )

        optimizer = torch.optim.Adam(self.parameters(), **kwargs)

        train_losses, valid_losses = [], []
        CBO.on_train_begin()
        for epoch in range(epochs):

            if not CBO.on_epoch_begin(epoch):
                break

            if shuffle:
                index_train, index_valid = train_test_split(
                    [i for i in range(len(dataset))], test_size=valid_size,
                    random_state=random_state
                )
                dataloader_train = DataLoader(
                    Subset(dataset, index_train), batch_size=batch_size,
                    shuffle=True
                )
                dataloader_valid = DataLoader(
                    Subset(dataset, index_valid), batch_size=len(index_valid),
                    shuffle=True
                )

            train_loss = 0.0
            self.train()

            for b_idx, batch in enumerate(dataloader_train):

                if not CBO.on_batch_begin(b_idx):
                    break

                optimizer.zero_grad()
                recon_x, mean, logvar = self(batch)

                if not CBO.on_batch_end(b_idx):
                    break
                if not CBO.on_loss_begin(b_idx):
                    break

                loss = elbo_loss(recon_x, batch, mean, logvar, beta)
                loss.backward()

                if not CBO.on_loss_end(b_idx):
                    break
                if not CBO.on_step_begin(b_idx):
                    break

                optimizer.step()
                train_loss += loss.item()

                if not CBO.on_step_end(b_idx):
                    break

            train_loss /= len(dataloader_train)
            if _validate:
                valid_loss = _validator._most_recent_loss
            else:
                valid_loss = 0.0
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if epoch % verbose == 0:
                print(f'Epoch: {epoch} | Training Loss: {train_loss} | Valid Loss: {valid_loss}')

            if not CBO.on_epoch_end(epoch):
                break

        CBO.on_train_end()
        return train_losses, valid_losses
    
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
        z = T.sigmoid(self.fc2(z))
        mean = T.sigmoid(self.fc3a(z))
        logvar = T.sigmoid(self.fc3b(z))
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

        z = T.sigmoid(self.fc4(z))
        z = T.sigmoid(self.fc5(z))
        z = T.sigmoid(self.fc6(z))
        # z = self.out(z)
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
