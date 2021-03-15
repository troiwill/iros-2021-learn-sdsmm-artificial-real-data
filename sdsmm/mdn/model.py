import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DistanceBearingMDN(nn.Module):

    def __init__(self):
        super(DistanceBearingMDN, self).__init__()

        nb_units = 35
        self.__n_variate = 2

        self.fc0 = nn.Linear(2, nb_units)
        self.fc1 = nn.Linear(nb_units, nb_units)
        self.fc2 = nn.Linear(nb_units, nb_units)
        self.fc3 = nn.Linear(nb_units, nb_units)
        self.fc4 = self._add_linear_output(n_features_in=nb_units)
    #end def

    def forward(self, x: torch.Tensor):
        """
        Computes a forward pass with the input tensor `x`.
        """
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        mu, sigma = self._compute_parameters(linear_output=x)
        return mu, sigma
    #end def

    def load_model(self, modelpath: str, load_to_device: torch.device):
        self.load_state_dict(torch.load(modelpath, map_location=load_to_device), strict=False)
    #end def

    def _add_linear_output(self, n_features_in: int):
        """
        Add a linear output at the end of the neural network. ``n_features_in`` is the number of
        output features from the previous layer.
        """
        # Sanity check.
        n_features_in = int(n_features_in)
        assert n_features_in >= 1

        # Create the slices for mu, sigma.
        self.__mu_slice = torch.tensor(list(range(self.__n_variate)), dtype=torch.int64)

        D_slice_stop = int(self.__mu_slice[-1] + 1 + self.__n_variate)
        self.__D_slice = torch.tensor(list(range(self.__mu_slice[-1] + 1, D_slice_stop)),
            dtype=torch.int64)

        L_slice_stop = int(self.__n_variate * (self.__n_variate + 1) / 2) - self.__n_variate
        self.__L_slice = torch.tensor(list(range(D_slice_stop, D_slice_stop + L_slice_stop)),
            dtype=torch.int64)

        # Compute the number of parameters in the last layer.
        n_param_sigma = self.__n_variate * (self.__n_variate + 1) / 2
        n_features_per_component = int(self.__n_variate + n_param_sigma)

        # Create the final output layer.
        return nn.Linear(n_features_in, n_features_per_component)
    #end def

    def _compute_parameters(self, linear_output: torch.Tensor):
        """
        Given the linear output from the model ``linear_output``, compute the Gaussian parameters
        mean and covariance.
        """
        # Extract the mixing coefficients, the means, and the parameters for sigma.
        mu = linear_output.index_select(1, self.__mu_slice.to(linear_output.device))
        D_params = linear_output.index_select(1, self.__D_slice.to(linear_output.device))
        L_params = linear_output.index_select(1, self.__L_slice.to(linear_output.device))

        # Extract elements of the covariance factorization and compute covariance.
        sigma_fact_matrix_size = (mu.size()[0], self.__n_variate, self.__n_variate)
        D = torch.zeros(sigma_fact_matrix_size, dtype=torch.float32, device=linear_output.device)
        diag_indices = range(self.__n_variate)
        D[:,diag_indices,diag_indices] = torch.exp(D_params)

        L = torch.zeros_like(D)
        L[:,diag_indices,diag_indices] = 1.
        i, j = torch.tril_indices(self.__n_variate, self.__n_variate, -1)
        L[:,i,j] = L_params

        sigma = (L @ D @ L.transpose(1,2))

        return mu, sigma
    #end def
#end class
