from sdsmm.utils import wrap_to_pi_

import numpy as np
import torch

EPSILON = 1e-8

def compute_gaussian_prob(z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    """
    Computes the probabilities given the Gaussian parameters ``mu`` and ``sigma``, and
    the measurements ``z``.
    """
    # Sanity checks.
    assert z.size() == mu.size()
    assert z.size()[0] == sigma.size()[0]
    assert z.size()[1] == sigma.size()[1]
    assert sigma.size()[1] == sigma.size()[2]

    # Compute the 
    twopi_inv = torch.tensor(1. / (2. * np.pi), dtype=torch.float32)
    sigma += EPSILON
    sigma_inv = sigma.inverse()

    center_error = (z - mu).reshape(-1,2,1)
    gexp = -0.5 * (center_error.transpose(1,2) @ sigma_inv @ center_error)
    gexp = torch.exp(gexp)
    gnorm = twopi_inv * torch.pow(sigma.det(), -0.5)
    return gnorm * gexp.reshape(-1)
#end def

def log_likelihood(p: torch.Tensor):
    """
    Computes the log-likelihood of a collection of probabilities ``p``.
    """
    return torch.log(p + EPSILON)
#end def

def negative_log_likelihood(p: torch.Tensor):
    """
    Computes the negative log-likelihood of a collection of probabilities ``p``.
    """
    return -1 * torch.log(p + EPSILON)
#end def

def negative_log_likelihood_loss(return_type: str):
    # Sanity checks.
    assert return_type in [ 'mean', 'identity' ]

    def loss(net_out: torch.Tensor, z: torch.Tensor):
        """
        Computes the negative log-likelihood loss given the network output ``net_out`` and
        the measurement ``z``. 
        """
        # Extract the mean and variances.
        mu, sigma = net_out

        # Compute the Gaussian probabilities.
        gauss_probs = compute_gaussian_prob(z=z, mu=mu, sigma=sigma)

        # Compute the negative log-likelihood loss.
        nllh = negative_log_likelihood(p=gauss_probs)
        
        if return_type == 'mean':
            nllh = torch.mean(nllh)

        return nllh
    #end def

    return loss
#end def

def measurement_bias(net_out: torch.Tensor, meas_true: torch.Tensor):
    # Compute the biases.
    mu, _ = net_out
    biases = meas_true - mu

    # Convert the biases to centimeters and degrees.
    biases[:,0] *= 100.
    wrap_to_pi_(biases[:,1])
    biases[:,1] *= (180.0 / np.pi)

    return biases
#end def

def measurement_mse(net_out: torch.Tensor, meas_true: torch.Tensor):
    return torch.mean(torch.square(measurement_bias(net_out=net_out, meas_true=meas_true)), 0)
#end def
