import h5py, logging, math, torch
import numpy as np
from typing import Dict

def is_all_finite(a: torch.Tensor) -> bool:
    """
    Checks if `a` contains finite numbers only.
    """
    return torch.isfinite(a).all()
#end def

def multivariate_normal_log_probs(x: torch.Tensor, loc: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """
    Computes the natural log of the probabilities from a normal distribution.
    """
    assert x.size()[1:] == loc.size()[1:] and (x.size()[0] == 1 or x.size()[0] == loc.size()[0])
    m, k = loc.size()
    
    klntwopi = torch.tensor(2 * math.pi, dtype=x.dtype, device=x.device)
    klntwopi = k * klntwopi.log().view(1).repeat([m])

    try:
        maha_dist = cov.view(m,k,k).inverse()
    except:
        error_msg = 'Could not compute the inverse of covariance matrices.'
        logging.error(error_msg)
        raise Exception(error_msg)
    #end try

    diff = (x - loc).view(m, k, 1)
    maha_dist = diff.transpose(1,2).bmm(maha_dist)
    maha_dist = maha_dist.bmm(diff).view(m)

    lndetcov = cov.view(m,k,k).logdet().view(m)

    return -0.5 * (lndetcov + maha_dist + klntwopi)
#end def

def sample_residual(size: int, p: torch.Tensor) -> torch.Tensor:
    """
    Performs residual sampling as described in Van Der Merwe et al.,
    The Unscented Particle Filter Technical Report.
    """
    if p.size() != (p.numel(),):
        raise Exception("`p` must be a 1D array.")

    size = int(size)
    if size != p.numel():
        raise Exception("`size` must be the same as the number of elements in `p`.")

    if torch.any(p < 0.):
        raise Exception("`p` must contain all non-negative values.")

    p_sum = p.sum()
    if not p_sum.isclose(torch.tensor(1., dtype=p.dtype)):
        print("WARNING: The sum of `p` must be close to 1. Sum = " + str(p_sum))

    n_p = p.numel()
    nchild = torch.empty(p.numel(), dtype=torch.long, device=p.device)
    nchild.copy_(torch.floor(n_p * p))
    selected_sample = torch.zeros(size, dtype=torch.long, device=p.device)
    k = 0
    for i in range(n_p):
        start_i, end_i = k, k + nchild[i]
        selected_sample[start_i : end_i].fill_(i)
        k = end_i

    residual = ((p*n_p) - nchild)
    residual /= residual.sum()
    cumulative_sum = residual.cumsum(0)
    randn = torch.randn(n_p-k).cpu()
    searchsorted = np.searchsorted(cumulative_sum.cpu().numpy(), randn.numpy())
    selected_sample[k:n_p] = torch.from_numpy(np.minimum(searchsorted, n_p - 1))
    
    return selected_sample
#end def
