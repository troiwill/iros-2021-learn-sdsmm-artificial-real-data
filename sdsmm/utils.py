import logging, math, torch
import numpy as np
from typing import Union

def angle_mean(angles: torch.Tensor, weights: Union[None, torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the weighted mean of a collection of angles. `angles` (in radians) is a vector of
    angles. `weights` is either None or a vector of weights, where weight_i is the corresponding
    weight for angle_i. If `weights` is None, then all angles are weighted equally.
    This algorithm was verified using the following R package:
    https://rdrr.io/rforge/circular/man/weighted.mean.circular.html
    """
    if weights is None:
        weights = 1. / angles.flatten().size()[0]
        wsum = 1.
    else:
        assert weights.size() == angles.size()
        wsum = weights.sum()
    #end if
    ws = weights * angles.sin()
    wc = weights * angles.cos()
    return torch.atan2(ws.sum() / wsum, wc.sum() / wsum)
#end def

def wrap_to_pi_(angle_vector: torch.tensor) -> None:
    """
    Wraps an array of radian angles ``angle_vector`` to the range -pi to +pi. This operation is
    performed inplace.
    """
    angle_vector.add_(np.pi)
    angle_vector.remainder_(2 * np.pi)
    angle_vector.sub_(np.pi)
#end def
