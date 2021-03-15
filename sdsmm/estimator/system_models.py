import logging, os, torch
import numpy as np
from typing import Tuple

from sdsmm.utils import wrap_to_pi_

class APrioriMrclamSystemModel(object):

    def __init__(self, robot_id: int, dtype: torch.dtype, device: torch.device):
        """
        Creates an a priori system model.
        """
        # Load the system noise.
        system_noise_file = os.path.join(os.environ['IROS21_SDSMM'],
            'model_params/system/apriori/Robot' + str(robot_id) + '_system_noise.npy')
        np_system_noise = np.load(system_noise_file).astype('float64')
        logging.info('Loaded system noise from file: ' + system_noise_file)

        # Create a PyTorch tensor using system noise.
        self.__system_noise = torch.empty((1,3,3), dtype=dtype, device=device)
        self.__system_noise.copy_(torch.tensor(np_system_noise * 1e3))
        logging.info('Created tensor using system noise.')
    #end def

    def predict(self, X: torch.Tensor, action: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the propagated states X_bar, the system Jacobian pred_FJ,
        and system noise pred_R, given the previous states `X` and control command `action`.
        """
        # Extract the linear velocity, angular velocity, and time step, respectively.
        v, w, dt = action[1], action[2], action[3]

        # Compute the change in linear and angular motion, respectively.
        vdt = v * dt
        wdt = w * dt

        cos_theta_vdt = X[:,2].cos() * vdt
        sin_theta_vdt = X[:,2].sin() * vdt

        # Propagate the particles.
        X_bar = X.detach().clone()

        X_bar[:,0] = X_bar[:,0] + cos_theta_vdt
        X_bar[:,1] = X_bar[:,1] + sin_theta_vdt

        X_bar[:,2] = X_bar[:,2] + wdt
        wrap_to_pi_(X_bar[:,2])

        # Compute the system Jacobian matrix.
        n_x, xdim = X.size()[:2]
        pred_FJ = torch.zeros([n_x, xdim, xdim], dtype=X.dtype, device=X.device)
        pred_FJ[:,0,0] = pred_FJ[:,1,1] = pred_FJ[:,2,2] = 1
        pred_FJ[:,0,2] = -1 * sin_theta_vdt[:,0]
        pred_FJ[:,1,2] = cos_theta_vdt[:,0]

        # Compute the system noise.
        pred_R = self.__system_noise * dt

        return X_bar, pred_R, pred_FJ
    #end def
#end class
