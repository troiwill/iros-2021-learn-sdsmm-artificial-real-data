import numpy as np
import logging, os, torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Tuple
from sdsmm.utils import wrap_to_pi_

class XIsNonPositive(Exception):
    """Exception raised when the input to a model was invalid."""

    def __init__(self, message='X value less than zero.'):
        super(XIsNonPositive, self).__init__(message)
#end class

class SensorModelBase(ABC):

    def __init__(self, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.__dtype = dtype
        self.__device = device
    #end def
    
    @abstractmethod
    def predict(self, X: torch.Tensor, landmark: torch.Tensor, compute_jacobian: bool) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("`predict` not implemented.")

    @property
    def dtype(self):
        return self.__dtype

    @property
    def device(self):
        return self.__device
#end class

class APrioriMrclamSensorModel(SensorModelBase):

    def __init__(self, robot_id: int, dtype: torch.dtype, device: torch.device):
        super().__init__(dtype=dtype, device=device)

        # Load the sensor noise.
        sensor_noise_file = os.path.join(os.environ['IROS21_SDSMM'],
            'model_params/sensor/apriori/Robot' + str(robot_id) + '_sensor_noise.npy')
        np_sensor_noise = np.load(sensor_noise_file).astype('float64')
        logging.info('Loaded sensor noise from file: ' + sensor_noise_file)

        # Create a PyTorch tensor using sensor noise.
        self.__sensor_noise = torch.empty((1,2,2), dtype=dtype, device=device)
        self.__sensor_noise.copy_(torch.tensor(np_sensor_noise))
        logging.info('Created tensor using sensor noise.')
    #end def

    def predict(self, X: torch.Tensor, landmark: torch.Tensor, compute_jacobian: bool):
        """
        Predicts the expected measurement noise at a particular state `X`.
        """
        robot_x, robot_y, robot_theta = X[:,0,0], X[:,1,0], X[:,2,0]
        ldmk_x, ldmk_y = landmark[0], landmark[1]

        delta_x = ldmk_x - robot_x
        delta_y = ldmk_y - robot_y

        # Compute the expected measurement.
        pred_z = torch.empty([X.size()[0], 2], device=self.device, dtype=self.dtype)
        pred_z[:,0] = torch.sqrt(delta_x.pow(2) + delta_y.pow(2))
        pred_z[:,1] = torch.atan2(delta_y, delta_x) - robot_theta
        wrap_to_pi_(pred_z[:,1])
        pred_z = pred_z.view(-1,2,1)

        # Compute the measurement Jacobian.
        if compute_jacobian:
            MJ = torch.empty([X.size()[0], 2, 3], device=self.device, dtype=self.dtype)

            delta_x = torch.max(delta_x, torch.tensor(1e-6, dtype=X.dtype, device=X.device))
            deltay_over_deltax = delta_y / delta_x

            delta_x_sqd = delta_x.pow(2)
            a1_plus_deltay_over_deltax_sqd = 1. + deltay_over_deltax.pow(2)
            row1d = torch.sqrt(delta_x_sqd + delta_y.pow(2))
            dtdx_denom = delta_x_sqd * a1_plus_deltay_over_deltax_sqd
            dtdy_denom = delta_x * a1_plus_deltay_over_deltax_sqd

            MJ[:,0,0] = (robot_x - ldmk_x) / row1d
            MJ[:,0,1] = (robot_y - ldmk_y) / row1d
            MJ[:,0,2] = 0.

            MJ[:,1,0] = delta_y / dtdx_denom
            MJ[:,1,1] = -1. / dtdy_denom
            MJ[:,1,2] = -1.
        else:
            MJ = torch.tensor([], device=self.device, dtype=self.dtype)
        #end if

        return pred_z, self.__sensor_noise.repeat(X.size()[0],1,1), MJ
    #end def
#end class

class MdnMrclamSensorModel(SensorModelBase):

    def __init__(self, torch_nn: nn.Module, dtype: torch.dtype, device: torch.device,
        nn_param_path: str):
        """
        Creates a distance and bearing sensor using a PyTorch neural network ``torch_nn``. The
        parameters for the network are located at path ``nn_param_path``. The network would be
        sent to compute device ``device``.
        """
        super().__init__(dtype=dtype, device=device)

        # Get the PyTorch neural network, send it the proper device, and load parameters.
        self.__torch_nn = torch_nn.to(device)
        self.__torch_nn.load_state_dict(torch.load(nn_param_path, map_location=device), strict=False)
        self.__torch_nn.eval()
    #end def

    def __forward__(self, X: torch.tensor, landmark: torch.tensor, require_X_grad: bool = True) \
        -> Tuple[ torch.tensor, torch.tensor, torch.tensor ]:
        """
        Computes a forward pass with the encapsulated neural network. ``X`` is the robot pose in 
        world coordinates. ``landmark`` is the observed landmark 2D position. ``require_X_grad``
        specifies if we should compute the gradient with respect to ``X``.
        """
        # Create the input for the networks.
        logging.debug('Computing the landmark input for the network.')
        L_in = landmark.view(-1,2).detach().clone().float().to(self.device)
        L_in.requires_grad = False

        L_dim0 = L_in.size()[0]
        X_dim0 = X.size()[0]
        if L_dim0 not in [ 1, X_dim0 ]:
            error_msg = "Cannot use the current landmark array."
            logging.error(error_msg)
            raise Exception(error_msg)

        if L_dim0 == 1:
            L_in = L_in.repeat(X_dim0, 1)

        lx_in, ly_in = L_in[:,0], L_in[:,1]

        logging.debug('Computing the state input for the network.')
        X_in = X.view(-1,3).detach().clone().float().to(self.device)
        X_in.requires_grad = require_X_grad
        rx_in, ry_in, rt_in = X_in[:,0], X_in[:,1], X_in[:,2]

        # Compute the robot-centric coordinates.
        logging.debug('Computing robot-centric coordinates for network input.')
        t_x = lx_in - rx_in
        t_y = ly_in - ry_in
        Rcos = torch.cos(rt_in)
        Rsin = torch.sin(rt_in)
        one = torch.ones(1, dtype=X_in.dtype, device=self.device)
        R_invd = one / ((Rcos * Rcos) - (-Rsin * Rsin))
        rcc_x = R_invd * ((Rcos * t_x) + (Rsin * t_y))
        rcc_y = R_invd * ((-Rsin * t_x) + (Rcos * t_y))

        # Construct the neural network input.
        net_in = torch.cat((rcc_x,rcc_y),axis=0).view(2,-1).t()
        logging.debug('RRC for network input = ' + str(net_in))
        if (net_in[:,0] <= 0).any():
            raise XIsNonPositive()

        logging.debug('Zero-ing the gradients for the network.')
        self.__torch_nn.zero_grad()

        logging.debug('Running network on input.')
        mu, sigma = self.__torch_nn(net_in)

        return mu.reshape(-1,2,1), sigma.reshape(-1,2,2), X_in
    #end def

    def predict(self, X: torch.Tensor, landmark: torch.Tensor, compute_jacobian: bool) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes a forward pass with the PyTorch model. ``X`` is the robot pose in world
        coordinates, and ``landmark`` is the 2D position of an observed landmark. Returns
        the expected measurement, measurement noise, and measurement Jacobian given the 
        robot pose and observed landmark.
        """
        # Compute the forward pass.
        logging.debug('Computing a forward pass with require_X_grad = False.')
        expected_meas, expected_noise, X_params = self.__forward__(
            X=X, landmark=landmark, require_X_grad=compute_jacobian)

        # Create the expected measurement and noise tensors.
        logging.debug('Ensuring the expected measurements and noises have proper shapes.')
        pred_z = expected_meas.detach().clone().to(self.dtype)
        expected_noise = expected_noise.detach().clone().to(self.dtype)

        if compute_jacobian:
            # Compute the Jacobian for the expected measurement.
            logging.debug('Computing the Jacobian of the expected measurement w.r.t. X.')
            emeas_jacobian = torch.empty((X.size()[0], 2, 3), device=self.device,
                dtype=self.dtype)

            expected_meas[:,0].backward(torch.ones_like(expected_meas[:,0]), retain_graph=True)
            emeas_jacobian[:,0].copy_(X_params.grad.detach().clone().reshape((-1,3)))
            X_params.grad = None

            expected_meas[:,1].backward(torch.ones_like(expected_meas[:,1]))
            emeas_jacobian[:,1].copy_(X_params.grad.detach().clone().reshape((-1,3)))

        else:
            emeas_jacobian = torch.tensor([], device=self.device, dtype=self.dtype)
        #end if

        return pred_z, expected_noise, emeas_jacobian
    #end def
#end class
