from sdsmm.utils import angle_mean, wrap_to_pi_

from sdsmm.estimator.utils import is_all_finite

import logging, torch
from typing import Union

class Pose:

    def __init__(self, n_x: int, dtype: torch.dtype, device: torch.device):
        """
        Creates a set of particles.
        """
        FUNC_NAME = '[ pose.__init__ ]: '
        
        # Sanity checks.
        logging.debug(FUNC_NAME + 'Performing sanity checks.')
        assert isinstance(n_x, int) and n_x >= 1
        assert isinstance(dtype, torch.dtype)
        assert isinstance(device, torch.device)

        logging.debug(FUNC_NAME + 'Setting internal variables.')
        self.__n_x = n_x
        self.__x_dim = 3

        # Initialize the weights W, particle states X, and particle state error P.
        self.__W = torch.empty([n_x, 1, 1], dtype=dtype, device=device)
        self.__X = torch.empty([n_x, self.__x_dim, 1], dtype=dtype, device=device)
        self.__P = torch.empty([n_x, self.__x_dim, self.__x_dim], dtype=dtype, device=device)
        #end if
    #end def

    @property
    def n_x(self) -> int:
        return self.__n_x
    
    @property
    def x_dim(self) -> int:
        return self.__x_dim

    @property
    def W(self) -> torch.Tensor:
        return self.__W

    @W.setter
    def W(self, weights: Union[float, torch.Tensor]) -> None:
        if isinstance(weights, torch.Tensor):
            self.__W.copy_(weights)
        else:
            self.__W.fill_(weights)
    #end def

    def set_uniform_weights_(self):
        """
        All weights have the same value 1. / n_x, where n_x is the number of particles.
        """
        self.__W.fill_(1. / self.n_x)
    #end def

    @property
    def X(self) -> torch.Tensor:
        return self.__X

    @X.setter
    def X(self, states: Union[float, torch.Tensor]) -> None:
        if isinstance(states, torch.Tensor):
            self.__X.copy_(states)
            wrap_to_pi_(self.__X[:,2])
        else:
            self.__X.fill_(states)
    #end def

    @property
    def P(self) -> torch.Tensor:
        return self.__P

    @P.setter
    def P(self, state_errors: Union[float, torch.Tensor]) -> None:
        if isinstance(state_errors, torch.Tensor):
            self.__P.copy_(state_errors)
        else:
            self.__P.fill_(state_errors)
    #end def

    def create_empty_pose(self) -> 'Pose':
        """
        Creates a new Pose object with the same dimensions as self but with randomly initialized
        weights, states, and state errors.
        """
        return Pose(n_x=self.n_x, dtype=self.__X.dtype, device=self.__X.device)
    #end def

    def create_pose_with_zeros(self) -> 'Pose':
        """
        Creates a new Pose object with the same dimensions as self. Weights, states, and state
        errors are initialized with zeros.
        """
        new_pose = self.create_empty_pose()
        new_pose.W = 0.
        new_pose.X = 0.
        new_pose.P = 0.

        return new_pose
    #end def
    
    def clone(self) -> 'Pose':
        """
        Clones the current pose object and returns an exact copy of the values.
        """
        FUNC_NAME = '[ pose.clone ]: '
        logging.debug(FUNC_NAME + 'Cloning the `Pose`.')
        new_pose = self.create_empty_pose()

        logging.debug(FUNC_NAME + 'Copying Tensor values.')
        new_pose.W = self.__W
        new_pose.X = self.__X
        new_pose.P = self.__P

        return new_pose
    #end def

    def is_finite(self) -> bool:
        """
        Checks if the internal weights, states, and state errors contain non-finite values.
        Returns True if ALL values are finite. Otherwise, False.
        """
        FUNC_NAME = '[ pose.is_finite ]: '
        logging.debug(FUNC_NAME + 'Checking if variables are finite.')
        if not is_all_finite(self.__W):
            logging.error(FUNC_NAME + '`W` contains non-finite values!')
            return False

        if not is_all_finite(self.__X):
            logging.error(FUNC_NAME + '`X` contains non-finite values!')
            return False

        if not is_all_finite(self.__P):
            logging.error(FUNC_NAME + '`P` contains non-finite values!')
            return False

        # All values are finite; return True.
        return True
    #end def

    def mean(self) -> torch.Tensor:
        """
        Computes the mean state X bar using the particle weights and states.
        """
        FUNC_NAME = '[ pose.mean ]: '
        logging.debug(FUNC_NAME + 'Computing the pose mean.')
        xbar = None

        # If there is only one particle, return itself.
        if self.__X.size()[0] == 1:
            xbar = self.__X.detach().clone()

        #Otherwise, compute the mean.
        else:
            xbar = self.__X[0].detach().clone()
            xbar[:2] = (self.__W * self.__X[:,:2]).sum(axis=0)
            xbar[2,0] = angle_mean(self.__X[:,2], self.__W[:,0])

        return xbar
    #end def

    def __str__(self) -> str:
        output =  'Pose data:\n'
        output += '  Pose size = [ {}, {} ]\n\n'.format(self.__X.size()[0], self.__X.size()[1])
        output += '  W: {}\n\n'.format(self.__W.flatten())
        output += '  X:\n{}\n\n'.format(self.__X.reshape([self.n_x, self.x_dim]))
        output += '  P:\n{}\n\n'.format(self.__P.reshape([-1, self.x_dim, self.x_dim]))

        return output
    #end def
#end class
            