import numpy as np
import os
import torch
from torch.utils.data import Dataset
from typing import List

class MrclamLearningDataset(Dataset):

    def __init__(self, X: torch.tensor, y: torch.tensor, batch_size: int,
        shuffle: bool, device: torch.device):
        # Sanity checks.
        assert X.size() == y.size()

        # Extract the 'valid' states.
        valid_indices = (X[:,0] > 0)

        self.X = X[valid_indices].clone().detach().to(device)
        self.y = y[valid_indices].clone().detach().to(device)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.reset()
    #end def

    def reset(self):
        self.idx = 0
        if self.shuffle:
            indices = torch.randperm(self.nobs, dtype=torch.int64, device=self.X.device)
            self.X = self.X[indices]
            self.y = self.y[indices]
        #end if
    #end def

    @property
    def nobs(self):
        return self.X.size()[0]

    @property
    def x_dim(self):
        return self.X.size()[1]

    @property
    def y_dim(self):
        return self.y.size()[1]

    @property
    def nbatches(self):
        return int(np.ceil(self.nobs / self.batch_size))

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        # Get the next batch if possible.
        if self.idx < self.nobs:
            start_idx = self.idx
            end_idx = start_idx + self.batch_size

            # Increase the batch index.
            self.idx = end_idx

            return self.X[start_idx : end_idx], self.y[start_idx : end_idx]
        else:
            self.reset()
            raise StopIteration()
    #end def
#end class

def read_data_with_data_ids(robot_id: int, data_id_list: List[ int ]):
    state_data, meas_data = list(), list()
    for data_id in data_id_list:
        states, measurements = read_real_learning_data(robot_id=robot_id, data_id=data_id)
        state_data += states.tolist()
        meas_data += measurements.tolist()
    #end for
    return torch.tensor(state_data, dtype=torch.float32), \
        torch.tensor(meas_data, dtype=torch.float32)
#end def

def read_artificial_learning_data(set_type: str):
    """
    Reads the artificial data from noise directory ``noise_dir`` and using set type ``set_type``.
    """
    # Form the read path.
    read_dir = os.path.join(os.environ['IROS21_SDSMM'], 'data/mrclam/learn/artificial', 'low_noise')

    # Read the state and measurement data.
    state_data = np.loadtxt(os.path.join(read_dir, set_type + '_states.txt'), dtype=np.float32)
    meas_data = np.loadtxt(os.path.join(read_dir, set_type + '_measurements.txt'), dtype=np.float32)

    return torch.tensor(state_data, dtype=torch.float32), \
        torch.tensor(meas_data, dtype=torch.float32)
#end def

def read_real_learning_data(robot_id: int, data_id: int):
    """
    Reads the ground truth state data and the measurement data.
    """
    datadir = os.path.join(os.environ['IROS21_SDSMM'], 'data/mrclam/learn/real/robot' + str(robot_id))
    assert os.path.exists(datadir), datadir + ' does not exist?!'

    fileprefix = 'Robot{}_Ds{}_'.format(robot_id, data_id)
    state_data = np.loadtxt(os.path.join(datadir, fileprefix + 'states.txt'), dtype=np.float32)
    meas_data = np.loadtxt(os.path.join(datadir, fileprefix + 'measurements.txt'), dtype=np.float32)

    return state_data, meas_data
#end def

def read_bootstraped_learning_data(robot_id: int, bs_size: str, bs_index: str, set_type: str):
    """
    Reads the ground truth state data and the measurement data.
    """
    datadir = os.path.join(os.environ['IROS21_SDSMM'],
            f'data/mrclam/learn/bootstrapped/robot{robot_id}/ss-{bs_size}', 'si-{:03}'.format(bs_index))
    assert os.path.exists(datadir), datadir + ' does not exist?!'

    state_data = np.loadtxt(os.path.join(datadir, set_type + '_states.txt'), dtype=np.float32)
    meas_data = np.loadtxt(os.path.join(datadir, set_type + '_measurements.txt'), dtype=np.float32)

    return torch.tensor(state_data, dtype=torch.float32), \
        torch.tensor(meas_data, dtype=torch.float32)
#end def

def read_synthetic_experiment_learning_data(data_type: str):
    """
    Reads the ground state data and the measurement data for the synthetic experiment.
    """
    datadir = os.path.join(os.environ['IROS21_SDSMM'], 'data/synexp/learn')
    meas_datapath = os.path.join(datadir, f'{data_type}_measurements.txt')
    assert os.path.exists(meas_datapath), f'Path ({meas_datapath}) does not exist!'
    meas_data = np.loadtxt(meas_datapath, dtype=np.float32)

    state_datapath = os.path.join(datadir, f'{data_type}_states.txt')
    assert os.path.exists(state_datapath), f'Path ({state_datapath}) does not exist!'
    state_data = np.loadtxt(state_datapath, dtype=np.float32)

    return torch.from_numpy(state_data), torch.from_numpy(meas_data)
#end def
