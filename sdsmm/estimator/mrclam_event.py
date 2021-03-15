import numpy as np
import os, torch

from sdsmm.data_readers import read_measurement_file, read_odometry_file

class ActionSensorEvent:

    def __init__(self, time: torch.Tensor, eid: torch.Tensor, action: torch.Tensor = None,
        meas: torch.Tensor = None):
        self.__time = time
        self.__eid = eid

        if action is None:
            self.__action = None
        else:
            self.__action = action.detach().clone()
        
        if meas is None:
            self.__meas = None
        else:
            self.__meas = meas.detach().clone()
    #end def

    @property
    def time(self):
        return self.__time
    
    @property
    def eid(self):
        return self.__eid
    
    @property
    def action(self):
        return self.__action
    
    @property
    def meas(self):
        return self.__meas
    
    @property
    def meas_sig(self):
        # Measurement signature is idx-1, type u1.
        return self.__meas[1].type(torch.uint8)

    @property
    def sensor_meas(self):
        # Measurement value is idx-2,3, type double, shape (2,1).
        return self.__meas[2:].type(torch.double).reshape(2,1)
    
    def has_action(self):
        return self.__action is not None
        
    def has_measurement(self):
        return self.__meas is not None
#end class

class ActionSensorEventIterator:

    def __init__(self, odometry_file: str, measurement_file: str, device: torch.device):
        """
        Creates a Filter Event Iterator with a collection of actions and
        measurements.
        """
        # Load the odometry data.
        self.__actions_i = torch.tensor(0, dtype=torch.long, device=device)
        actions = read_odometry_file(odometry_file)
        delta_times = (actions["time"][1:] - actions["time"][:-1]).tolist() + [0.1]
        self.__actions = torch.zeros([len(delta_times), 4], dtype=torch.double, device=device)
        self.__actions[:,:3] = torch.from_numpy(np.array(actions.tolist(), dtype=np.float64))
        self.__actions[:,3] = torch.from_numpy(np.array(delta_times, dtype=np.float64))

        # Load the measurement data.
        self.__meas_i = torch.tensor(0, dtype=torch.long, device=device)
        measurements = read_measurement_file(measurement_file)
        self.__measurements = torch.zeros([measurements.shape[0], 4], dtype=torch.double, device=device)
        self.__measurements.copy_(torch.from_numpy(np.array(measurements.tolist())))
        
        self.__count = torch.tensor(0, dtype=torch.long, device=device)
        self.__actions_len = torch.tensor(self.__actions.size()[0], dtype=torch.long, device=device)
        self.__meas_len = torch.tensor(self.__measurements.size()[0], dtype=torch.long, device=device)

        self.__inf = torch.tensor(1e20, dtype=torch.double, device=device)
        self.__inc_by_1 = torch.tensor(1, dtype=torch.long, device=device)
    #end def

    @property
    def nb_timesteps(self):
        return self.nb_measurements + self.nb_actions
    #end def

    @property
    def time0(self):
        return torch.min(self.__actions[0,0], self.__measurements[0,0])

    @property
    def nb_actions(self):
        return self.__actions_len
    
    @property
    def nb_measurements(self):
        return self.__meas_len
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next action and/or measurement in the sequence in a tuple.
        Tuple order: `event time`, `new action`, `new measurement`.
        """
        # Sanity check
        has_no_more_actions = self.nb_actions <= self.__actions_i
        has_no_more_meas = self.nb_measurements <= self.__meas_i
        if has_no_more_actions and has_no_more_meas:
            raise StopIteration()

        # Get the next action or measurement in the sequence and their times.
        if has_no_more_actions:
            new_action = None
            action_time = self.__inf.clone()
        else:
            new_action = self.__actions[self.__actions_i]
            action_time = new_action[0]
        
        if has_no_more_meas:
            new_meas = None
            measurement_time = self.__inf.clone()
        else:
            new_meas = self.__measurements[self.__meas_i]
            measurement_time = new_meas[0]

        # Determine which event to return: action or measurement? If both occurs at
        # the same time, only return the action (action has priority over measurements
        # for events).
        as_action = None
        as_meas = None        
        if action_time <= measurement_time:
            as_action = new_action
            self.__actions_i.add_(self.__inc_by_1)
        
        else:
            as_meas = new_meas
            self.__meas_i.add_(self.__inc_by_1)
        #end if

        as_event = ActionSensorEvent(time=torch.min(action_time, measurement_time),
            eid=self.__count, action=as_action, meas=as_meas)
        
        self.__count.add_(self.__inc_by_1)

        return as_event
    #end def

#end class
