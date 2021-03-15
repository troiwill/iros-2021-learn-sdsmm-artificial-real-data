import torch, logging
import numpy as np

from sdsmm.estimator.sensor_models import XIsNonPositive
from sdsmm.estimator.pose import Pose
from progress.bar import ChargingBar

class LocalizeLog(object):

    def __init__(self, nb_timesteps: int, dtype: torch.dtype, device: torch.device):
        self.__logdata = torch.empty((nb_timesteps, 7), dtype=dtype, device=device)

        self.__action_event_id = torch.tensor(0, dtype=dtype, device=device)
        self.__meas_event_id = torch.tensor(1, dtype=dtype, device=device)

        self.__event_i = torch.tensor(0, dtype=torch.int32, device=device)
        self.__event_inc = torch.tensor(1, dtype=torch.int32, device=device)
    #end def

    def record(self, time: torch.Tensor, eid: torch.Tensor, time_elapsed: torch.Tensor,
        pose: torch.Tensor, is_action_event: bool):
        """
        Record the data from the current event.
        """
        event_id = self.__action_event_id if is_action_event else self.__meas_event_id

        self.__logdata[self.__event_i, 0] = time
        self.__logdata[self.__event_i, 1] = eid
        self.__logdata[self.__event_i, 2] = time_elapsed
        self.__logdata[self.__event_i, 3] = event_id
        self.__logdata[self.__event_i, 4:] = pose.view(-1)
    #end def

    def increment_timestep(self):
        self.__event_i += self.__event_inc

    def extract_log(self):
        return self.__logdata[:self.__event_i].cpu().numpy()
#end class

class LocalizerLoop(object):

    def __init__(self):
        pass

    def run(self, ekpf, pose, filter_event_iter, environ_map):
        FUNC_NAME = '[ localizerloop.run ]: '

        # Sanity checks.
        logging.debug(FUNC_NAME + 'Performing sanity checks.')
        assert isinstance(pose, Pose)

        logging.debug(FUNC_NAME + 'Cloning the pose to create `pred_pose`.')
        pred_pose = pose.clone()

        pbar = ChargingBar("Simulation Progress", max=filter_event_iter.nb_timesteps.cpu().item(),
            suffix="%(percent).2f%% - %(elapsed)ds")
        time0 = filter_event_iter.time0

        # Allocate a storage space for the log data.
        logging.info(FUNC_NAME + 'Allocating storage for the log data.')
        localize_log = LocalizeLog(nb_timesteps=filter_event_iter.nb_timesteps.cpu().item(),
            dtype=pose.W.dtype, device=pose.W.device)

        logging.info(FUNC_NAME + "Beginning filter event loop.")
        for event in filter_event_iter:
            if logging.root.level == logging.DEBUG:
                logging.debug(FUNC_NAME + 'Event ID     - {:09d}'.format(event.eid.item()))
                logging.debug(FUNC_NAME + 'Event Time   - {:.4f}'.format(event.time.item()))
                logging.debug(FUNC_NAME + 'Time elapsed - {:.6f}'.format((event.time - time0).item()))

            # Does the event have a command?
            if event.has_action():
                logging.debug('\n')
                logging.debug(FUNC_NAME + 'Received ACTION event.')

                # Extract the action, place it in a tensor.
                logging.debug(FUNC_NAME + 'Extracting the system action.')
                action = event.action

                if action[-1] > 0:
                    logging.debug(FUNC_NAME + 'Propagating the particle states.')
                    pred_pose = ekpf.propagate(action=action, pose_in=pose)

                else:
                    logging.debug(FUNC_NAME + 'Skipping propagation since action DT == 0.')
                #end if

                logging.debug(FUNC_NAME + 'Copying propagated state into current state.')
                pose.X = pred_pose.X
                pose.P = pred_pose.P

                logging.debug('\n')
            #end if

            # Does the event have a measurement?
            if event.has_measurement():
                logging.debug('\n')
                logging.debug(FUNC_NAME + 'Receiving MEASUREMENT event.')

                # Try to perform a measurement update. If a specific failure occurs, continue.
                logging.debug(FUNC_NAME + 'Updating the particles with sensor measurement.')
                try:
                    pose = ekpf.update(
                        sensor_meas=event.sensor_meas.view(1,2,1),
                        landmark=environ_map.get_state(event.meas_sig),
                        pose_in=pred_pose)

                except XIsNonPositive:
                    logging.debug(FUNC_NAME + \
                        'ERROR! Update experienced a non-positive (or invalid) X value. Skipping this measurement.')
                    continue
                #end try

                # Copy the values from the updated pose.
                logging.debug(FUNC_NAME + 'Copying current state into predicted state.')
                pred_pose.W = pose.W
                pred_pose.X = pose.X
                pred_pose.P = pose.P

                logging.debug('\n')
            #end if

            # Sanity check: Are all values (weights, states, and errors) finite? I.e., no NaNs!
            if not pose.is_finite():
                error_msg =  'Experienced a none-finite number.'
                error_msg += '\nEvent time = ' + str(event.time)
                error_msg += '\nEvent ID = ' + str(event.eid)
                error_msg += '\nEvent type = ' + "ACTION" if event.has_action() else "UPDATE"
                error_msg += '\n' + str(pose)

                logging.error(error_msg)
                raise Exception('Experienced a non-finite number in pose arrays.')
            #end if

            # Record the mean of the particle distribution.
            X_mean = pose.mean().view(1,3)
            localize_log.record(time=event.time, eid=event.eid, time_elapsed=event.time - time0,
                pose=X_mean, is_action_event=event.has_action())

            if logging.root.level == logging.DEBUG:
                logging.debug(FUNC_NAME + 'X_mean = ' + str(X_mean.cpu()))

            # Increment the event index.
            localize_log.increment_timestep()
            pbar.next()
            logging.debug('\n\n')
        #end for
        pbar.finish()
        logging.info('')

        # Extract the log data and return it.
        logging.info('Loop complete! Transferring data to CPU.')
        logdata = localize_log.extract_log()
        final_pose_str = 'Final pose:      X_mean = ' + str(logdata[-1, 4:]) + '\n'
        logging.info(final_pose_str)
        print('')
        print(final_pose_str)

        return logdata
    #end def
#end class
