import matplotlib.pyplot as plt
import numpy as np
import argparse, h5py, matplotlib, numpy, os, sys, torch
matplotlib.use('Agg')

from numpy.lib.recfunctions import structured_to_unstructured

from sdsmm.utils import wrap_to_pi_
from sdsmm.data_readers import read_groundtruth_file
from interpolations import compute_interpolated_states_from_datatime

class LocalizeResult:

    def __init__(self, resfile: str):
        # Sanity checks.
        assert os.path.exists(resfile), 'Results file ({}) does not exist.'.format(resfile)
        self.__resfile = resfile

        self.__read_results_file_data__()

        # Read the ground truth data.
        gdth_data = self.__read_groundtruth_poses__()

        # Determine the starting time for the predicted poses.
        pred_start_time_idx = 0
        gdth_start_time = gdth_data['time'][0]
        while self.__times[pred_start_time_idx] < gdth_start_time:
            pred_start_time_idx += 1

        self.__times = self.__times[pred_start_time_idx:]
        self.__estd_states = self.__estd_states[pred_start_time_idx:]
        self.__update_indices = self.__update_indices[np.where(self.__update_indices >= pred_start_time_idx)]
        self.__update_indices -= pred_start_time_idx

        # Determine the ending time for the predicted poses.
        pred_end_time_idx = self.__times.shape[0] - 1
        gdth_end_time = gdth_data['time'][-1]
        while self.__times[pred_end_time_idx] > gdth_end_time:
            pred_end_time_idx -= 1

        self.__times = self.__times[:pred_end_time_idx + 1]
        self.__estd_states = self.__estd_states[:pred_end_time_idx + 1]
        self.__update_indices = self.__update_indices[np.where(self.__update_indices <= pred_end_time_idx)]

        # Gather the ground truth states using the predicted pose times.
        self.__gdth_states = compute_interpolated_states_from_datatime(
            reference_times=self.__times,
            robot_times=gdth_data["time"],
            robot_states=structured_to_unstructured(
                gdth_data[["x","y","hdg"]]).reshape((-1,3)))
        
        # Adjust the units for position.
        self.__estd_states[:,:2] *= 100.
        self.__gdth_states[:,:2] *= 100.
    #end def

    # Compute the position errors.
    def __compute_position_errors__(self, y_true, y_pred):
        if len(y_true.shape) == 1:
            return np.abs(y_true - y_pred)
        else:
            return np.sqrt(np.sum(np.square(y_true - y_pred), axis=-1))

    # Compute the position RMSE for this trial.
    def __compute_position_rmse__(self, y_true, y_pred):
        return float(np.sqrt(np.mean(np.square(y_true.flatten() - y_pred.flatten()))))

    # Compute the angle errors.
    def __compute_angle_errors__(self, y_true, y_pred):
        errors = torch.from_numpy(y_true.flatten() - y_pred.flatten())
        wrap_to_pi_(errors)
        return errors

    # Compute the angle RMSE for this trial.
    def __compute_angle_rmse__(self, y_true, y_pred):
        angle_errors = self.__compute_angle_errors__(y_true, y_pred)
        return float(torch.sqrt((torch.square(angle_errors)).mean()))

    @property
    def traj_len(self):
        return self.__times.shape[0]

    @property
    def robot_id(self):
        return self.__robot_id

    @property
    def data_id(self):
        return self.__data_id

    @property
    def meas_model(self):
        return self.__meas_model

    @property
    def bstp_size(self):
        return self.__bsize

    @property
    def bstp_index(self):
        return self.__bindex

    @property
    def filter_type(self):
        return self.__filter_type.lower()

    def times(self, updates_only: bool):
        if updates_only:
            rv = self.__times[self.__update_indices]
        else:
            rv = self.__times
        return rv.copy()
    #end def

    def pred_poses(self, updates_only: bool):
        if updates_only:
            rv = self.__estd_states[self.__update_indices]
        else:
            rv = self.__estd_states
        return rv.copy()
    #end def

    def gdth_states(self, updates_only: bool):
        if updates_only:
            rv = self.__gdth_states[self.__update_indices]
        else:
            rv = self.__gdth_states
        return rv.copy()
    #end def

    def pos_rmse(self, updates_only: bool):
        return self.__compute_position_rmse__(
            y_true=self.gdth_states(updates_only)[:,:2],
            y_pred=self.pred_poses(updates_only)[:,:2])
    #end def

    def x_rmse(self, updates_only: bool):
        return self.__compute_position_rmse__(
            y_true=self.gdth_states(updates_only)[:,0],
            y_pred=self.pred_poses(updates_only)[:,0])
    #end def

    def y_rmse(self, updates_only: bool):
        return self.__compute_position_rmse__(
            y_true=self.gdth_states(updates_only)[:,1],
            y_pred=self.pred_poses(updates_only)[:,1])
    #end def

    def hdg_rmse(self, updates_only: bool):
        return np.degrees(self.__compute_angle_rmse__(
            y_true=self.gdth_states(updates_only)[:,2],
            y_pred=self.pred_poses(updates_only)[:,2]))
    #end def

    def pos_errors(self, updates_only: bool):
        return self.__compute_position_errors__(
            y_true=self.gdth_states(updates_only)[:,:2],
            y_pred=self.pred_poses(updates_only)[:,:2])
    #end def

    def x_errors(self, updates_only: bool):
        return self.__compute_position_errors__(
            y_true=self.gdth_states(updates_only)[:,0],
            y_pred=self.pred_poses(updates_only)[:,0])
    #end def

    def y_errors(self, updates_only: bool):
        return self.__compute_position_errors__(
            y_true=self.gdth_states(updates_only)[:,1],
            y_pred=self.pred_poses(updates_only)[:,1])
    #end def

    def hdg_errors(self, updates_only: bool):
        return np.degrees(self.__compute_angle_errors__(
            y_true=self.gdth_states(updates_only)[:,2],
            y_pred=self.pred_poses(updates_only)[:,2]).numpy())
    #end def

    def __read_results_file_data__(self):
        with h5py.File(self.__resfile, "r") as h5data:
            # Read the results file.
            self.__robot_id = h5data.attrs["robot_id"]
            self.__data_id = h5data.attrs["data_id"]
            self.__meas_model = h5data.attrs["meas_model"].decode("utf-8")
            self.__filter_type = h5data.attrs["filter_type"].decode("utf-8")
            self.__bsize = h5data.attrs["bstp_size"].decode("utf-8")
            self.__bindex = h5data.attrs["bstp_index"]
            
            self.__times = h5data.get("time/value")[...]
            self.__estd_states = h5data.get("X_mean/value")[...]
            self.__estd_states = self.__estd_states.reshape((-1,3))
            self.__update_indices = h5data.get("upd/step")[...]
    #end def

    def __read_groundtruth_poses__(self):
        # Read the ground truth poses.
        gdthfile = os.path.join(os.environ["IROS21_SDSMM"],
            "data/mrclam/clean/courses_data_assoc/MRCLAM_Dataset" + str(self.__data_id),
            "Robot" + str(self.__robot_id) + "_Groundtruth.dat")
        return read_groundtruth_file(gdthfile)
    #end def
#end class

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('resfile', help='Path to the results file.')
    parser.add_argument('--savegraph', default=os.path.join(os.environ['IROS21_SDSMM'],
        'tmp/localize_result'), help='The save directory for graphs.')
    args = parser.parse_args()

    if args.savegraph and not os.path.exists(args.savegraph):
        os.makedirs(args.savegraph)

    # Compute the localization results and then print the values.
    result = LocalizeResult(args.resfile)

    print('\n========================\n')
    print(result.meas_model.upper(), end='')
    if result.bstp_index >= 1:
        print(' (Size-{}, Index-{:05})'.format(result.bstp_size, result.bstp_index), end='')
    print(', Course {}, Robot {}\n'.format(result.data_id, result.robot_id))
    print('RMSE statistics:')
    print('  Position RMSE [cm]: {:.3f}'.format(result.pos_rmse(True)))
    print('  X RMSE [cm]:        {:.3f}'.format(result.x_rmse(True)))
    print('  Y RMSE [cm]:        {:.3f}'.format(result.y_rmse(True)))
    print('  Heading RMSE [deg]: {:.3f}'.format(result.hdg_rmse(True)))

    print('\nError statistics:')
    pos_errors = result.pos_errors(True)
    print('  Positition error [cm]: {:.3f} +/- {:.3f}'.format(
        np.mean(pos_errors), np.std(pos_errors)))
    print('  X error [cm]:          {:.3f} +/- {:.3f}'.format(
        np.mean(result.x_errors(True)), np.std(result.x_errors(True))))
    print('  Y error [cm]:          {:.3f} +/- {:.3f}'.format(
        np.mean(result.y_errors(True)), np.std(result.y_errors(True))))
    print('  Heading error [deg]:   {:.3f} +/- {:.3f}'.format(
        np.mean(result.hdg_errors(True)), np.std(result.hdg_errors(True))))

    for bool_val, plot_color in zip([False, True], ['orange', 'black']):
        pp = result.pred_poses(bool_val).transpose() / 100
        px, py = pp[0], pp[1]
        gp = result.gdth_states(bool_val).transpose() / 100
        gx, gy = gp[0], gp[1]
        if bool_val:
            plt.scatter(px, py, s=5, c=plot_color)
        else:
            plt.scatter(px[0], py[0], s=20, c='red', label='Start')
            plt.scatter(px[-1], py[-1], s=20, c='green', label='End')
            plt.plot(px, py, plot_color, label='Predicted position')
            plt.plot(gx, gy, 'blue', label='Ground truth position')
        #end if
    #end for

    meas_model_str = result.meas_model
    if result.bstp_index > 0:
        meas_model_str += '_s' + result.bstp_size + '_i{:05}'.format(result.bstp_index)
    graphfilename = 'position_graph_crs{}_robot{}_{}_{}.svg'.format(result.data_id,
        result.robot_id, meas_model_str, result.filter_type)
    graphsavefile = os.path.join(args.savegraph, graphfilename)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Position Estimates for {}, {}, Crs {}, Robot {}'.format(
        result.filter_type.replace('_','-').upper(),
        result.meas_model.replace('_full','').upper(),
        result.data_id, result.robot_id))
    plt.legend()
    plt.grid()
    plt.savefig(graphsavefile, dpi=1200)
    plt.close()
    print('\nSaved graph to path: ' + graphsavefile + '\n')
#end if
