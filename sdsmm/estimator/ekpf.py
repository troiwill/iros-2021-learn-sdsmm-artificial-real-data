import logging, os, torch
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Union

from sdsmm.utils import wrap_to_pi_

from sdsmm.estimator.pose import Pose
from sdsmm.estimator.sensor_models import APrioriMrclamSensorModel, MdnMrclamSensorModel
from sdsmm.estimator.system_models import APrioriMrclamSystemModel
from sdsmm.estimator.utils import is_all_finite, multivariate_normal_log_probs, sample_residual

from sdsmm.mdn.model import DistanceBearingMDN

class EKPF:

    def __init__(self, n_particles: int, robot_id: int, dtype: torch.dtype, device: torch.device,
        mdn_params_path: Union[None, str] = None):
        assert isinstance(n_particles, int) and n_particles >= 1
        assert isinstance(robot_id, int)
        assert isinstance(dtype, torch.dtype)
        assert isinstance(device, torch.device)
        assert isinstance(mdn_params_path, str) or mdn_params_path is None

        self.__n_particles = n_particles

        # Create the system model.
        self.__system_model = APrioriMrclamSystemModel(robot_id=robot_id, dtype=dtype,
            device=device)

        # Create the sensor model.
        if mdn_params_path is None:
            self.__sensor_model = APrioriMrclamSensorModel(robot_id=robot_id, dtype=dtype,
                device=device)
        else:
            self.__sensor_model = MdnMrclamSensorModel(torch_nn=DistanceBearingMDN(),
                dtype=dtype, device=device, nn_param_path=mdn_params_path)
        #end if
    #end def

    def propagate(self, action: torch.Tensor, pose_in: Pose) -> Pose:
        """
        Propagates each particle's state and error using the EKF propagation equations.
        """
        FUNC_NAME = "[ epkf.ekf_propagate ]: "

        # Predict the process noise.
        logging.debug(FUNC_NAME + "Computing the propagated states X_bar, " \
            + "the system noise R, and the system Jacobian FJ.")
        X_bar, pred_R, pred_FJ = self.__system_model.predict(X=pose_in.X, action=action)

        # Propagate the state error P.
        logging.debug(FUNC_NAME + "Computing the propagated state error pred_P.")
        pred_P = pred_FJ @ pose_in.P @ pred_FJ.transpose(1,2) + pred_R

        # Copy weights from input pose, store X_bar and pred_P, and return the predicted pose.
        pred_pose = pose_in.create_empty_pose()
        pred_pose.W = pose_in.W
        pred_pose.X = X_bar
        pred_pose.P = pred_P

        return pred_pose
    #end def

    def update(self, sensor_meas: torch.Tensor, landmark: torch.Tensor, pose_in: Pose) -> Pose:
        """
        Uses the Particle filter equations to produce the a posteriori distribution.
        """
        FUNC_NAME = "[ ekpf.update ]: "

        # -------------------------------------------------------------------------------
        # Update particle states using the EKF. Then sample from a proposal distribution.
        # -------------------------------------------------------------------------------
        # Compute the state update using the Kalman Filter.
        logging.debug(FUNC_NAME + 'Using the Kalman Filter to update the state variables.')
        kf_upd_pose = self.__ekf_update__(sensor_meas=sensor_meas, landmark=landmark, pose_in=pose_in)

        # Sample a 'proposed_X' using the Kalman Filter update.
        logging.debug(FUNC_NAME + \
            'Sampling from the proposal distribution (EKF update state) to get `proposed X`.')
        proposed_dist = MultivariateNormal(loc=kf_upd_pose.X.view(-1,3), covariance_matrix=kf_upd_pose.P)
        proposed_X = proposed_dist.sample().view(-1,3,1)
        wrap_to_pi_(proposed_X[:,2])

        # ---------------------------------------------------------------------
        # Compute the importance factors.
        # ---------------------------------------------------------------------
        importance_weights = self.__compute_importance_weights__(propagated_pose=pose_in,
            proposed_X=proposed_X, kf_upd_pose=kf_upd_pose, sensor_meas=sensor_meas,
            landmark=landmark)

        # ---------------------------------------------------------------------
        # Selection step: Multiply/Supress.
        # ---------------------------------------------------------------------
        # Compute the ESS value for resampling.
        apost_pose = pose_in.create_empty_pose()
        
        logging.debug(FUNC_NAME + 'Computing ESS value for resampling.')
        ess = torch.reciprocal(pose_in.W.pow(2).sum())
        if ess < (pose_in.W.numel() / 2):
            # Perform a resampling method to determine which particles 
            # we should multiply or suppress.
            logging.debug(FUNC_NAME + 'Sampling states.')
            selected_p_indices = sample_residual(size=pose_in.W.numel(),
                p=importance_weights.view(-1))
            
            # Obtain the new set of N particles.
            logging.debug(FUNC_NAME + 'Copying the sampled states into `apost_pose`.')
            apost_pose.set_uniform_weights_()
            apost_pose.X = proposed_X[selected_p_indices]
            apost_pose.P = kf_upd_pose.P[selected_p_indices]
        
        # Otherwise, do not resample. Simply keep the "sampled" particles.
        else:
            logging.debug(FUNC_NAME + 'Keeping sampled states and copying into `apost_pose`.')
            apost_pose.W = importance_weights.view(apost_pose.W.size())
            apost_pose.X = proposed_X
            apost_pose.P = kf_upd_pose.P
        #end if

        return apost_pose
    #end def

    def __ekf_update__(self, pose_in: Pose, sensor_meas: torch.Tensor, landmark: torch.Tensor) -> Pose:
        """
        Updates each particle's state and state error using the EKF update equations,
        a sensor measurement model, and the current measurement.
        """
        FUNC_NAME = "[ ekpf.__ekf_update__ ]: "

        # Predict the expected measurement, measurement noise, and measurement Jacobian.
        logging.debug(FUNC_NAME + "Predicting the expected measurement, measurement noise, " \
            + "and measurement Jacobian.")
        pred_z, pred_Q, pred_MJ = self.__sensor_model.predict(X=pose_in.X, landmark=landmark,
            compute_jacobian=True)

        # Compute the measurement innovation.
        logging.debug(FUNC_NAME + "Computing the sensor measurement innovation.")
        innov = sensor_meas - pred_z

        # Compute the Kalman Gain.
        logging.debug(FUNC_NAME + "Computing the Kalman gain.")
        P_mul_pred_MJ_T = pose_in.P @ pred_MJ.transpose(1,2)
        innov_cov = pred_MJ @ P_mul_pred_MJ_T + pred_Q
        kalman_gain = P_mul_pred_MJ_T @ innov_cov.inverse()

        # Compute the updated X and P.
        logging.debug(FUNC_NAME + "Computing the updated state and state errors.")
        upd_pose = pose_in.create_empty_pose()
        upd_pose.W = pose_in.W
        upd_pose.X = pose_in.X + (kalman_gain @ innov)
        upd_pose.P = pose_in.P - (kalman_gain @ pred_MJ @ pose_in.P)

        return upd_pose
    #end def

    def __compute_importance_weights__(self, propagated_pose: Pose, kf_upd_pose: Pose,
        proposed_X: torch.Tensor, sensor_meas: torch.Tensor, landmark: torch.Tensor) -> torch.Tensor:
        """
        Computes an importance weight for each particle using a propagated pose `propagated_pose`,
        the EKF updated pose `kf_upd_pose`, the proposed states `proposed_X`,
        the sensor measurement `sensor_meas`, and the observed landmark `landmark`.
        Returns the importance weights for the particle set.
        """
        FUNC_NAME = "[ ekpf.__compute_importance_weights__ ]: "

        # Compute the measurement probabilities: p(z | state).
        logging.debug(FUNC_NAME + "Predicting the expected measurement and measurement noise.")
        pred_z, pred_Q, _ = self.__sensor_model.predict(X=proposed_X, landmark=landmark,
            compute_jacobian=False)

        logging.debug(FUNC_NAME + 'Computing the measurement probabilities: log[ p(z | state) ].')
        measurement_log_probs = multivariate_normal_log_probs(
            x=sensor_meas.view(-1,2),
            loc=pred_z.view(-1,2),
            cov=pred_Q)

        # Compute the transition probabilities: log(N(x_hat | x)).
        logging.debug(FUNC_NAME + 'Computing the transition probabilities: log[ N(x_hat | x) ].')
        transition_log_probs = multivariate_normal_log_probs(
            x=proposed_X.view(-1,3),
            loc=propagated_pose.X.view(-1,3),
            cov=propagated_pose.P)

        # Compute the proposal probabilities: N(x_hat | kf_upd_X, kf_upd_P).
        logging.debug(FUNC_NAME + \
            'Computing the proposal probabilities: log[  N(x_hat | kf_upd_X, kf_upd_P) ].')
        proposal_log_probs = multivariate_normal_log_probs(
            x=proposed_X.view(-1,3),
            loc=kf_upd_pose.X.view(-1,3),
            cov=kf_upd_pose.P)

        # Performing sanity checks on the log probabilities.
        logging.debug(FUNC_NAME + "Performing sanity checks on log probabilities.")
        if not is_all_finite(measurement_log_probs):
            error_str = FUNC_NAME + 'Measurement probabilities are not finite!'
            logging.error(error_str)
            raise Exception(error_str)

        if not is_all_finite(transition_log_probs):
            error_str = FUNC_NAME + 'Transition probabilities are not finite!'
            logging.error(error_str)
            raise Exception(error_str)

        if not is_all_finite(proposal_log_probs):
            error_str = FUNC_NAME + 'Proposal probabilities are not finite!'
            logging.error(error_str)
            raise Exception(error_str)

        logging.debug(FUNC_NAME + "Sanity checks passed!")

        # Compute the importance weights and then normalize the weights.
        logging.debug(FUNC_NAME + \
            'Computing the importance factor (proposed weights) of each particle.')
        importance_weights = measurement_log_probs.view(-1) + transition_log_probs.view(-1) \
            - proposal_log_probs.view(-1)
        importance_weights = (importance_weights - importance_weights.max()).exp()

        logging.debug(FUNC_NAME + 'Normalizing the importance weights.')
        importance_weights = importance_weights / importance_weights.sum()

        return importance_weights
    #end def
#end class
