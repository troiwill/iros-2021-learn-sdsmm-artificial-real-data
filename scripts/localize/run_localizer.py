import argparse, gc, logging, os, re, shutil, sys, time, torch
from collections import OrderedDict
from datetime import datetime
from torch.distributions.multivariate_normal import MultivariateNormal

from sdsmm.data_readers import get_initial_state

from sdsmm.estimator.ekpf import EKPF
from sdsmm.estimator.ekpf_loop import LocalizerLoop
from sdsmm.estimator.mrclam_event import ActionSensorEventIterator
from sdsmm.estimator.mrclam_map import MrclamEnvironmentMap
from sdsmm.estimator.pose import Pose

from localize_utils import save_data

# Names of the different measurement models.
SMM_APRIORI = "apriori"
SMM_RDO = "rdo"
SMM_FLRD = "flrd"
SMM_LRDO = "lrdo"
SMM_NAME_LIST = [ SMM_APRIORI, SMM_RDO, SMM_FLRD, SMM_LRDO ]

# Names of log levels.
LL_INFO = 'info'
LL_DEBUG = 'debug'
LL_OPTIONS = OrderedDict({LL_INFO: logging.INFO, LL_DEBUG: logging.DEBUG })

# Get the environment path.
sdsmm_repo_root = os.environ["IROS21_SDSMM"]

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("data_id", type=int, choices=[1, 4, 7],
    help="The MR.CLAM dataset identifier.")
parser.add_argument("robot_id", type=int, choices=range(1,6),
    help="The robot identifier.")
parser.add_argument("meas_model", help="Selects the type of sensor model to use.",
    choices=SMM_NAME_LIST)

parser.add_argument("--savedir", help="Root save directory for all localization experiments.",
    default=os.path.join(sdsmm_repo_root, 'exps/localize'))

parser.add_argument('--bstrp_size', choices=['all', '10k', '5k', '2.5k', '0.5k', '0.1k'],
    help='The bootstrap size of the data if using an FLRD or LRDO model.')
parser.add_argument('--bstrp_index', default=-1, type=int,
    help='The bootstrap index/ID (1-100) if using an FLRD or LRDO model. ')

parser.add_argument("-n", "--n_particles", default=500, type=int,
    help="Specifies the number of particles to use in the Extended Kalman Particle Filter.")

parser.add_argument("--loglevel", default=LL_INFO, choices=LL_OPTIONS.keys(),
    help="The log level for this localization trial.")

args = parser.parse_args()

DTYPE = torch.double
DEVICE = torch.device('cpu')

# Create the save directory.
SAVE_DIR = os.path.join(args.savedir, args.meas_model)
if args.bstrp_size is not None and args.bstrp_index is not None:
    SAVE_DIR = os.path.join(SAVE_DIR,
        'ss-' + args.bstrp_size,
        'si-{:05}'.format(args.bstrp_index))

SAVE_DIR = os.path.join(SAVE_DIR, 'robot-' + str(args.robot_id))
SAVE_DIR = os.path.join(SAVE_DIR, 'ekpf/data-' + str(args.data_id))

if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)

# Set up the logger.
LOG_FILE = os.path.join(SAVE_DIR, 'log.txt')
logging.basicConfig(format='%(asctime)-15s %(message)s',
    level=LL_OPTIONS[args.loglevel],
    filename=LOG_FILE)
logging.info('Localize log file: ' + LOG_FILE)

# Get the starting time.
SCRIPT_START_TIME = time.time()

# Enable the garbage collector.
logging.info('Enabling the Python garbage collector.')
gc.enable()

# Define the data path.
rootdatapath = os.path.join(sdsmm_repo_root,
    "data/mrclam/clean/courses_data_assoc",
    "MRCLAM_Dataset{}".format(args.data_id))
logging.info('Data Root Path: ' + rootdatapath)

# Create the Action-Measurement Event Iterator.
logging.info('Creating the filter event iterator.')
filter_event_iter = ActionSensorEventIterator(
    odometry_file=os.path.join(rootdatapath, "Robot{}_Odometry.dat".format(args.robot_id)),
    measurement_file=os.path.join(rootdatapath, 
        "Data_Assoc_Robot{}_Measurement.dat".format(args.robot_id)),
    device=DEVICE)

# Create the landmark map.
logging.info('Creating the MR.CLAM landmark map.')
mrclam_landmark_file = os.path.join(rootdatapath, "Landmark_Groundtruth.dat")
environ_map = MrclamEnvironmentMap(
    mrclam_landmark_file=mrclam_landmark_file,
    device=DEVICE)

# Compute the initial state and the initial state error covariance.
logging.info('Computing the initial state and the inital state error covariance.')
robot_groundtruth_file = os.path.join(rootdatapath,
    "Robot{}_Groundtruth.dat".format(args.robot_id))
state0 = get_initial_state(robot_groundtruth_file)
time0 = state0["time"]

pose = Pose(n_x=args.n_particles, dtype=DTYPE, device=DEVICE)
pose.set_uniform_weights_()

# Set up the EKPF state estimator.
print("Using the Extended Kalman Filter + Particle Filter.")
pose.P = torch.diag(torch.tensor(data=[0.001, 0.001, 0.01],
    dtype=DTYPE, device=DEVICE)).view(1,3,3).repeat(args.n_particles,1,1)

X_init = torch.tensor(data=[state0["x"], state0["y"], state0["hdg"]],
    dtype=DTYPE).view(1,3,1).repeat(args.n_particles,1,1)
X_init = MultivariateNormal(X_init.view(-1,3), pose.P.view(-1,3,3))
pose.X = X_init.sample().view(args.n_particles, 3, 1)

print('Creating EKPF.')
logging.debug('Creating EKPF.')
mdn_params_path = None
if args.meas_model != SMM_APRIORI:
    mdn_params_path = os.path.join(os.environ['IROS21_SDSMM'], 'exps/models', args.meas_model)

    if args.bstrp_size is not None and args.bstrp_index is not None:
        mdn_params_path = os.path.join(mdn_params_path, 'ss-' + args.bstrp_size,
            'si-{:05}'.format(args.bstrp_index))
    #end if
    mdn_params_path = os.path.join(mdn_params_path, 'robot-' + str(args.robot_id), 'model.pt')

    if os.path.exists(mdn_params_path):
        print('Found the MDN model at path: ' + mdn_params_path)
    else:
        print('Could not find MDN model at path: ' + mdn_params_path)
#end if

ekpf = EKPF(n_particles=args.n_particles, robot_id=args.robot_id, dtype=DTYPE, device=DEVICE,
    mdn_params_path=mdn_params_path)

# Set up and run the filter.
logging.debug('Creating Localizer Loop.')
localizer_loop = LocalizerLoop()

logging.info("************* Running the filter *************")
data_to_save = localizer_loop.run(ekpf=ekpf, pose=pose,
    filter_event_iter=filter_event_iter, environ_map=environ_map)

attr_dict = dict()
attr_dict["n_x"] = args.n_particles
attr_dict["bstp_index"] = args.bstrp_index
attr_dict["bstp_size"] = str(args.bstrp_size) if args.bstrp_size is None else args.bstrp_size
attr_dict["robot_id"] = args.robot_id
attr_dict["data_id"] = args.data_id
attr_dict["meas_model"] = args.meas_model
attr_dict["filter_type"] = 'EKPF'

print('\nSaving results.')
h5datafp = os.path.join(SAVE_DIR, 'results.h5')
save_data(data_to_save=data_to_save, attrs=attr_dict, h5datafp=h5datafp)

logging.info("Localization successfully completed!")
logging.info('Results saved in path: ' + SAVE_DIR)

print('\nLocalization successfully completed!')
print('Results saved in path: ' + SAVE_DIR)

SCRIPT_END_TIME = time.time()
logging.info('Total elapsed time: {}s'.format(SCRIPT_END_TIME - SCRIPT_START_TIME))
print('Total elapsed time: {}s\n\n'.format(SCRIPT_END_TIME - SCRIPT_START_TIME))
