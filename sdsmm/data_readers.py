import numpy as np
import os

def read_barcode_file(mrclam_barcode_file):
    """
    Reads a subject to barcode data from the specified file.
    """
    # Sanity checks.
    assert os.path.exists(mrclam_barcode_file), \
        "``" + str(mrclam_barcode_file) + "'' does not exist."

    # Read the barcode file from the specified file.
    dtype = [
        ("subject", np.uint8),
        ("barcode", np.uint8)
    ]
    return np.loadtxt(mrclam_barcode_file, dtype=dtype)
#end def

def read_groundtruth_file(mrclam_groundtruth_file):
    """
    Reads the ground truth data from the specified ground truth file.
    """
    # Sanity checks.
    assert os.path.exists(mrclam_groundtruth_file), \
        "``" + str(mrclam_groundtruth_file) + "'' does not exist."
    
    # Read the robot ground truth file.
    dtype = [
        ("time", np.float64),
        ("x", np.float64),
        ("y", np.float64),
        ("hdg", np.float64)
    ]
    return np.loadtxt(mrclam_groundtruth_file, dtype=dtype)
#end def

def read_landmark_file(mrclam_landmark_file, is_original_file=False):
    """
    Reads the landmark data from the specified landmark file.
    """
    # Sanity check.
    assert os.path.exists(mrclam_landmark_file), \
        "``" + str(mrclam_landmark_file) + "'' does not exist."
    
    # Read the landmark data from the specified file.
    dtype = [
        ("sig" if not is_original_file else "subject", np.uint8),
        ("x", np.float64),
        ("y", np.float64),
        ("x_std", np.float64),
        ("y_std", np.float64)
    ]
    return np.loadtxt(mrclam_landmark_file, dtype=dtype)
#end def

def read_measurement_file(mrclam_measurements_file, is_original_file=False):
    """
    Reads the measurement data from the speciied measurement file.
    """
    # Sanity check.
    assert os.path.exists(mrclam_measurements_file), \
        "``" + str(mrclam_measurements_file) + "'' does not exist."
    
    # Read the measurements data from the specified file.
    dtype = [
        ("time", np.float64),
        ("barcode", np.uint8),
        ("range", np.float64),
        ("bearing", np.float64)
    ]
    return np.loadtxt(mrclam_measurements_file, dtype=dtype)
#end def

def read_odometry_file(mrclam_odometry_file):
    """
    Reads the odometry data from the specified odometry file.
    """
    # Sanity check.
    assert os.path.exists(mrclam_odometry_file), \
        "``" + str(mrclam_odometry_file) + "'' does not exist."
    # Read the measurements data from the specified file.
    dtype = [
        ("time", np.float64),
        ("v", np.float64),
        ("w", np.float64)
    ]
    return np.loadtxt(mrclam_odometry_file, dtype=dtype)
#end def

def get_initial_state(mrclam_groundtruth_file):
    """
    Gets the initial state from a specified groundtruth file.
    """
    return read_groundtruth_file(mrclam_groundtruth_file)[0]
#end def
