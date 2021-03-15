import logging
import numpy as np

def linear_interpolate(ratio: np.array, prev_x: np.array, next_x: np.array) -> np.array:
    """
    Computes a linear interpolated between a previous X and a next X. Equation came from
    https://en.wikipedia.org/wiki/Linear_interpolation#Programming_language_support
    """
    # Sanity checks.
    assert prev_x.shape == next_x.shape
    ratio_copy = None
    if len(prev_x.shape) == 2 and ratio.shape != prev_x.shape:
        ratio_copy = np.reshape(ratio, (-1, 1))
        ratio_copy = np.tile(ratio_copy, [1, prev_x.shape[1]])
    else:
        ratio_copy = np.copy(ratio)
    #end if
    if prev_x.shape != ratio_copy.shape:
        raise Exception('{} --- {}'.format(prev_x.shape, ratio_copy.shape))
    
    # Do the linear interpolation.
    return ((1. - ratio_copy) * prev_x) + (ratio_copy * next_x)
# end def

def compute_linear_interpolation_points_from_time(interp_ref_times: np.array,
    times_to_interp: np.array) -> np.array:
    """
    """
    # Flatten the time arrays.
    interp_ref_times = interp_ref_times.flatten()
    times_to_interp = times_to_interp.flatten()
    times_to_interp_len = times_to_interp.shape[0]
    
    # Sanity checks.
    # e.g. grnd time (time to interp) must be less than meas time (interp ref time)
    if interp_ref_times[0] <= times_to_interp[0]:
        raise Exception("The start times are bad. Start reference time:", interp_ref_times[0],
                        "... Start interpolation time:", times_to_interp[0])

    prev_states_indices = list()
    next_states_indices = list()
    j = 0
    for ref_time in interp_ref_times:
        # Find a time just before the current interpolation reference time.
        while j < times_to_interp_len and times_to_interp[j] < ref_time:
            j+=1
        
        # Loop break check.
        if j >= times_to_interp_len:
            break

        if times_to_interp[j-1] >= times_to_interp[j]:
            raise Exception('Time j[{}] is not less than time j[{}]?'.format(j-1, j))

        # Record the index of the previous info and the next info to a list.
        prev_states_indices.append(j-1)
        next_states_indices.append(j)
    #end for

    # Sanity checks.
    prev_state_len = len(prev_states_indices)
    if prev_state_len != len(next_states_indices):
        raise Exception('The previous state and next state lists are not the same length?')
    if prev_state_len == 0:
        raise Exception('There is nothing in the state lists???')
    #end if

    # Compute the ratio information for linear interpolation.
    ratio_nums = interp_ref_times[:prev_state_len] - times_to_interp[prev_states_indices]
    ratio_denoms = times_to_interp[next_states_indices] - times_to_interp[prev_states_indices]
    ratios = ratio_nums / ratio_denoms

    # Sanity checks.
    nb_ratios = ratios.shape[0]
    if nb_ratios != len(prev_states_indices):
        raise Exception('The computed time ratios are not the same length as the number of previous states?')
    if nb_ratios != len(next_states_indices):
        raise Exception('The computed time ratios are not the same length as the number of next states?')
    
    return ratios, prev_states_indices, next_states_indices
#end def

def compute_interpolated_states_from_datatime(reference_times: np.array, robot_times: np.array,
    robot_states: np.array) -> np.array:
    """
    Computes an array of interpolated robot states from reference times.
    """
    # Sanity checks.
    if robot_times.shape[0] != robot_states.shape[0]:
        raise Exception('The robot ground information is not the same length?')
    if robot_times.shape[0] == 0:
        raise Exception('There is no time data?')
    if reference_times.shape[0] == 0:
        raise Exception('There are no reference times?')
    
    # Compute the linear interpolation information.
    ratios, prev_indices, next_indices = compute_linear_interpolation_points_from_time(
        reference_times, robot_times)
    
    # Compute the interpolated robot states from the linear interpolation information
    #   above and return the interpolated states.
    return linear_interpolate(ratios, robot_states[prev_indices], robot_states[next_indices])
#end def
