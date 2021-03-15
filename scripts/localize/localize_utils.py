import h5py, logging, torch
import numpy as np
from typing import Dict

def save_data(data_to_save: torch.Tensor, attrs: Dict, h5datafp: str):
    """
    Saves the log data to a specified location on the disk.
    """
    logging.info('Organizing filter results to save to disk.')

    # Iteratively collect the data from the simulation results.
    h5dict = dict()
    for dkey in [ 'time', 'id', 'time_elapsed', 'pred', 'upd', 'X_mean' ]:
        h5dict[dkey] = dict(step=list(), value=list())
    
    for col_i, dkey in zip([ 0, 1, 2, slice(4,7) ], [ 'time', 'id', 'time_elapsed', 'X_mean' ]):
        h5dict[dkey]['step'] = list(range(data_to_save.shape[0]))
        value = data_to_save[:, col_i]
        h5dict[dkey]['value'] = value.tolist() if dkey != 'X_mean' else value.reshape((-1,3,1)).tolist()
    
    for step_i, is_upd in enumerate(data_to_save[:,3]):
        if np.isclose(is_upd, 1):
            h5dict['upd']['step'].append(step_i)
            h5dict['upd']['value'].append(True)
        
        else:
            h5dict['pred']['step'].append(step_i)
            h5dict['pred']['value'].append(False)
    #end for

    # Transfer the data to a the h5 file.
    logging.info('Transferring filter results to H5 file.')
    h5data = h5py.File(h5datafp, mode="w")
    for _, (dkey, datum_dict) in enumerate(h5dict.items()):
        # Create the new dataset within the h5 file.
        h5data.create_dataset(dkey + "/step", 
            data=np.array(datum_dict["step"], dtype="u4"))
        h5data.create_dataset(dkey + "/value",
            data=np.array(datum_dict["value"]))
    #end for

    # Write the attributes to the h5 file.
    logging.info('Writing the attributes to the H5 file.')
    for akey, avalue in attrs.items():
        assert isinstance(akey, str)
        if isinstance(avalue, str):
            nvalue = np.string_(avalue.strip())
        else:
            nvalue = np.array(avalue)
        h5data.attrs.create(akey, nvalue)
    #end for
    h5data.close()
    logging.info('Complete!')
#end def
