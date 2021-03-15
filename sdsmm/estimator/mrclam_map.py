from collections import OrderedDict as orddict
import numpy as np
import os, torch

from sdsmm.data_readers import read_landmark_file

class MrclamEnvironmentMap:

    def __init__(self, mrclam_landmark_file: str, device: torch.Tensor):
        """
        Creates an environment map using the landmark groundtruth file.
        """
        landmarks = read_landmark_file(mrclam_landmark_file)
        self.__ldmk_dict = orddict()
        for ldmk in landmarks:
            self.__ldmk_dict[int(ldmk["sig"])] = torch.tensor([ldmk["x"], ldmk["y"]],
                dtype=torch.double, device=device)
        #end for
    #end def

    def get_state(self, sig):
        """
        Returns the state with the specified signature `sig`. `sig` must be an integer.
        """
        # Sanity check.
        if torch.is_tensor(sig):
            sig = int(sig.cpu().item())

        assert isinstance(sig, int)
        return None if sig not in self.__ldmk_dict \
            else self.__ldmk_dict.get(sig).clone()
    #end def
#end class
