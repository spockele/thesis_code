import numpy as np

import helper_functions as hf


"""
========================================================================================================================
===                                                                                                                  ===
=== The reception model for this auralisation tool                                                                   ===
===                                                                                                                  ===
========================================================================================================================
"""


class Receiver(hf.Cartesian):
    def __init__(self, pos: np.array, ):
        super().__init__(*pos)
