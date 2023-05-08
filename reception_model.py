import numpy as np
import pandas as pd

import helper_functions as hf


"""
========================================================================================================================
===                                                                                                                  ===
=== The reception model for this auralisation tool                                                                   ===
===                                                                                                                  ===
========================================================================================================================
"""


class ReceivedSound:
    def __init__(self, last_dir: hf.Cartesian, spectrum: pd.DataFrame, t_received: float, head_rotation: float):
        self.spectrum = spectrum
        self.t = t_received

        coming_from_cartesian = - last_dir
        self.incoming_dir = coming_from_cartesian.to_hr_spherical(hf.Cartesian(0, 0, 0, ), head_rotation)


class Receiver(hf.Cartesian):
    def __init__(self, index: int, x: float, y: float, z: float, ):
        super().__init__(x, y, z)
        self.index = index

        self.received = {}

    def __repr__(self):
        return f'<Receiver {self.index}: {str(self)}>'

    def receive(self, sound: ReceivedSound):
        if sound.t in self.received.keys():
            self.received[sound.t].append(sound)

        else:
            self.received[sound.t] = [sound, ]

    def get(self, t: float):
        if t in self.received.keys():
            return self.received[t]
        else:
            return []

