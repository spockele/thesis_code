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
    def __init__(self, t_received: float, head_rotation: float, last_dir: hf.Cartesian, spectrum: pd.DataFrame):
        if 'gaussian' not in spectrum.columns:
            raise RuntimeError('Sound spectrum DataFrame does not contain the Gaussian reception transfer function.')

        self.t = t_received
        self.head_rotation = head_rotation
        self.spectrum = spectrum

        self.sound_spectrum = spectrum['gaussian'] * spectrum['a'] * np.exp(1j * spectrum['p'])

        # Incoming direction is the inverse of the last travel direction, adjusted for the head rotation
        coming_from_cartesian = - last_dir
        self.incoming_dir = coming_from_cartesian.to_hr_spherical(hf.Cartesian(0, 0, 0, ), head_rotation)


class Receiver(hf.Cartesian):
    def __init__(self, parse_dict: dict, ):
        x, y, z = parse_dict['pos']
        super().__init__(x, y, z)
        self.index = parse_dict['index']
        self.rotation = hf.limit_angle(np.radians(parse_dict['rotation']))

        self.received = {}
        self.spectrogram = pd.DataFrame()

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

    def sum_spectra(self):
        for t, sounds in self.received.items():
            spectra = [sound.sound_spectrum for sound in sounds]
            self.spectrogram[t] = sum(spectra)
