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
# TODO: write ReceptionModel class to handle the reception model


class ReceivedSound:
    def __init__(self, t_received: float, head_rotation: float, last_dir: hf.Cartesian, spectrum: pd.DataFrame):
        """
        TODO: ReceivedSound.__init__ > write docstring and comments
        :param t_received:
        :param head_rotation:
        :param last_dir:
        :param spectrum:
        """
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
        """
        TODO: Receiver.__init__ > write docstring and comments
        :param parse_dict:
        """
        x, y, z = parse_dict['pos']
        super().__init__(x, y, z)
        self.index = parse_dict['index']
        self.rotation = hf.limit_angle(np.radians(parse_dict['rotation']))

        self.received = {}
        self.spectrogram = pd.DataFrame()

    def __repr__(self):
        return f'<Receiver {self.index}: {str(self)}>'

    def receive(self, sound: ReceivedSound):
        """
        TODO: Receiver.receive > write docstring and comments.
        :param sound:
        :return:
        """
        sound.t = round(sound.t, 2)
        if sound.t in self.received.keys():
            self.received[sound.t].append(sound)

        else:
            self.received[sound.t] = [sound, ]

    def get(self, t: float):
        """
        TODO: Receiver.get > write docstring and comments
        TODO: Receiver.get > Check if I even need this functions??
        :param t:
        :return:
        """
        if t in self.received.keys():
            return self.received[t]
        else:
            return []

    def sum_spectra(self):
        """
        TODO: Receiver.sum_spectra > write docstring and comments
        :return:
        """
        self.received = dict(sorted(self.received.items()))
        sums = {}

        p_thread = hf.ProgressThread(len(self.received.keys()), 'Summing received sound')
        p_thread.start()

        for t, sounds in self.received.items():
            spectra = [sound.sound_spectrum for sound in sounds]
            sums[t] = sum(spectra)

            p_thread.update()

        self.spectrogram = pd.concat(sums.values(), axis=1)
        self.spectrogram.columns = self.received.keys()

        p_thread.stop()
        del p_thread
