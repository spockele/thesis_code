import queue

import numpy as np
import pandas as pd

import helper_functions as hf
from propagation_model import *


"""
========================================================================================================================
===                                                                                                                  ===
=== The reception model for this auralisation tool                                                                   ===
===                                                                                                                  ===
========================================================================================================================
"""
__all__ = ['ReceivedSound', 'Receiver', 'ReceptionModel']


class ReceivedSound:
    def __init__(self, t_received: float, last_dir: hf.Cartesian, spectrum: pd.DataFrame, head_rotation: float):
        """
        TODO: ReceivedSound.__init__ > write docstring and comments
        :param t_received:
        :param last_dir:
        :param spectrum:
        :param head_rotation:
        """
        self.t = t_received
        self.head_rotation = head_rotation
        self.spectrum = spectrum

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

        sound: ReceivedSound
        for t, sounds in self.received.items():
            spectra = [sound.spectrum['a'] * np.exp(1j * sound.spectrum['p']) for sound in sounds]
            sums[t] = sum(spectra)

            p_thread.update()

        self.spectrogram = pd.concat(sums.values(), axis=1)
        self.spectrogram.columns = self.received.keys()

        p_thread.stop()
        del p_thread


class ReceptionModel:
    def __init__(self, aur_conditions_dict: dict, aur_reception_dict: dict):
        """
        TODO: ReceptionModel.__init__ > write docstring and comments
        :param aur_conditions_dict:
        :param aur_reception_dict:
        """
        self.conditions_dict = aur_conditions_dict
        self.params = aur_reception_dict

        self.rays = {}

    def run(self, receiver: Receiver, in_queue: queue.Queue):
        """
        TODO: ReceptionModel.run > write docstring and comments
        :param receiver:
        :param in_queue:
        """
        p_thread = hf.ProgressThread(in_queue.qsize(), 'Receiving sound rays')
        p_thread.start()

        ray: SoundRay
        for ray in in_queue.queue:
            if ray.received:
                sound = ReceivedSound(*ray.receive(receiver), receiver.rotation)
                receiver.receive(sound)

                t = round(ray.t[-1], 10)
                if t in self.rays.keys():
                    self.rays[t].append(ray)
                else:
                    self.rays[t] = [ray, ]

            p_thread.update()

        p_thread.stop()
        del p_thread
        receiver.sum_spectra()
