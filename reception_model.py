import queue

import numpy as np
import pandas as pd

import helper_functions as hf
# from propagation_model import SoundRay


"""
========================================================================================================================
===                                                                                                                  ===
=== The reception model for this auralisation tool                                                                   ===
===                                                                                                                  ===
========================================================================================================================
"""
__all__ = ['ReceivedSound', 'Receiver', 'ReceptionModel']


class ReceivedSound:
    def __init__(self, t_received: float, last_dir: hf.Cartesian, spectrum: pd.DataFrame, head_rotation: float) -> None:
        """
        ================================================================================================================
        Class to store incoming sound from SoundRays.
        ================================================================================================================
        :param t_received: time of reception (s) of the SoundRay (as obtained from SoundRay.receive)
        :param last_dir: last item of SoundRay.dir (as obtained from SoundRay.receive)
        :param spectrum: sound spectrum with amplitude and phase (as obtained from SoundRay.receive)
        :param head_rotation: Receiver.rotation of the Receiver where this sound is received
        """
        self.t = t_received
        self.head_rotation = head_rotation
        self.spectrum = spectrum

        # Incoming direction is the inverse of the last travel direction, adjusted for the head rotation
        coming_from = - last_dir
        self.incoming_dir = coming_from.to_hr_spherical(hf.Cartesian(0, 0, 0, ), head_rotation)


class Receiver(hf.Cartesian):
    def __init__(self, parse_dict: dict, ) -> None:
        """
        ================================================================================================================
        Class to with the receiver point and its data. Subclass of Cartesian for ease of use.
        ================================================================================================================
        :param parse_dict: dictionary resulting from CaseLoader._parse_receiver
        """
        x, y, z = parse_dict['pos']
        super().__init__(x, y, z)
        self.index = parse_dict['index']
        self.rotation = hf.limit_angle(np.radians(parse_dict['rotation']))

        self.received = dict[float: list]()
        self.spectrogram = pd.DataFrame()

    def __repr__(self):
        return f'<Receiver {self.index}: {str(self)}>'

    def receive(self, sound: ReceivedSound) -> None:
        """
        Put an instance of ReceivedSound in the received dictionary.
        :param sound: an instance of ReceivedSound
        """
        sound.t = round(sound.t, 10)
        if sound.t in self.received.keys():
            self.received[sound.t].append(sound)

        else:
            self.received[sound.t] = list[ReceivedSound]([sound, ])

    def sum_spectra(self) -> None:
        """
        Sum all the sound spectra at the same time step.
        """
        # Sort the received dictionary by its keys
        self.received = dict[float: list](sorted(self.received.items()))
        # Initialise the dictionary to receive the sums per time step
        sums = dict[float: pd.DataFrame]()

        # Start a ProgressThread
        p_thread = hf.ProgressThread(len(self.received.keys()), 'Summing received sound')
        p_thread.start()

        # Loop over the timesteps in the received dictionary
        sounds: list[ReceivedSound]
        for t, sounds in self.received.items():
            # Merge spectral amplitude and phase for each incoming sound
            spectra = [sound.spectrum['a'] * np.exp(1j * sound.spectrum['p']) for sound in sounds]
            # Sum these merged spectra and put them into the sums dictionary
            sums[t] = sum(spectra)
            # Update the ProgressThread
            p_thread.update()

        # Put all these sums into a pd.DataFrame
        self.spectrogram = pd.concat(sums.values(), axis=1)
        # Set the column names to the time steps
        self.spectrogram.columns = self.received.keys()

        # Stop the ProgressThread
        p_thread.stop()
        del p_thread


class ReceptionModel:
    def __init__(self, aur_conditions_dict: dict, aur_reception_dict: dict) -> None:
        """
        ================================================================================================================
        Class to manage the Reception model of the Auralisation tool.
        ================================================================================================================
        :param aur_conditions_dict: conditions_dict from the Case class
        :param aur_reception_dict: reception_dict from the Case class
        """
        self.conditions_dict = aur_conditions_dict
        self.params = aur_reception_dict

        # Initialise the dictionary to store the SoundRays
        self.rays = dict[float: list]()

    def run(self, receiver: Receiver, in_queue: queue.Queue) -> None:
        """
        Run the reception model.
        :param receiver: instance of rm.Receiver
        :param in_queue: queue of SoundRays to receive
        """
        # Start a ProgressThread
        p_thread = hf.ProgressThread(in_queue.qsize(), 'Receiving sound rays')
        p_thread.start()

        # Loop over the input queue
        for ray in in_queue.queue:
            # Check if ray is received at the Receiver
            if ray.received:
                # Create the ReceivedSound from the ray at the Receiver
                sound = ReceivedSound(*ray.receive(receiver), receiver.rotation)
                # Receive the sound with the Receiver
                receiver.receive(sound)
                # Another one of these fun floating point error correction points :)
                t = round(ray.t[-1], 10)
                # Put the ray in the dictionary for later use (say for plots...)
                if t in self.rays.keys():
                    self.rays[t].append(ray)
                else:
                    self.rays[t] = list([ray, ])

            # Update the ProgressThread
            p_thread.update()
        # Stop the ProgressThread
        p_thread.stop()
        del p_thread

        # Sum the sound spectra in the receiver
        receiver.sum_spectra()
