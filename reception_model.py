import queue
import numpy as np
import pandas as pd
import compress_pickle as pickle

import helper_functions as hf


"""
========================================================================================================================
===                                                                                                                  ===
=== The reception model for this auralisation tool                                                                   ===
===                                                                                                                  ===
========================================================================================================================
"""
__all__ = ['ReceivedSound', 'Receiver', 'ReceptionModel']


hrtf = hf.MITHrtf()


class ReceivedSound:
    def __init__(self, t_received: float, spectrum: pd.DataFrame, source_pos: hf.Cartesian, receiver_pos: hf.Cartesian,
                 head_rotation: float) -> None:
        """
        ================================================================================================================
        Class to store incoming sound from SoundRays.
        ================================================================================================================
        :param t_received: time of reception (s) of the SoundRay (as obtained from SoundRay.receive)
        :param spectrum: sound spectrum with amplitude and phase (as obtained from SoundRay.receive)
        :param head_rotation: Receiver.rotation of the Receiver where this sound is received
        """
        self.t = t_received
        self.head_rotation = head_rotation
        self.spectrum = spectrum

        self.relative_source_pos = source_pos.to_hr_spherical(receiver_pos, self.head_rotation)
        self.receiver_pos = receiver_pos

        self.spectrum_binaural = pd.DataFrame(0., index=hrtf.f, columns=['al', 'pl', 'ar', 'pr'])

    def apply_hrtf(self):
        """

        """
        hrtf_l, hrtf_r = hrtf.get_hrtf(self.relative_source_pos[1], self.relative_source_pos[2])

        self.spectrum_binaural['al'] = np.interp(hrtf.f, self.spectrum.index, self.spectrum['a']) * np.abs(hrtf_l)
        self.spectrum_binaural['pl'] = np.interp(hrtf.f, self.spectrum.index, self.spectrum['p']) + np.angle(hrtf_l)

        self.spectrum_binaural['ar'] = np.interp(hrtf.f, self.spectrum.index, self.spectrum['a']) * np.abs(hrtf_r)
        self.spectrum_binaural['pr'] = np.interp(hrtf.f, self.spectrum.index, self.spectrum['p']) + np.angle(hrtf_r)


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
        self.cartesian = hf.Cartesian(x, y, z)

        self.index = parse_dict['index']
        self.rotation = hf.limit_angle(np.radians(parse_dict['rotation']))

        self.received = dict[float: list]()

        self.spectrogram_left = pd.DataFrame()
        self.spectrogram_right = pd.DataFrame()

        self.mode = ''

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

    def sum_spectra(self, mode: str) -> None:
        """
        Sum all the sound spectra at the same time step.
        """
        # Sort the received dictionary by its keys
        self.received = dict[float: list](sorted(self.received.items()))
        self.mode = mode

        # Start a ProgressThread
        p_thread = hf.ProgressThread(len(self.received.keys()), 'Summing received sound')
        p_thread.start()

        if self.mode == 'mono':
            # Initialise the dictionary to receive the sums per time step
            sums = dict[float: pd.DataFrame]()
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
            self.spectrogram_left = pd.concat(sums.values(), axis=1)
            self.spectrogram_right = pd.concat(sums.values(), axis=1)
            # Set the column names to the time steps
            self.spectrogram_left.columns = self.received.keys()
            self.spectrogram_right.columns = self.received.keys()

        elif self.mode == 'stereo':
            # Initialise the dictionaries to receive the sums per time step
            sums_left = dict[float: pd.DataFrame]()
            sums_right = dict[float: pd.DataFrame]()
            # Loop over the timesteps in the received dictionary
            sounds: list[ReceivedSound]
            for t, sounds in self.received.items():
                # Apply the head-related transfer function to the received sounds at this timestep
                [sound.apply_hrtf() for sound in sounds]

                # Merge spectral amplitude and phase for each incoming sound
                spectra_left = [sound.spectrum_binaural['al'] * np.exp(1j * sound.spectrum_binaural['pl'])
                                for sound in sounds]

                spectra_right = [sound.spectrum_binaural['ar'] * np.exp(1j * sound.spectrum_binaural['pr'])
                                 for sound in sounds]

                # Sum these merged spectra and put them into the sums dictionaries
                sums_left[t] = sum(spectra_left)
                sums_right[t] = sum(spectra_right)
                # Update the ProgressThread
                p_thread.update()

            # Put all these sums into a pd.DataFrame
            self.spectrogram_left = pd.concat(sums_left.values(), axis=1)
            self.spectrogram_right = pd.concat(sums_right.values(), axis=1)
            # Set the column names to the time steps
            self.spectrogram_left.columns = self.received.keys()
            self.spectrogram_right.columns = self.received.keys()

        # Stop the ProgressThread
        p_thread.stop()
        del p_thread

    def spectrogram_to_csv(self, path_left: str, path_right: str):
        """

        :param path_left:
        :param path_right:
        :return:
        """
        self.spectrogram_left.to_csv(path_left)
        self.spectrogram_right.to_csv(path_right)

    @staticmethod
    def spectrogram_from_csv(path_left: str, path_right: str):
        """

        :param path_left:
        :param path_right:
        :return:
        """
        spectrogram_left = pd.read_csv(path_left, header=0, index_col=0).applymap(complex)
        spectrogram_left.columns = spectrogram_left.columns.astype(float)
        spectrogram_left.index = spectrogram_left.index.astype(float)

        spectrogram_right = pd.read_csv(path_right, header=0, index_col=0).applymap(complex)
        spectrogram_right.columns = spectrogram_right.columns.astype(float)
        spectrogram_right.index = spectrogram_right.index.astype(float)

        return spectrogram_left, spectrogram_right

    def pickle(self, path: str):
        """

        :param path:
        :return:
        """
        pickle_file = open(path, 'wb')
        pickle.dump(self, pickle_file)
        pickle_file.close()

    @staticmethod
    def unpickle(path: str):
        pickle_file = open(path, 'rb')
        receiver = pickle.load(pickle_file)
        pickle_file.close()

        return receiver


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
                t_received, spectrum, source_pos = ray.receive(receiver)
                sound = ReceivedSound(t_received, spectrum, source_pos, receiver.cartesian, receiver.rotation)
                # Receive the sound with the Receiver
                receiver.receive(sound)
                # Another one of these fun floating point error correction points :)
                t = round(ray.t[-1], 10)
                # Put the ray in the dictionary for later use (say for plots...)
                if t in self.rays.keys():
                    self.rays[t].append(ray)
                else:
                    self.rays[t] = list['SoundRay']([ray, ])

            # Update the ProgressThread
            p_thread.update()
        # Stop the ProgressThread
        p_thread.stop()
        del p_thread

        # Sum the sound spectra in the receiver
        receiver.sum_spectra(self.params['mode'])
