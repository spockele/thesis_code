import math
import time
import numpy as np
import scipy.io as spio
import scipy.fft as spfft
import scipy.signal as spsig
import matplotlib.pyplot as plt

import helper_functions as hf
import reception_model as rm


"""
========================================================================================================================
===                                                                                                                  ===
=== The reconstruction model for this auralisation tool                                                              ===
===                                                                                                                  ===
========================================================================================================================

Copyright (c) 2023 Josephine Pockelé. Licensed under MIT license.

"""
__all__ = ['ReconstructionModel', ]


class ReconstructionModel:
    def __init__(self, aur_conditions_dict: dict, aur_reconstruction_dict: dict, ):
        """
        ================================================================================================================
        Class that manages the reconstruction model.
        ================================================================================================================
        :param aur_conditions_dict: conditions_dict from the Case class
        :param aur_reconstruction_dict: source_dict from the Case class
        """
        self.conditions_dict = aur_conditions_dict
        self.params = aur_reconstruction_dict

    def run(self, receiver: rm.Receiver, wav_path: str) -> None:
        """
        Run the reconstruction model with the input parameters
        :param receiver: an rm.Receiver instance
        :param wav_path: path to write output WAV file to
        """
        t_0 = time.time()
        # Run the reconstruction function :)
        getattr(self, self.params['model'])(receiver, wav_path)

        elapsed = round(time.time() - t_0, 2)
        print(f'Reconstructing signal from spectrogram: Done! (Elapsed time: {elapsed} s)')

    def random(self, receiver: rm.Receiver, wav_path: str) -> None:
        """
        Signal reconstruction of the receiver spectrograms with random phase.
        :param receiver: an rm.Receiver instance
        :param wav_path: path to write output WAV file to
        """
        # Obtain parameters from the input dictionaries
        f_s_desired = self.params['f_s_desired']
        n_base = int(f_s_desired * self.conditions_dict['delta_t']) + 1
        overlap = self.params['overlap']
        # Define number of points that will be in each signal segment
        n_perseg = n_base * overlap

        # Set up the reconstruction window function
        if overlap == 1:
            # A square (boxcar) window with no overlap
            window = spsig.windows.boxcar(n_perseg)
        else:
            # A hanning window when overlap is present
            window = spsig.windows.hann(n_perseg)

        # Determine the number of signal points required to obtain the desire file duration
        n_required = int(math.ceil(self.params['t_audio'] * f_s_desired))
        # Prepare the data array for the final stereo signal
        p_long = np.empty((n_required, 2), dtype=np.int16)

        # Put the spectrograms in a dictionary
        spectrograms = {'left': receiver.spectrogram_left, 'right': receiver.spectrogram_right}
        # Loop over this dictionary
        for si, (side, spectrogram) in enumerate(spectrograms.items()):
            # Initialise the numpy array for the FFTs of each time segment
            x_stft = 1j * np.zeros((n_perseg // 2 + 1, spectrogram.columns.size, ))
            # Define the FFT frequencies for interpolation
            f = np.linspace(0, f_s_desired / 2, n_perseg // 2 + 1)
            # Extract the spectrogram frequencies for interpolation
            f_spectrogram = spectrogram.index.to_numpy().flatten()

            # Loop over the time segments of the spectrogram
            for ti, t in enumerate(spectrogram.columns):
                # Extract the spectrogram at this time segment
                x_spectrogram = spectrogram.loc[:, t].to_numpy()
                # Interpolate the FFT to required frequency resolution
                x_stft[:, ti] = np.interp(f, f_spectrogram, x_spectrogram)
                # Correct FFT for window function
                x_stft[:, ti] *= np.sqrt(np.sum(window))
                # Add the random phase to the FFT
                x_stft[:, ti] *= np.exp(1j * np.random.default_rng().uniform(0, 2 * np.pi, f.size))

            # Get the inverse Short-Time Fourier Transform of the created spectrogram
            t, p = spsig.istft(x_stft, f_s_desired, nperseg=n_perseg, noverlap=n_perseg - n_base, window=window, )
            # Correct the output signal for the amount of overlapping.
            # Since integer overlap amounts are used, this is quite simple
            p /= max(1, overlap / 2)

            p_long[:, si] = self.post_process_signal(t, p, receiver, side)

        # Write the stereo signal to WAV
        spio.wavfile.write(wav_path, f_s_desired, p_long)

    def gla(self, receiver: rm.Receiver, wav_path: str) -> None:
        """
        Signal reconstruction of the receiver spectrograms with the Fast Griffin-Lim algorithm.

         - Perraudin, N., Balazs, P., & Sondergaard, P. L. (2013). A fast Griffin-Lim algorithm.
            2013 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics, 1–4.
            doi: 10.1109/WASPAA.2013.6701851
         - McFee, B., Metsai, A., McVicar, M., Balke, S., Thomé, C., Raffel, C., Zalkow, F., Malek, A., Dana,
            Kyungyun Lee, Nieto, O., Ellis, D., Mason, J., Battenberg, E., Seyfarth, S., Yamamoto, R.,
            Viktorandreevichmorozov, Keunwoo Choi, Moore, J., & Bittner, R. (2023).
            librosa/librosa: 0.9.2 (0.9.2) [Python]. Zenodo. doi: 10.5281/ZENODO.7618817

        :param receiver: an rm.Receiver instance
        :param wav_path: path to write output WAV file to
        """
        # Obtain parameters from the input dictionaries
        f_s_desired = self.params['f_s_desired']
        n_base = int(f_s_desired * self.conditions_dict['delta_t']) + 1
        overlap = self.params['overlap']
        # Define number of points that will be in each signal segment
        n_perseg = n_base * overlap

        # Set up the reconstruction window function
        if overlap == 1:
            # A square (boxcar) window with no overlap
            window = spsig.windows.boxcar(n_perseg)
        else:
            # A hanning window when overlap is present
            window = spsig.windows.hann(n_perseg)

        momentum = .99

        # Determine the number of signal points required to obtain the desire file duration
        n_required = int(math.ceil(self.params['t_audio'] * f_s_desired))
        # Prepare the data array for the final stereo signal
        p_long = np.empty((n_required, 2), dtype=np.int16)

        # Put the spectrograms in a dictionary
        spectrograms = {'left': receiver.spectrogram_left, 'right': receiver.spectrogram_right}
        # Loop over this dictionary
        for si, (side, spectrogram) in enumerate(spectrograms.items()):
            # Initialise the numpy array for the FFTs of each time segment
            x_stft = 1j * np.zeros((n_perseg // 2 + 1, spectrogram.columns.size, ))
            a_stft = np.zeros((n_perseg // 2 + 1, spectrogram.columns.size, ))
            # Define the FFT frequencies for interpolation

            f = np.linspace(0, f_s_desired / 2, n_perseg // 2 + 1)
            # Extract the spectrogram frequencies for interpolation
            f_spectrogram = spectrogram.index.to_numpy().flatten()

            # Loop over the time segments of the spectrogram
            for ti, t in enumerate(spectrogram.columns):
                # Extract the spectrogram at this time segment
                x_spectrogram = spectrogram.loc[:, t].to_numpy()
                # Interpolate the FFT to required frequency resolution
                a_stft[:, ti] = np.abs(np.interp(f, f_spectrogram, x_spectrogram))
                # Correct FFT for window function
                a_stft[:, ti] *= np.sqrt(np.sum(window))
                # Add the random phase to the FFT
                x_stft[:, ti] = a_stft[:, ti] * np.exp(1j * np.random.default_rng().uniform(0, 2 * np.pi, f.size))

            # Fast Griffin-Lim algorithm, adapted from Librosa 0.9.2 by McFee et al. (2023)
            tprev = None
            for _ in range(50):
                # Invert with our current estimate of the phases
                t, inverse = spsig.istft(x_stft, f_s_desired, nperseg=n_perseg, noverlap=n_perseg - n_base, window=window)

                # Rebuild the spectrogram
                _, _, rebuilt = spsig.stft(inverse, f_s_desired, nperseg=n_perseg, noverlap=n_perseg - n_base, window=window)

                # Update our phase estimates
                x_stft[:] = rebuilt
                if tprev is not None:
                    x_stft -= (momentum / (1 + momentum)) * tprev
                x_stft /= np.abs(x_stft)
                x_stft *= a_stft
                # Store
                rebuilt, tprev = tprev, rebuilt

            t, p = spsig.istft(x_stft, f_s_desired, nfft=n_perseg, nperseg=n_perseg, noverlap=n_perseg - n_base, window=window, )

            # Correct the output signal for the amount of overlapping.
            # Since integer overlap amounts are used, this is quite simple
            p /= max(1, overlap / 2)

            p_long[:, si] = self.post_process_signal(t, p, receiver, side)

        # Write the stereo signal to WAV
        spio.wavfile.write(wav_path, f_s_desired, p_long)

    def post_process_signal(self, t, p, receiver, side):
        """
        Extend the short signal from the reconstruction to the desired output duration.
        :param t: Time array, as generated by scipy.signal.istft
        :param p: Signal array, as generated by scipy.signal.istft
        :param receiver: an rm.Receiver instance
        :param side: string indicating the side of the head where the audio came from
        :return: The extended signal array
        """
        # Obtain parameters from the input dictionaries
        f_s_desired = self.params['f_s_desired']
        n_base = int(f_s_desired * self.conditions_dict['delta_t'])

        # Set the rotation time, based on the nominal rotor RPM
        rotation_time = 60 / self.conditions_dict['rotor_rpm']  # 60 (s/min) / RPM (1 / min)
        # Determine the number of rotations required to obtain the desired file duration
        n_tiles = int(math.ceil(self.params['t_audio'] / rotation_time))
        # Determine the number of signal points required to obtain the desire file duration
        n_required = int(math.ceil(self.params['t_audio'] * f_s_desired))

        # Determine the direction between receiver and turbine rotor hub
        receiver_pos = receiver.cartesian
        source_pos = self.conditions_dict['hub_pos']
        relative_source_pos = (source_pos - receiver_pos).to_hr_spherical(receiver_pos, receiver.rotation)
        # Determine the ITD associated with this direction
        side_itd, itd = hf.woodworth_itd(relative_source_pos[1])

        # Select a single rotation from the generated signal. Also apply the ITD if binaural rendering is required
        # Leave some margin from the signal ramp down due to ray-tracing
        if side_itd == side and receiver.mode == 'stereo':
            t_start, t_stop = t[-1] / 2 - .5 * rotation_time - itd, t[-1] / 2 + 0.5 * rotation_time - itd
        else:
            t_start, t_stop = t[-1] / 2 - .5 * rotation_time, t[-1] / 2 + 0.5 * rotation_time
        # Take signal corresponding to 1 rotation
        p_rotation = p[np.logical_and(t_start <= t, t <= t_stop)]

        # Take a little part after that for overlap adding to avoid transition artefacts
        p_overlap = p[t_stop <= t][:n_base]
        # Make a hanning window specifically for overlap-add
        hann = spsig.windows.hann(2 * n_base)
        # Overlap add with half hanning window
        p_rotation[:n_base] = p_rotation[:n_base] * hann[:n_base] + p_overlap * hann[n_base:]

        # Extract the normalisation factor
        norm = self.params['wav_norm']
        # Normalise, clip and scale the signal for the wav file. Use 16 bit integer (see scipy.io.wavfile docs)
        p_norm = (np.clip(p_rotation / norm, -1, 1) * 32767).astype(np.int16)

        # Extend the signal and make it the exact required length
        return np.tile(p_norm, n_tiles)[:n_required]
