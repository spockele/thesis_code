import math
import time
import numpy as np
import scipy.io as spio
import scipy.fft as spfft
import scipy.signal as spsig

import reception_model as rm


"""
========================================================================================================================
===                                                                                                                  ===
=== The reconstruction model for this auralisation tool                                                              ===
===                                                                                                                  ===
========================================================================================================================
"""
__all__ = ['random', 'ReconstructionModel', ]


def random(receiver: rm.Receiver, aur_conditions_dict: dict, aur_reconstruction_dict: dict, wav_path: str) -> None:
    """
    Signal reconstruction of the receiver spectrograms with random phase
    :param receiver: an rm.Receiver instance
    :param aur_conditions_dict: conditions_dict from the Case class
    :param aur_reconstruction_dict: source_dict from the Case class
    :param wav_path: path to write output WAV file to
    """
    # Obtain parameters from the input dictionaries
    f_s_desired = aur_reconstruction_dict['f_s_desired']
    n_base = int(f_s_desired * aur_conditions_dict['delta_t'])
    overlap = aur_reconstruction_dict['overlap']
    # Define number of points that will be in each signal segment
    n_perseg = n_base * overlap

    # Set up the reconstruction window function
    if overlap == 1:
        # A square (boxcar) window with no overlap
        window = spsig.windows.boxcar(n_perseg)
    else:
        # A hanning window when overlap is present
        window = spsig.windows.hann(n_perseg)

    # Initialise the numpy array for the FFTs of each time segment
    x_stft = 1j * np.zeros((receiver.spectrogram_left.columns.size, n_perseg // 2))
    # Define the FFT frequencies for interpolation
    f = spfft.fftfreq(n_perseg, 1 / f_s_desired)[:n_perseg // 2]
    # Extract the spectrogram frequencies for interpolation
    f_spectrogram = receiver.spectrogram_left.index.to_numpy().flatten()

    # Loop over the time segments of the spectrogram
    for ti, t in enumerate(receiver.spectrogram_left.columns):
        # Extract the spectrogram at this time segment
        x_spectrogram = receiver.spectrogram_left.loc[:, t].to_numpy()
        # Interpolate the FFT to required frequency resolution
        x_stft[ti] = np.interp(f, f_spectrogram, x_spectrogram)
        # Correct FFT for window function
        x_stft[ti] *= np.sqrt(np.sum(window))
        # Add the random phase to the FFT
        x_stft[ti] *= np.exp(1j * np.random.default_rng().uniform(0, 2 * np.pi, f.size))

    # Get the inverse Short-Time Fourier Transform of the created spectrogram
    t, p = spsig.istft(x_stft.T, f_s_desired, nfft=n_perseg, nperseg=n_perseg, noverlap=n_perseg - n_base,
                       window=window,)
    # Correct the output signal for the amount of overlapping.
    # Since integer overlap amounts are used, this is quite simple
    p /= max(1, overlap / 2)

    # Set the rotation time, based on the nominal rotor RPM
    rotation_time = 60 / aur_conditions_dict['rotor_rpm']  # 60 (s/min) / RPM (1 / min)
    # Determine the number of rotations required to obtain the desired file duration
    n_tiles = int(math.ceil(aur_reconstruction_dict['t_audio'] / rotation_time))
    # Determine the number of signal points required to obtain the desire file duration
    n_required = int(math.ceil(aur_reconstruction_dict['t_audio'] * f_s_desired))

    # Select a single rotation from the generated signal. Leave some margin from the signal ramp down due to ray-tracing
    p_rotation = p[np.logical_and(t[-1] - 1.1 * rotation_time <= t, t <= t[-1] - 0.1 * rotation_time)]
    # Extract the normalisation factor
    norm = aur_reconstruction_dict['wav_norm']
    # Normalise, clip and scale the signal for the wav file. Use 16 bit integer formating (see scipy.io.wavfile docs)
    p_norm = (np.clip(p_rotation / norm, -1, 1) * 32767).astype(np.int16)
    # Extend the signal and make it the exact required length
    p_long = np.tile(p_norm, n_tiles)[:n_required]
    # Write the signal to WAV
    spio.wavfile.write(wav_path, f_s_desired, p_long)


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
        # Get the correct reconstruction function
        self._reconstruct = globals()[self.params['model']]

    def run(self, receiver: rm.Receiver, wav_path: str) -> None:
        """
        Run the reconstruction model with the input parameters
        :param receiver: an rm.Receiver instance
        :param wav_path: path to write output WAV file to
        """
        t_0 = time.time()
        # Run the reconstruction function :)
        self._reconstruct(receiver, self.conditions_dict, self.params, wav_path)

        elapsed = round(time.time() - t_0, 2)
        print(f'Reconstructing signal from spectrogram: Done! (Elapsed time: {elapsed} s)')
