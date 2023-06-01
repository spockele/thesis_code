import math
import numpy as np
import scipy.fft as spfft
import scipy.signal as spsig
import scipy.io as spio

import helper_functions as hf
import reception_model as rm


"""
========================================================================================================================
===                                                                                                                  ===
=== The reconstruction model for this auralisation tool                                                              ===
===                                                                                                                  ===
========================================================================================================================
"""
__all__ = ['random', 'ReconstructionModel', ]


def random(receiver: rm.Receiver, aur_conditions_dict: dict, aur_reconstruction_dict: dict, wav_path: str):
    """

    :param receiver:
    :param aur_conditions_dict:
    :param aur_reconstruction_dict:
    :param wav_path:
    :return:
    """
    f_s_desired = aur_reconstruction_dict['f_s_desired']
    n_base = int(f_s_desired * aur_conditions_dict['delta_t'])
    n_fft = n_base * aur_reconstruction_dict['zeropad']
    n_perseg = n_base * aur_reconstruction_dict['overlap']

    window = spsig.windows.tukey(n_perseg, .75, )

    x_fft = 1j * np.zeros((receiver.spectrogram.columns.size, n_fft // 2))
    f = spfft.fftfreq(n_fft, 1 / f_s_desired)[:n_fft // 2]
    f_spectrogram = receiver.spectrogram.index.to_numpy().flatten()

    for ti, t in enumerate(receiver.spectrogram.columns):
        x_spectrogram = receiver.spectrogram.loc[:, t].to_numpy()
        x_fft[ti] = np.sqrt(np.interp(f, f_spectrogram, x_spectrogram) * np.sum(window)) * np.exp(
            1j * np.random.default_rng().uniform(0, 2 * np.pi, f.size))

    t, x = spsig.istft(x_fft.T, f_s_desired, nfft=n_fft, nperseg=n_perseg, noverlap=n_perseg - n_base,
                       window=window,)

    # Longer sound files :)
    rotation_time = 60 / aur_conditions_dict['rotor_rpm']  # 60 (s/min) / RPM (1 / min)
    n_tiles = int(math.ceil(aur_reconstruction_dict['t_audio'] / rotation_time))
    n_required = int(math.ceil(aur_reconstruction_dict['t_audio'] * f_s_desired))

    x_rotation = x[t >= t[-1] - rotation_time]
    x_long = np.tile(x_rotation, n_tiles)

    norm = aur_reconstruction_dict['wav_norm']
    wav_dat = (np.clip(x_long[:n_required] / norm, -1, 1) * 32767).astype(np.int16)
    spio.wavfile.write(wav_path, f_s_desired, wav_dat)


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

        self._reconstruct = globals()[self.params['model']]

    def run(self, receiver: rm.Receiver, wav_path: str):
        self._reconstruct(receiver, self.conditions_dict, self.params, wav_path)
