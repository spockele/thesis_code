import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import pandas as pd
import librosa

import helper_functions as hf


def hawc2_sample_testing(n):
    """

    :param n: blade nr. (0=all, 1-3=individual blade)
    """
    pos, ts_data, spectrograms = hf.read_hawc2_aero_noise('hawc2_out/case10ms_noise_psd_Obs064.out')

    blade_dat: pd.DataFrame = np.sqrt(spectrograms[n]['All'])
    t_lst = np.array(blade_dat.columns)
    delta_t = round(t_lst[1] - t_lst[0], 3)
    f_lst = np.array(blade_dat.index)
    n_fft = int(44100 * delta_t)
    f_desired = np.linspace(0, 44100 // 2, n_fft // 2 + 1, )

    interpolated = np.empty((f_desired.size, t_lst.size))
    for it, ti in enumerate(t_lst):
        orig = blade_dat.loc[:, ti]
        interpolated[:, it] = np.interp(f_desired, f_lst, orig)

    test = librosa.griffinlim(interpolated,
                              n_iter=200, hop_length=n_fft, n_fft=n_fft, window=np.ones(n_fft), center=True)

    test = (test / np.max(test)).astype('float32')

    plt.plot(test)
    plt.show()

    spio.wavfile.write(f'test_{n}.wav', 44100, test)


if __name__ == '__main__':
    hawc2_sample_testing(0)
