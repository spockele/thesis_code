import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import pandas as pd
import librosa

import helper_functions as hf


if __name__ == '__main__':
    pos, ts_data, spectrograms = hf.read_hawc2_aero_noise('hawc2_out/case10ms_noise_psd_Obs064.out')

    all_blades: pd.DataFrame = np.sqrt(spectrograms[0]['All'])
    t_lst = np.array(all_blades.columns)
    delta_t = round(t_lst[1] - t_lst[0], 3)
    f_lst = np.array(all_blades.index)
    n_fft = int(44100 * delta_t)
    f_desired = np.linspace(0, 44100 // 2, n_fft // 2 + 1, )

    interpolated = np.empty((f_desired.size, t_lst.size))
    for it, ti in enumerate(t_lst):
        orig = all_blades.loc[:, ti]
        interpolated[:, it] = np.interp(f_desired, f_lst, orig)

    test = librosa.griffinlim(interpolated,
                              n_iter=200, hop_length=n_fft, n_fft=n_fft, window=np.ones(n_fft), center=False)

    test = (test / np.max(test)).astype('float32')

    plt.plot(test)
    plt.show()

    spio.wavfile.write('test.wav', 44100, test)
