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
                              n_iter=200, hop_length=n_fft, n_fft=n_fft, win_length=n_fft,
                              window='boxcar', center=True, momentum=0)

    test = (test / np.max(test)).astype('float32')

    plt.plot(test)
    plt.show()

    spio.wavfile.write(f'test_{n}.wav', 44100, test)


def directivity(theta, phi, mach):
    theta = hf.limit_angle(theta)
    phi = hf.limit_angle(phi)
    mach_conv = .8 * mach
    num = 2 * ((np.sin(.5 * theta)) ** 2) * np.sin(phi) ** 2
    den = (1 + mach * np.cos(theta)) * (1 + (mach - mach_conv) * np.cos(theta)) ** 2

    return num / den


if __name__ == '__main__':
    # hawc2_sample_testing(0)
    th = np.linspace(0, 1, 500) * 2 * np.pi
    ph = np.linspace(0, 1, 500) * 2 * np.pi

    thg, phg = np.meshgrid(th, ph)

    m = 300 / hf.c
    d = directivity(thg, phg, m) / directivity(np.pi / 2, np.pi / 2, m)

    x = d * np.cos(thg)
    y = d * np.sin(thg) * np.cos(phg)
    z = d * np.sin(thg) * np.sin(phg)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(x, y, z)
    plt.show()
