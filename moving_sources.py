import numpy as np
import scipy.io as spio
import pyroomacoustics as pra
import matplotlib.pyplot as plt

import helper_functions.coordinate_systems as cs
import helper_functions.io as io


class CircleMovingSource:
    def __init__(self, start_point: cs.Cylindrical, omega: float):
        self.starting_point = start_point
        self.pos = start_point
        self.omega = omega


def cb(epoch, _, y):
    print(epoch)


def wav_and_gla_test():
    freq, dat = spio.wavfile.read("music_samples/Queen - Bohemian Rhapsody.wav")

    # These are the parameters of the STFT
    fft_size = 512
    hop = fft_size // 4
    win_a = np.hamming(fft_size)
    win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)
    n_iter = 32

    engine = pra.transform.STFT(
        fft_size, hop=hop, analysis_window=win_a, synthesis_window=win_s
    )
    fxx = engine.analysis(dat[:, 0])
    x_mag = np.abs(fxx)

    x_0 = pra.phase.gl.griffin_lim(x_mag, hop, win_a, fft_size, n_iter=n_iter, callback=cb, )

    fxx = engine.analysis(dat[:, 1])
    x_mag = np.abs(fxx)

    x_1 = pra.phase.gl.griffin_lim(x_mag, hop, win_a, fft_size, n_iter=n_iter, callback=cb, )

    new_dat = np.empty(dat.shape)
    new_dat[:, 0] = x_0
    new_dat[:, 1] = x_1

    new_dat = new_dat / np.max(np.abs(new_dat[np.isfinite(new_dat)]))
    print(new_dat)

    spio.wavfile.write('music_samples/Test.wav', freq, new_dat.astype('float32'))


if __name__ == '__main__':
    print(io.read_hawc2_aero_noise('hawc2_out/case10ms_noise_psd_Obs064.out'))

    # print('Hello World!')
