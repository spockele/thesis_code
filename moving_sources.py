import numpy as np
import scipy.signal as spsig
import scipy.io as spio
import helper_functions.coordinate_systems as cs
import pyroomacoustics as pra
import matplotlib.pyplot as plt


class CircleMovingSource:
    def __init__(self, start_point: cs.Cylindrical, omega: float):
        self.starting_point = start_point
        self.pos = start_point
        self.omega = omega


def cb(epoch, _, y):
    print(epoch)


if __name__ == '__main__':
    freq, dat = spio.wavfile.read("music_samples/Queen - Bohemian Rhapsody.wav")

    # These are the parameters of the STFT
    fft_size = 512
    hop = fft_size // 4
    win_a = np.hamming(fft_size)
    win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)
    n_iter = 200

    engine = pra.transform.STFT(
        fft_size, hop=hop, analysis_window=win_a, synthesis_window=win_s
    )
    X = engine.analysis(dat[:, 0])
    X_mag = np.abs(X)

    x_0 = pra.phase.gl.griffin_lim(X_mag, hop, win_a, fft_size, n_iter=32, callback=cb, )

    X = engine.analysis(dat[:, 1])
    X_mag = np.abs(X)

    x_1 = pra.phase.gl.griffin_lim(X_mag, hop, win_a, fft_size, n_iter=32, callback=cb, )

    new_dat = np.empty(dat.shape)
    new_dat[:, 0] = x_0
    new_dat[:, 1] = x_1

    new_dat = new_dat / np.max(np.abs(new_dat[np.isfinite(new_dat)]))
    print(new_dat)

    spio.wavfile.write('music_samples/Test.wav', freq, new_dat.astype('float32'))
