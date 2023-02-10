import numpy as np
import scipy.io as spio
import librosa
import matplotlib.pyplot as plt

import helper_functions.coordinate_systems as cs
import helper_functions.io as io


class CircleMovingSource:
    def __init__(self, start_point: cs.Cylindrical, omega: float):
        self.starting_point = start_point
        self.omega = omega

    def motion(self, t: np.array, ):
        r_0, psi_0, y_0 = self.starting_point.vec
        origin = self.starting_point.origin
        psi = psi_0 + self.omega * t
        return np.array([cs.Cylindrical(r_0, p, y_0, origin) for p in psi])

    def velocities_cartesian(self, t: np.array, motion_out: bool = False):
        motion = self.motion(t)
        v = np.zeros((motion.size, 3))
        for it, tt in enumerate(t):
            pos = motion[it]
            circle_dot = self.omega * pos.vec[0]
            v[it, 0] = -circle_dot * np.sin(pos.vec[1])
            v[it, 2] = -circle_dot * np.cos(pos.vec[1])

        return (v, motion) if motion_out else v


def wav_and_gla_test_librosa():
    freq, dat = spio.wavfile.read("music_samples/Queen - Bohemian Rhapsody.wav")

    fxx_0 = librosa.stft(dat[:, 0] / np.max(np.abs(dat)))
    fxx_1 = librosa.stft(dat[:, 1] / np.max(np.abs(dat)))

    x_0 = librosa.griffinlim(np.abs(fxx_0))
    x_1 = librosa.griffinlim(np.abs(fxx_1))

    plt.plot(dat[:, 0] / np.max(np.abs(dat)))
    plt.plot(x_0)  # / np.max(np.abs(x_0)))
    plt.show()

    new_dat = np.empty(dat.shape)
    new_dat[:, 0] = x_0
    new_dat[:, 1] = x_1
    new_dat = new_dat / np.max(np.abs(new_dat))

    spio.wavfile.write('music_samples/Test_librosa.wav', freq, new_dat.astype('float32'))


if __name__ == '__main__':
    # sp = cs.Cylindrical(100, 0, 0, cs.Cartesian(0, 0, -150))
    # source = CircleMovingSource(sp, 1, )
    #
    # t_lst = np.linspace(0, 6.3, 631)
    # vel_lst, pos_lst = source.velocities_cartesian(t_lst, motion_out=True)
    #
    # observer = cs.Cartesian(0, -500, 0)

    wav_and_gla_test_librosa()
    # print('Hello World!')
