import numpy as np
import scipy.io as spio
import scipy.signal as spsig
import librosa
import matplotlib.pyplot as plt

import helper_functions.coordinate_systems as cs
import helper_functions.io as io


c = 343  # [m/s]


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
        for it, pos in enumerate(motion):
            circle_dot = self.omega * pos.vec[0]
            v[it, 0] = -circle_dot * np.sin(pos.vec[1])
            v[it, 2] = -circle_dot * np.cos(pos.vec[1])

        return (v, motion) if motion_out else v

    def observer_time(self, t: np.array, obs: cs.Cartesian, motion_out: bool = False):
        motion = self.motion(t)
        dist = np.empty(motion.shape)
        for ip, pos in enumerate(motion):
            diff = obs - pos
            dist[ip] = np.sqrt(np.sum(diff.vec**2))

        t_obs = t + (1 / c) * dist
        return (t_obs, motion) if motion_out else t_obs

    def emit_sound_to_observer(self, t_sound, f_sound, fxx_sound, obs: cs.Cartesian):
        t_obs = self.observer_time(t_sound, obs)
        freq_mod = self.doppler(t_sound, t_obs)
        v, motion = self.velocities_cartesian(t_sound, motion_out=True)

        delta_t = t_sound[1] - t_sound[0]

        f_obs = np.empty((f_sound.size, t_sound.size))
        fxx_doppler = np.zeros(fxx_sound.shape)
        amp_mod = np.zeros(t_obs.size)

        for di, dm in enumerate(freq_mod):
            f = dm * f_sound
            f_obs[:, di] = f

            amp_mod[di] = self.amplitude_modulation(t_sound[di], t_obs[di], obs, motion[di], v[di])
            fxx = fxx_sound[:, di] * amp_mod[di]
            fxx_doppler[:, di] = np.interp(f_sound, f, fxx)

        t_new = np.arange(0, t_obs[-1] - t_obs[0] + delta_t, delta_t)
        fxx_obs = np.zeros((f_sound.size, t_new.size))
        for fi, f in enumerate(f_sound):
            time_serie_doppler = fxx_doppler[fi, :]
            fxx_obs[fi, :] = np.interp(t_new, t_obs - t_obs[0], time_serie_doppler)

        return fxx_obs

    @staticmethod
    def doppler(em_time: np.array, obs_time: np.array):
        lst = (em_time[1:] - em_time[:-1]) / (obs_time[1:] - obs_time[:-1])
        return np.append(lst, lst[-1])

    @staticmethod
    def amplitude_modulation(t_sound, t_obs, obs, pos, vel):
        term_0 = c * (t_obs - t_sound)
        factor_1 = (-1 / c) * vel
        factor_2 = (obs - pos).vec

        return 1 / (4 * np.pi * (term_0 + factor_1.dot(factor_2)))


def wav_and_gla_test_librosa():
    freq, *_, fxx_0, fxx_1 = io.wav_to_stft("music_samples/Queen - Bohemian Rhapsody.wav")

    n_iter = 32
    x_0 = librosa.griffinlim(np.abs(fxx_0), n_iter=n_iter, )
    x_1 = librosa.griffinlim(np.abs(fxx_1), n_iter=n_iter, )

    new_dat = np.empty((x_0.size, 2))
    new_dat[:, 0] = x_0
    new_dat[:, 1] = x_1
    new_dat = new_dat / np.max(np.abs(new_dat))

    spio.wavfile.write('music_samples/Test_librosa.wav', freq, new_dat.astype('float32'))


def bohemian_rotorsody():
    f_s, br_t, br_f, br_fxx0, br_fxx1 = io.wav_to_stft("music_samples/Queen - Bohemian Rhapsody.wav")

    sp = cs.Cylindrical(100, 0, 0, cs.Cartesian(0, 0, -150))
    source = CircleMovingSource(sp, 1, )

    observer = cs.Cartesian(0, -50, 0)

    br_fxx0_emit = source.emit_sound_to_observer(br_t, br_f, br_fxx0, observer)
    br_fxx1_emit = source.emit_sound_to_observer(br_t, br_f, br_fxx1, observer)

    print('Emission done!')

    n_iter = 5
    x_0 = librosa.griffinlim(br_fxx0_emit, n_iter=n_iter, )
    x_1 = librosa.griffinlim(br_fxx1_emit, n_iter=n_iter, )

    new_dat = librosa.to_mono(np.array([x_0, x_1]))
    new_dat = new_dat / np.max(np.abs(new_dat))

    spio.wavfile.write('music_samples/Bohemian_Rhapsody_Rotor.wav', f_s, new_dat.astype('float32'))


if __name__ == '__main__':
    raise RuntimeError('Thou shalt not run this module on its own!')
