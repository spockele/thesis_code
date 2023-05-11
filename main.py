import os
import numpy as np
import pandas as pd
import scipy.fft as spfft
import scipy.signal as spsig
import scipy.interpolate as spint
import scipy.io as spio
import matplotlib.pyplot as plt

import helper_functions as hf

import case_mgmt as cm
import source_model as sm
import propagation_model as pm
import reception_model as rm


"""
========================================================================================================================
===                                                                                                                  ===
===                                                                                                                  ===
===                                                                                                                  ===
========================================================================================================================
"""


class Project:
    def __init__(self, project_path: str,):
        """
        ================================================================================================================

        ================================================================================================================
        :param project_path:
        """
        # Check if project folder exists.
        if not os.path.isdir(project_path):
            raise NotADirectoryError('Invalid project folder path given.')

        # Create paths for project and for the HAWC2 model
        self.project_path = project_path
        self.h2model_path = os.path.join(project_path, 'H2model')

        # Check that the project contains a HAWC2 model
        if not os.path.isdir(self.h2model_path):
            raise NotADirectoryError('The given project folder does not contain a HAWC2 model in folder "H2model".')

        # Make atmosphere folder if that does not exist yet
        if not os.path.isdir(os.path.join(self.project_path, 'atm')):
            os.mkdir(os.path.join(self.project_path, 'atm'))

        # Obtain cases from the project folder
        self.cases = [cm.Case(self.project_path, aur_file)
                      for aur_file in os.listdir(self.project_path) if aur_file.endswith('.aur')]

        if len(self.cases) <= 0:
            raise FileNotFoundError('No input files found in project folder.')

        print('Project loaded succesfully!')

    def run(self):
        """

        """
        for ci, case in enumerate(self.cases):
            print(f'==================== Simulating case {ci + 1}/{len(self.cases)} ====================')
            # case.run_hawc2()
            case.run()
            print()


def interpolate_octave(x_octave, f_desired, b):
    b_min = -6 * b
    b_max = 4 * b
    f_desired[f_desired < 1e-99] = 1.

    band_number = np.round(np.log(f_desired / 1e3) / np.log(2)).astype(int) - b_min

    interpolated = 1j * np.zeros(f_desired.shape)

    below_band = band_number < 0
    band_number[below_band] = 0

    in_band = np.logical_and(0 <= band_number, band_number <= b_max - b_min)
    interpolated[in_band] = x_octave[band_number[in_band]]

    return interpolated


if __name__ == '__main__':
    # proj_path = os.path.abspath('NTK')
    # proj = Project(proj_path)
    # proj.run()

    # (n_sensor, f_sampling, n_samples), data = hf.read_ntk_data(
    #     '../Stuff/samples/NTK/NTK_Oct2016/nordtank_20150901_122400.tim',
    #     '../Stuff/samples/NTK/NTK_Oct2016/calib.txt')
    # # Sampling period
    # t_sampling = 1 / f_sampling
    # # Set the window sample size for the fft
    # n_fft = 512
    # # Determine the frequency list of the fft with this window size
    # f_fft = spfft.fftfreq(n_fft, t_sampling)[:n_fft // 2]
    # delta_f = f_fft[1] - f_fft[0]
    # # Determine the A-weighting function for the frequency list
    # a_weighting = -145.528 + 98.262 * np.log10(f_fft) - 19.509 * np.log10(f_fft) ** 2 + 0.975 * np.log10(f_fft) ** 3
    #
    # # Determine the start and end indexes of the windows
    # samples = np.arange(0, n_samples, n_fft)
    #
    # # Specify the list of time data points
    # time = np.arange(0, n_samples / f_sampling, 1 / f_sampling)
    # # With these indexes, create the time-frequency grid for the pcolormesh plot of the spectogram
    # time_grd, f_grd = np.meshgrid(time[samples], np.append(f_fft, f_fft[-1] + f_fft[1] - f_fft[0]))
    #
    # shape = (f_fft.size, time[samples].size)
    # x_fft = 1j * np.empty(shape)
    # psl = np.empty(shape)
    # psd = np.empty(shape)
    # pbl = np.empty(shape)
    # pbl_a = np.empty(shape)
    # ospl_f = np.empty(samples.size - 1)
    # oaspl_f = np.empty(samples.size - 1)
    #
    # # Create the hanning window of size n_fft
    # hanning = spsig.get_window('hann', n_fft)
    # # for si in range(n_sensor):
    # si = 3
    # # Cycle through the windows using the indexes in the list "samples"
    # for ii, idx in enumerate(samples[:-1]):
    #     # Determine X_m for the positive frequencies
    #     x_fft[:, ii] = spfft.fft(data[idx:idx + n_fft, si] * hanning)[:n_fft // 2]
    #     # Determine P_m = |X_m|^2 * delta_t / N
    #     psd[:, ii] = (np.abs(x_fft[:, ii]) ** 2) * t_sampling / np.sum(hanning)
    #     # Determine the PSL = 10 log(P_m / p_e0^2) where p_e0 = 2e-5 Pa
    #     psl[:, ii] = 10 * np.log10(2 * psd[:, ii] / 4e-10)
    #     # Determine the PBL using the PSL
    #     pbl[:, ii] = psl[:, ii] + 10 * np.log10(delta_f)
    #     pbl_a[:, ii] = pbl[:, ii] + a_weighting
    #     # Determine the OSPLs with the PBLs of the signal
    #     ospl_f[ii] = 10 * np.log10(np.sum(10 ** (pbl[:, ii] / 10)))
    #     oaspl_f[ii] = 10 * np.log10(np.sum(10 ** (pbl_a[:, ii] / 10)))
    #
    # time_pe = (time[samples][:-1] + time[samples][1:]) / 2
    # # All determined above in the loop for the spectogram, so just plot here
    # plt.figure(10 * si + 1)
    # plt.plot(time_pe, ospl_f, label="Normal")
    # plt.plot(time_pe, oaspl_f, label="A-weighted")
    # plt.xlabel('$t$ (s)')
    # plt.xlim(0, 2.5)
    # plt.ylabel('$OSPL$ (dB)')
    # plt.grid()
    # plt.legend()
    # # plt.savefig(f'/home/josephine/Documents/EWEM/7 - MASTER THESIS/Demo_stuff/NTK_sensor{si + 1}.pdf')
    #
    # # Plot the spectogram
    # plt.figure(10 * si + 2)
    # # Setting the lower and upper boundary value for the SPL
    # print(f'max(PSL) = {round(np.max(psl), 1)} dB / Hz')
    # vmin, vmax = -20, 30
    # # Create the spectogram and its colorbar
    # spectrogram = plt.pcolor(time[samples], f_fft / 1e3, psl, vmin=vmin, vmax=vmax)
    # cbar = plt.colorbar(spectrogram)
    # # Set the tick values on the colorbar with 5 dB/Hz separation
    # cbar.set_ticks(np.arange(vmin, vmax + 5, 5))
    # # Set all the labels
    # cbar.set_label('$PSL$ (dB / Hz)')
    # plt.xlabel('$t$ (s)')
    # plt.xlim(0, 2.5)
    # plt.ylim(0, 18)
    # plt.yscale('log')
    # plt.ylabel('$f$ (kHz)')
    # # plt.ylim(0, 16000)
    # # plt.savefig(f'/home/josephine/Documents/EWEM/7 - MASTER THESIS/Demo_stuff/NTK_sensor{si + 1}.png')
    #
    # # plt.show()
    # plt.close('all')

    spectrogram = pd.read_csv(os.path.abspath('spectrogram_tst.csv'), header=0, index_col=0).applymap(complex)
    spectrogram.columns = spectrogram.columns.astype(float)
    spectrogram.index = spectrogram.index.astype(float)

    n_base = 512
    n_fft = n_base * 8
    n_perseg = n_base * 2
    x = np.empty((spectrogram.columns.size, n_fft))
    x_fft = 1j * np.empty((spectrogram.columns.size, n_fft//2))
    f_s_desired = n_base * 1e2
    window = spsig.get_window('hamming', n_fft)
    f = spfft.fftfreq(n_fft, 1 / f_s_desired)[:n_fft//2]
    f_octave = hf.octave_band_fc(1)
    for ti, t in enumerate(spectrogram.columns):
        x_spectrogram = spectrogram.loc[:, t].to_numpy()
        x_fft[ti] = np.sqrt(interpolate_octave(x_spectrogram, f, 1)) * np.exp(
            1j * np.random.uniform(0, 2 * np.pi, f.size))

    plt.figure(1)
    ctr = plt.pcolor(spectrogram.columns, f, 20 * np.log10(np.abs(x_fft.T) / hf.p_ref), )
    plt.colorbar(ctr)
    t, x = spsig.istft(x_fft.T, f_s_desired, nfft=n_fft, nperseg=n_perseg, noverlap=n_perseg-n_base, window=('tukey', .75))
    plt.figure(3)
    plt.plot(t, x)
    #
    plt.figure(2)
    f_stft, t_stft, x_stft = spsig.stft(x, f_s_desired, nperseg=n_perseg, noverlap=n_perseg-n_base, window=('tukey', .75))
    ctr = plt.pcolor(t_stft, f_stft, 20 * np.log10(np.abs(x_stft) / hf.p_ref), )
    plt.colorbar(ctr)
    # plt.close('all')
    plt.show()


    rotation_time = 60 / 27.1  # 1 / RPM

    x_rotation = x[t >= t[-1] - rotation_time]
    t_rotation = t[t >= t[-1] - rotation_time] - (t[-1] - rotation_time)

    x_rotation[t_rotation > t_rotation[-1] - (t[-1] - rotation_time)] += x[t < t[-1] - rotation_time]
    x_long = np.tile(x_rotation, 10)
    plt.plot(x_long)
    plt.plot(x)
    plt.show()

    wav_dat = (x_long / np.max(np.abs(x_long)) * 32767).astype(np.int16)
    spio.wavfile.write('test.wav', int(f_s_desired), wav_dat)
