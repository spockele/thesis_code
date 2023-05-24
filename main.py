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
        TODO: Project.__init__ > write docstring
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

        print('Project directory loaded successfully!')

    def run(self):
        """
        TODO Project.run > write docstring and comments
        """
        for ci, case in enumerate(self.cases):
            print(f'==================== Simulating case {ci + 1}/{len(self.cases)} ====================')
            # case.run_hawc2()
            case.run()
            print()


def interpolate_octave(x_octave, f_desired, b):
    """
    TODO: interpolate_octave > check what to do with this function???
    :param x_octave:
    :param f_desired:
    :param b:
    :return:
    """
    b_min = -6 * b
    b_max = 4 * b
    f_desired[f_desired < 1e-99] = 1.

    band_number = np.round(np.log(f_desired / 1e3) / np.log(2)).astype(int) - b_min

    interpolated = 1j * np.zeros(f_desired.shape)

    below_band = band_number < 0
    band_number[below_band] = 0

    in_band = np.logical_and(0 <= band_number, band_number <= b_max - b_min)
    interpolated[in_band] = x_octave[band_number[in_band]]
    interpolated[np.logical_not(in_band)] = np.min(interpolated[in_band]) * 1e-6

    return interpolated


if __name__ == '__main__':
    proj_path = os.path.abspath('NTK')
    proj = Project(proj_path)
    proj.run()

# ----------------------------------------------------------------------------------------------------------------------
# Post-Processing of NTK Data
# ----------------------------------------------------------------------------------------------------------------------
#     (n_sensor, f_sampling, n_samples), data = hf.read_ntk_data(
#         '../Stuff/samples/NTK/NTK_Oct2016/nordtank_20150901_122400.tim',
#         '../Stuff/samples/NTK/NTK_Oct2016/calib.txt')
#     idx = int(f_sampling * 2.35)
#     p = data[:idx, 2]
#
#     # plt.figure(6)
#     # f_stft, t_stft, x_stft = spsig.stft(p, f_sampling)
#     # ctr = plt.pcolor(t_stft, f_stft, 20 * np.log10(np.abs(x_stft) / hf.p_ref), vmin=0, vmax=40)
#     # cbar = plt.colorbar(ctr)
#     # plt.xlabel('t (s)')
#     # plt.ylabel('f (Hz)')
#     # cbar.set_label('PSL (dB / Hz)')
#
#     t_max = 2.35
#     # Sampling period
#     t_sampling = 1 / f_sampling
#     # Set the window sample size for the fft
#     n_fft = 512
#     # Determine the frequency list of the fft with this window size
#     f_fft = spfft.fftfreq(n_fft, t_sampling)[:n_fft // 2]
#     delta_f = f_fft[1] - f_fft[0]
#     # Determine the A-weighting function for the frequency list
#     a_weighting = -145.528 + 98.262 * np.log10(f_fft) - 19.509 * np.log10(f_fft) ** 2 + 0.975 * np.log10(f_fft) ** 3
#
#     # Determine the start and end indexes of the windows
#     samples = np.arange(0, int(f_sampling * t_max), n_fft)
#
#     # Specify the list of time data points
#     time = np.arange(0, t_max, 1 / f_sampling)
#     # With these indexes, create the time-frequency grid for the pcolormesh plot of the spectogram
#     time_grd, f_grd = np.meshgrid(time[samples], np.append(f_fft, f_fft[-1] + f_fft[1] - f_fft[0]))
#
#     shape = (f_fft.size, time[samples].size)
#     x_fft = 1j * np.empty(shape)
#     psl = np.empty(shape)
#     psd = np.empty(shape)
#     pbl = np.empty(shape)
#     pbl_a = np.empty(shape)
#     ospl_f = np.empty(samples.size - 1)
#     oaspl_f = np.empty(samples.size - 1)
#
#     # for si in range(n_sensor):
#     si = 3
#     # Cycle through the windows using the indexes in the list "samples"
#     for ii, idx in enumerate(samples[:-1]):
#         # Determine X_m for the positive frequencies
#         x_fft[:, ii] = spfft.fft(data[idx:idx + n_fft, si])[:n_fft // 2]
#         # Determine P_m = |X_m|^2 * delta_t / N
#         psd[:, ii] = (np.abs(x_fft[:, ii]) ** 2) * t_sampling / n_fft
#         # Determine the PSL = 10 log(P_m / p_e0^2) where p_e0 = 2e-5 Pa
#         psl[:, ii] = 10 * np.log10(2 * psd[:, ii] / 4e-10)
#         # Determine the PBL using the PSL
#         pbl[:, ii] = psl[:, ii] + 10 * np.log10(delta_f)
#         pbl_a[:, ii] = pbl[:, ii] + a_weighting
#         # Determine the OSPLs with the PBLs of the signal
#         ospl_f[ii] = 10 * np.log10(np.sum(10 ** (pbl[:, ii] / 10)))
#         oaspl_f[ii] = 10 * np.log10(np.sum(10 ** (pbl_a[:, ii] / 10)))
#
#     time_pe = (time[samples][:-1] + time[samples][1:]) / 2
#     # All determined above in the loop for the spectogram, so just plot here
#     plt.figure(10 * si + 1)
#     plt.plot(time_pe, ospl_f, label="Normal")
#     plt.plot(time_pe, oaspl_f, label="A-weighted")
#     plt.xlabel('$t$ (s)')
#     plt.xlim(0, t_max)
#     plt.ylabel('$OSPL$ (dB)')
#     plt.grid()
#     plt.legend()
#     # plt.savefig(f'/home/josephine/Documents/EWEM/7 - MASTER THESIS/Demo_stuff/NTK_sensor{si + 1}.pdf')
#
#     # Plot the spectogram
#     plt.figure(10 * si + 2)
#     # Setting the lower and upper boundary value for the SPL
#     print(f'max(PSL) = {round(np.max(psl), 1)} dB / Hz')
#     vmin, vmax = -10, 30
#     # Create the spectogram and its colorbar
#     spectrogram = plt.pcolor(time[samples], f_fft / 1e3, psl, vmin=vmin, vmax=vmax)
#     cbar = plt.colorbar(spectrogram)
#     # Set the tick values on the colorbar with 5 dB/Hz separation
#     # Set all the labels
#     cbar.set_label('$PSL$ (dB / Hz)')
#     plt.xlabel('$t$ (s)')
#     plt.xlim(0, t_max)
#     plt.ylabel('$f$ (kHz)')
#     # plt.savefig(f'/home/josephine/Documents/EWEM/7 - MASTER THESIS/Demo_stuff/NTK_sensor{si + 1}.png')
#
#     plt.show()
#     plt.close('all')
