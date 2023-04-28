import os
import matplotlib.pyplot as plt

import helper_functions as hf

import case_mgmt as cm
import source_model as sm
import propagation_model as pm


class Project:
    def __init__(self, project_path: str,):
        """

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

    def run_cases(self):
        """

        """
        for case in self.cases:
            case.run_hawc2()


if __name__ == '__main__':
    proj_path = os.path.abspath('NTK')
    case_obj = cm.Case(proj_path, 'ntk_05.5ms.aur')

    case_obj.generate_hawc2_sphere()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for point in case_obj.h2result_sphere:
        ax.scatter(*point.vec)

    plt.show()

    case_obj.run_hawc2()


    # source = sm.SourceModel(case_obj.conditions, case_obj.source, os.path.abspath('./NTK/H2model/res/055ms/'))
    # source.h2sphere.load_sphere()
    # print(source.h2sphere)
    # source.h2sphere.interpolate_sound(hf.Cartesian(0, 0, 0))







    # (n_sensor, f_sampling, n_samples), data = hf.read_ntk_data('./samples/NTK_Oct2016/nordtank_20150901_122400.tim')
    # # Sampling period
    # t_sampling = 1 / f_sampling
    # # Set the window sample size for the fft
    # n_fft = 2 ** 12
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
    # shape = (time_grd.shape[0] - 1, f_grd.shape[1] - 1)
    # x_fft = 1j * np.empty(shape)
    # psl = np.empty(shape)
    # psd = np.empty(shape)
    # pbl = np.empty(shape)
    # pbl_a = np.empty(shape)
    # ospl_f = np.empty(samples.size - 1)
    # oaspl_f = np.empty(samples.size - 1)
    #
    # for si in range(n_sensor):
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
    #     plt.xlim(0, 15)
    #     plt.ylabel('$OSPL$ (dB)')
    #     plt.grid()
    #     plt.legend()
    #     plt.savefig(f'/home/josephine/Documents/EWEM/7 - MASTER THESIS/Demo_stuff/NTK_sensor{si + 1}.pdf')
    #
    #     # Plot the spectogram
    #     plt.figure(10 * si + 2)
    #     # Setting the lower and upper boundary value for the SPL
    #     print(f'max(PSL) = {round(np.max(psl), 1)} dB / Hz')
    #     vmin, vmax = -10, 30
    #     # Create the spectogram and its colorbar
    #     spectrogram = plt.pcolormesh(time_grd, f_grd / 1e3, psl, cmap='jet', vmin=vmin, vmax=vmax)
    #     cbar = plt.colorbar(spectrogram)
    #     # Set the tick values on the colorbar with 5 dB/Hz separation
    #     cbar.set_ticks(np.arange(vmin, vmax + 5, 5))
    #     # Set all the labels
    #     cbar.set_label('$PSL$ (dB / Hz)')
    #     plt.xlabel('$t$ (s)')
    #     plt.xlim(0, 15)
    #     plt.ylabel('$f$ (kHz)')
    #     plt.savefig(f'/home/josephine/Documents/EWEM/7 - MASTER THESIS/Demo_stuff/NTK_sensor{si + 1}.png')
    #
    # plt.close()


    # hf.Atmosphere(generate=True)
    # atmosphere = hf.Atmosphere(50, 10)
    # # h = random.uniform(atmosphere.alt[0], atmosphere.alt[-1])
    # h = np.linspace(0, 1000, 5)
    # print(h, atmosphere.conditions(h)[-1])
    # atmosphere.plot()

    # hrtf = hf.MitHrtf()
    # hrtf.plot_horizontal()
