import random
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.fft as spfft
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
    """
    BPM directivity pattern (Brooks et al. 1989, Equation B1)
    :param theta: angle in the rotated zy-plane in Brooks et al. 1989, Figure B3
    :param phi: angle in the yz-plane defined in Brooks et al. 1989, Figure B3
    :param mach: mach number in the airfoil motion direction
    :return: the directivity value of BPM at the given angular coordinate and mach number
    """
    # Limit the angles to [-pi, pi]
    theta = hf.limit_angle(theta)
    phi = hf.limit_angle(phi)
    # Determine convective mach number with assumption m_c = .8 M
    mach_conv = .8 * mach
    # Determine the numerator and denominator of the BPM directivity (Brooks et al. 1989, Equation B1)
    num = 2 * ((np.sin(.5 * theta)) ** 2) * np.sin(phi) ** 2
    den = (1 + mach * np.cos(theta)) * (1 + (mach - mach_conv) * np.cos(theta)) ** 2
    # Return the sh!t out of the directivity value
    return num / den


def uniform_spherical_grid(n_points):
    """
    Create a uniform grid in spherical coordinates where each point represents an equal fraction of the surface area
    of the unit sphere. Algorithm by Deserno, 2004.
    ------------------------------------------------------------------------------------------------
    !!! NOTE: output is not guaranteed to contain exact number of points input into the function !!!
    ------------------------------------------------------------------------------------------------
    :param n_points: desired number of points in the grid
    :return: the polar and azimuth angles in numpy arrays of length n_count
    """
    # Area represented per point of the grid
    point_area = 4 * np.pi / n_points
    # Length scale associated with this area
    dist = np.sqrt(point_area)
    # number of polar coordinates
    m_pol = round(np.pi / dist)
    # Angular distance between points in polar direction
    d_pol = np.pi / m_pol
    # Angular distance between points in azimuth direction
    d_azi = point_area / d_pol

    # Create counter and arrays for output
    n_count = 0
    polar, azimuth = np.empty((2, n_points))
    # Loop in the polar direction
    for mi in range(m_pol):
        # Determine polar angle
        pol = np.pi * (mi + .5) / m_pol
        # Determine number of azimuthal coordinates
        m_azi = round(2 * np.pi * np.sin(pol) / d_azi)
        # Loop over azimuth angles
        for ni in range(m_azi):
            # Add polar angle to output array
            polar[n_count] = pol
            # Determine azimuth angle and add to output array
            azimuth[n_count] = 2 * np.pi * ni / m_azi
            # Up the counter by 1
            n_count += 1

    # Output the output arrays, shortened to actual number of generated points
    return polar[:n_count + 1], azimuth[:n_count + 1]


if __name__ == '__main__':
    # phg, thg = uniform_spherical_grid(1000)
    # #
    # # d = directivity(phg, thg, .9)
    # #
    # y = np.sin(phg) * np.cos(thg)
    # z = np.sin(phg) * np.sin(thg)
    # x = np.cos(phg)
    # #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # # idx = d > 10 ** (np.log10(np.max(d)) - 5)
    # # ax.scatter(x[idx], y[idx], z[idx], c=10 * np.log10(d[idx]), cmap='jet', )
    # ax.scatter(x, y, z, 'k')
    #
    # ax.set_aspect('equal')
    # # Create cubic bounding box to simulate equal aspect ratio
    # max_range = np.max([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
    # xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
    # yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
    # zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
    # for xbb, ybb, zbb in zip(xb, yb, zb):
    #     ax.plot([xbb], [ybb], [zbb], 'w')
    # plt.show()

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


    hf.Atmosphere(generate=True)
    atmosphere = hf.Atmosphere()
    # h = random.uniform(atmosphere.alt[0], atmosphere.alt[-1])
    h = np.linspace(0, 1000, 5)
    print(h, atmosphere.conditions(h)[-1])
    atmosphere.plot()
