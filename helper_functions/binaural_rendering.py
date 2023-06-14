import os
import numpy as np
import scipy.fft as spfft
import matplotlib.pyplot as plt
import pysofaconventions as sofa
import scipy.interpolate as spint

from . import limit_angle, c, HeadRelatedSpherical


"""
========================================================================================================================
===                                                                                                                  ===
=== Definition of the MIT measured HRTF function                                                                     ===
===                                                                                                                  ===
========================================================================================================================
"""
__all__ = ["MITHrtf", "woodworth_itd", "plot_woodworth_itd"]


head_radius = 87.5e-3  # (m)


def woodworth_itd(angle: float) -> (str, float):
    """
    Get the Inter-aural Time Delay at a given angle of incoming sound. Based on:
     - Woodworth, R. S. (1938). Experimental psychology. Holt.
     - Aaronson, N. L., & Hartmann, W. M. (2014). Testing, correcting, and extending the Woodworth model for interaural
        time difference. The Journal of the Acoustical Society of America, 135(2), 817–823. doi: 10.1121/1.4861243

    :param angle: azimuth angle of the incoming sound direction (radians)
    :return: the side of the head and the ITD
    """
    # Determine the side of the head the sound is coming at first
    side = {-1.: 'left', 0.: 'centre', 1.: 'right'}[np.sign(limit_angle(angle))]
    # Take the absolute value of the angle, as WoodWorth is defined for one side only
    angle = abs(limit_angle(angle))
    # Return the correct ITD
    if angle <= np.pi/2:
        return side, (head_radius / c) * (angle + np.sin(angle))

    elif np.pi/2 < angle <= np.pi:
        return side, (head_radius / c) * (np.pi - angle + np.sin(angle))

    else:
        raise ValueError(f'Given angle invalid.')


def plot_woodworth_itd() -> None:
    """
    Make a plot of the above woodworth inter-aural time delay function
    """
    plt.figure(1, figsize=(4.8, 4.8))
    ax = plt.subplot(projection='polar')
    th = np.linspace(-np.pi, np.pi, 361)

    ax.plot(th, [woodworth_itd(angle)[1] * 1e3 for angle in th], label='ITD (ms)')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.legend(loc=3, bbox_to_anchor=(.8, .95))
    plt.tight_layout()
    plt.show()


class MITHrtf:
    def __init__(self, large=False):
        """
        Class that reads and stores the MIT HRTF function from the sofa file
        :param large: optional, give True for the large pinna data instead of the normal one
        """
        # Read the SOFA file with pysofaconventions
        size = "large" if large else "normal"
        path = os.path.abspath(f'./helper_functions/data/mit_kemar_{size}_pinna.sofa')
        file = sofa.SOFAFile(path, 'r')

        # Extract the list of positions
        pos_lst = file.getVariableValue('SourcePosition')
        # Get the sampling frequency and number of samples of the HRIRs
        fs = file.getSamplingRate()
        n = file.getDimension('N').size
        # Extract the HRIRs
        hrir = file.getDataIR()

        # Set the FFT frequency list
        self.f = spfft.fftfreq(n, 1 / fs)[:n // 2]
        # Create empty arrays for the HRTFs
        self.hrtf_l = 1j * np.empty((pos_lst.shape[0], self.f.size))
        self.hrtf_r = 1j * np.empty((pos_lst.shape[0], self.f.size))

        self.azimuth = np.empty((pos_lst.shape[0], ))
        self.polar = np.empty((pos_lst.shape[0], ))
        # Loop over the positions
        for pi, pos in enumerate(pos_lst):
            # Store azimuth angle, which is opposite in my coordinate systems compared to the HRTF files
            self.azimuth[pi] = -limit_angle(pos[0] * np.pi / 180)
            # Store polar angle
            self.polar[pi] = limit_angle(pos[1] * np.pi / 180)

            # Obtain the correct HRIRs
            hrir_l, hrir_r = hrir[pi, :, :]
            # FFT of the HRIRs are the HRTFs
            self.hrtf_l[pi] = spfft.fft(hrir_l)[:n // 2]
            self.hrtf_r[pi] = spfft.fft(hrir_r)[:n // 2]

        self.points = np.vstack((self.azimuth, self.polar)).T

    def get_hrtf(self, direction: HeadRelatedSpherical) -> (np.array, np.array):
        """
        Returns the HRTF functions closest to the given azimuth and polar angle
        :param direction: HeadRelatedSpherical containing the direction where the sound is coming from
        :return: numpy arrays containing the HRTF for the left and right ear
        """
        hrtf_l = spint.griddata(self.points, self.hrtf_l, (direction[1], direction[2]), method='nearest')
        hrtf_r = spint.griddata(self.points, self.hrtf_r, (direction[1], direction[2]), method='nearest')

        return hrtf_l, hrtf_r

    def plot_horizontal(self):
        """
        Plot the HRTF for all azimuth angles in the horizontal plane in dB
        """
        th_lst = self.azimuth[self.polar == 0.]
        th_sort = np.argsort(th_lst)
        th_lst = th_lst[th_sort]
        x_l_lst = 20 * np.log10(np.abs(self.hrtf_l[self.polar == 0., :]))[th_sort]
        x_r_lst = 20 * np.log10(np.abs(self.hrtf_r[self.polar == 0., :]))[th_sort]
        f_lst = self.f

        # Define the lowest value for the colorbar
        vmin = -40
        # Plot for the left ear
        plt.figure(1)
        cmesh = plt.pcolor(f_lst, th_lst, x_l_lst, vmin=vmin, )
        cbar = plt.colorbar(cmesh)
        plt.xlabel('$f$ (Hz)')
        plt.ylabel('Azimuth (degrees)')
        cbar.set_label('(dB)')
        plt.tight_layout()
        cbar.set_ticks(np.append(np.arange(vmin, np.max(x_l_lst), 10), np.max(x_l_lst)))
        # plt.yticks((-180, -120, -60, 0, 60, 120, 180, ))
        plt.savefig('./plots/HRTF_left.png')
        # Plot for the right ear
        plt.figure(2)
        cmesh = plt.pcolor(f_lst, th_lst, x_r_lst, vmin=vmin, )
        cbar = plt.colorbar(cmesh)
        plt.xlabel('$f$ (Hz)')
        plt.ylabel('Azimuth (degrees)')
        cbar.set_label('(dB)')
        plt.tight_layout()
        cbar.set_ticks(np.append(np.arange(vmin, np.max(x_r_lst), 10), np.max(x_r_lst)))
        # plt.yticks((-180, -120, -60, 0, 60, 120, 180, ))
        plt.savefig('./plots/HRTF_right.png')

        plt.show()

