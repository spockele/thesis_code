import pysofaconventions as sofa
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spfft


class MitHrtf:
    def __init__(self, large=False):
        """
        Class that reads and stores the MIT HRTF function from the sofa file
        :param large: optional, give True for the large pinna data instead of the normal one
        """
        # Read the SOFA file with pysofaconventions
        size = "large" if large else "normal"
        file = sofa.SOFAFile(f'./helper_functions/hrtf/mit_kemar_{size}_pinna.sofa', 'r')
        # Extract the list of positions
        self.pos = file.getVariableValue('SourcePosition')
        # Get the sampling frequency and number of samples of the HRIRs
        fs = file.getSamplingRate()
        n = file.getDimension('N').size
        # Extract the HRIRs
        hrir = file.getDataIR()

        # Set the FFT frequency list
        self.f = spfft.fftfreq(n, 1 / fs)[:n // 2]
        # Create empty arrays for the HRTFs
        self.hrtf_l = 1j * np.empty((self.pos.shape[0], self.f.size))
        self.hrtf_r = 1j * np.empty((self.pos.shape[0], self.f.size))

        # Loop over the positions
        for pi, pos in enumerate(self.pos):
            self.pos[pi] = (pos[0] * np.pi / 180, pos[1] * np.pi / 180, pos[2])
            # Obtain the correct HRIRs
            hrir_l, hrir_r = hrir[pi, :, :]
            # FFT of the HRIRs are the HRTFs. Only care about amplitudes
            self.hrtf_l[pi] = spfft.fft(hrir_l)[:n // 2]
            self.hrtf_r[pi] = spfft.fft(hrir_r)[:n // 2]

    def get_hrtf(self):
        raise NotImplementedError("HAHA, f you")

    def plot_horizontal(self):
        x_l_lst = []
        x_r_lst = []
        f_lst = []
        th_lst = []

        for pi, pos in enumerate(self.pos):
            if pos[1] == 0.:
                th_lst.append(pos[0] * np.ones(self.f.size))
                f_lst.append(self.f)
                x_l_lst.append(20 * np.log10(2 * np.abs(self.hrtf_l[pi])))
                x_r_lst.append(20 * np.log10(2 * np.abs(self.hrtf_r[pi])))

        x_l_lst = np.array(x_l_lst)
        x_r_lst = np.array(x_r_lst)
        f_lst = np.array(f_lst)
        th_lst = np.degrees(np.array(th_lst))

        vmin = -40

        plt.figure(1)
        cmesh = plt.pcolormesh(f_lst, th_lst, x_l_lst, vmin=vmin, )
        cbar = plt.colorbar(cmesh)
        plt.xlabel('$f$ (Hz)')
        plt.ylabel('Azimuth (degrees)')
        cbar.set_label('(dB)')
        plt.tight_layout()
        cbar.set_ticks(np.append(np.arange(vmin, np.max(x_l_lst), 10), np.max(x_l_lst)))
        plt.yticks((0, 60, 120, 180, 240, 300, 360))
        # plt.savefig(f'/home/josephine/Documents/EWEM/7 - MASTER THESIS/Demo_stuff/HRTF_left.pdf')

        plt.figure(2)
        cmesh = plt.pcolormesh(f_lst, th_lst, x_r_lst, vmin=vmin, )
        cbar = plt.colorbar(cmesh)
        plt.xlabel('$f$ (Hz)')
        plt.ylabel('Azimuth (degrees)')
        cbar.set_label('(dB)')
        plt.tight_layout()
        cbar.set_ticks(np.append(np.arange(vmin, np.max(x_r_lst), 10), np.max(x_r_lst)))
        plt.yticks((0, 60, 120, 180, 240, 300, 360))
        # plt.savefig(f'/home/josephine/Documents/EWEM/7 - MASTER THESIS/Demo_stuff/HRTF_right.pdf')

        plt.show()


if __name__ == '__main__':
    raise RuntimeError('Thou shalt not run this module on its own!')
