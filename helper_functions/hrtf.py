import pysofaconventions as sofa
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spfft


def sofa_suf():
    file = sofa.SOFAFile('./hrtf/mit_kemar_normal_pinna.sofa', 'r')
    sourcepos = file.getVariableValue('SourcePosition')
    fs = file.getSamplingRate()
    n = file.getDimension('N').size

    f_fft = spfft.fftfreq(n, 1/fs)[:n//2]
    x_l_lst = []
    x_r_lst = []
    f_lst = []
    th_lst = []
    for pi, pos in enumerate(sourcepos):
        if pos[1] == 0.:
            hrtf_l, hrtf_r = file.getDataIR()[pi, :, :]

            x_l = spfft.fft(hrtf_l)[:n//2]
            x_r = spfft.fft(hrtf_r)[:n//2]

            th_lst.append(pos[0] * np.ones(f_fft.size))
            f_lst.append(f_fft)
            x_l_lst.append(10 * np.log10(np.abs(x_l)))
            x_r_lst.append(10 * np.log10(np.abs(x_r)))

    x_l_lst = np.array(x_l_lst)
    x_r_lst = np.array(x_r_lst)
    f_lst = np.array(f_lst)
    th_lst = np.array(th_lst)

    plt.figure(1)
    cmesh = plt.pcolormesh(f_lst, th_lst, x_l_lst, vmin=-20, )
    cbar = plt.colorbar(cmesh)
    plt.xlabel('$f$ (Hz)')
    plt.ylabel('Azimuth (degrees)')
    cbar.set_label('(dB)')
    plt.tight_layout()
    cbar.set_ticks(np.append(np.arange(-20, 10, 5), np.max(x_l_lst)))
    plt.savefig(f'/home/josephine/Documents/EWEM/7 - MASTER THESIS/Demo_stuff/HRTF_left.pdf')

    plt.figure(2)
    cmesh = plt.pcolormesh(f_lst, th_lst, x_r_lst, vmin=-20, )
    cbar = plt.colorbar(cmesh)
    plt.xlabel('$f$ (Hz)')
    plt.ylabel('Azimuth (degrees)')
    cbar.set_label('(dB)')
    plt.tight_layout()
    cbar.set_ticks(np.append(np.arange(-20, 10, 5), np.max(x_r_lst)))
    plt.savefig(f'/home/josephine/Documents/EWEM/7 - MASTER THESIS/Demo_stuff/HRTF_right.pdf')

    plt.show()


if __name__ == '__main__':
    sofa_suf()
