import os
import numpy as np
from wetb.hawc2 import HTCFile
from wetb.hawc2.htc_contents import HTCSection

import helper_functions as hf


class Case:
    def __init__(self, project_path: str, case_file: str):
        # Create paths for project and for the HAWC2 model.
        self.project_path = project_path
        self.h2model_path = os.path.join(project_path, 'H2model')

        self.case_file = os.path.join(project_path, case_file)
        # Check that the input file actually exists.
        if not os.path.isfile(self.case_file):
            raise FileNotFoundError('Given input file name does not exist.')

        # Open the input file and read the lines
        with open(self.case_file, 'r') as f:
            lines = f.readlines()
        # Remove all leading and trailing spaces from the lines for parsing
        lines = [line.strip(' ') for line in lines]

        # Create variables for the parsing process
        self.conditions = {}
        self.source = {}
        self.case_name = ''
        self.htc_base_name = ''
        self.htc_base_path = ''
        self.htc_path = ''
        self.htc = HTCFile()

        # Parse the input file lines
        self._parse_input_file(lines)

    @staticmethod
    def _get_blocks(lines: list):
        """

        :param lines:
        :return:
        """
        # Create empty list for returning the blocks
        blocks = {}
        # Go over all the lines
        while lines:
            line = lines.pop(0)
            # Figure out the blocks
            if line.startswith('begin'):
                # Get the name of the current block
                _, block_name, *_ = line.split(' ')
                # Create block collection list
                block = [line, ]
                # Obtain all lines in the current block
                while not block[-1].strip(' ').startswith(f'end {block_name}'):
                    block.append(lines.pop(0))
                # Add this block to the list
                blocks[block_name] = block

            elif not (line.startswith(';') or line.startswith('\n')):
                key, value, *_ = line.split(' ')
                blocks[key] = value

        return blocks

    def _parse_input_file(self, lines: list):
        """
        Parse the input file into the blocks and then parse those.
        :param lines:
        :return:
        """
        # Obtain the blocks from the input file lines
        blocks = self._get_blocks(lines)
        self.case_name = blocks['name']
        # Check if input file contains required stuff
        for block_name_required in ('conditions', 'HAWC2', 'source', 'propagation', 'reception', 'reconstruction'):
            if block_name_required not in blocks.keys():
                raise KeyError(f'No {block_name_required} block found in {self.case_name}')

        for block_name, block in blocks.items():
            # Parse Conditions block
            if block_name == 'conditions':
                self._parse_conditions(block)
            # Parse HAWC2 block
            elif block_name == 'HAWC2':
                self._parse_hawc2(block)
            # Parse source model block
            elif block_name == 'source':
                self._parse_source(block)

    def _parse_conditions(self, lines: list):
        """
        Parse the conditions block into the conditions dictionary
        :param lines:
        :return:
        """
        self.conditions = dict()
        for line in lines[1:-1]:
            key, value, *_ = line.split(' ')

            if key in ('wsp', 'groundtemp', 'groundpres',):
                self.conditions[key] = float(value)

            elif key in ():
                self.conditions[key] = int(value)

            else:
                self.conditions[key] = value

    def _parse_hawc2(self, lines: list):
        """
        Parse the HAWC2 block and create the case .htc file
        :param lines:
        :return:
        """
        # Get the blocks from the HAWC2 block
        blocks = self._get_blocks(lines[1:-1])
        # Get the name and path of the base .htc file to work with
        self.htc_base_name = blocks["htc_name"]
        self.htc_base_path = os.path.join(self.h2model_path, f'{blocks["htc_name"]}.htc')
        # Load the base htc file
        self.htc = HTCFile(filename=self.htc_base_path)
        # Add the case "wind" and "aero_noise" sections
        self.htc.add_section(HTCSection.from_lines(blocks['wind']))
        self.htc.aero.add_section(HTCSection.from_lines(blocks['aero_noise']))
        # Create the path for the case-specific htc file and save the htc file
        self.htc_path = os.path.join(self.h2model_path, f'{self.htc_base_name}_{self.case_name}.htc')
        self.htc.save(self.htc_path)

    def _parse_source(self, lines: list):
        pass

    def _parse_propagation(self, lines: list):
        pass

    def _parse_reception(self, lines: list):
        pass

    def _parse_reconstruction(self, lines: list):
        pass

    def run_hawc2(self, hawc2_path: str):
        pass


class Project:
    def __init__(self, project_path: str,):
        # Check if project folder exists.
        if not os.path.isdir(project_path):
            raise NotADirectoryError('Invalid project folder path given.')

        # Create paths for project and for the HAWC2 model
        self.project_path = project_path
        self.h2model_path = os.path.join(project_path, 'H2model')

        # Check that the project contains a HAWC2 model
        if not os.path.isdir(self.h2model_path):
            raise NotADirectoryError('The given project folder does not contain a HAWC2 model in folder "H2model".')

        # Obtain cases from the project folder
        self.cases = [Case(self.project_path, aur_file)
                      for aur_file in os.listdir(self.project_path) if aur_file.endswith('.aur')]

        if len(self.cases) <= 0:
            raise FileNotFoundError('No input files found in project folder.')

if __name__ == '__main__':
    # project = Project(os.path.abspath('NTK'))
    proj_path = os.path.abspath('NTK')
    case = Case(proj_path, 'ntk_05.5ms.aur')









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
