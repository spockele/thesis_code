import os
import numpy as np
import shutil as sh
from wetb.hawc2 import HTCFile
from wetb.hawc2.htc_contents import HTCSection

import helper_functions as hf
import propagation_model as pm


class CaseLoader:
    def __init__(self, project_path: str, case_file: str):
        """
        Class to load an auralisation case defined by a .aur file.
        :param project_path: path of the overarcing auralisation project folder.
        :param case_file: file name of the .aur file inside the project folder.
        """
        ''' Preprocessing of the input parameters '''
        # Create paths for project and for the HAWC2 model.
        self.project_path = project_path
        self.h2model_path = os.path.join(project_path, 'H2model')

        self.case_file = os.path.join(project_path, case_file)
        # Check that the input file actually exists.
        if not os.path.isfile(self.case_file):
            raise FileNotFoundError('Given input file name does not exist.')

        ''' Input file parsing process '''
        # Open the input file and read the lines
        with open(self.case_file, 'r') as f:
            lines = f.readlines()
        # Remove all leading and trailing spaces from the lines for parsing
        lines = [line.strip(' ') for line in lines]

        # Dictionaries to store inputs for the models
        self.conditions = {}
        self.source = {}
        self.propagation = {}
        self.reception = {}
        self.reconstruction = {}
        # File paths and names
        self.case_name = ''
        self.htc_base_name = ''
        self.htc_base_path = ''
        self.htc_path = ''
        self.hawc2_path = ''
        # Placeholder for the real .htc file
        self.htc = HTCFile()
        # Parameters from the hawc2 input block
        self.n_obs = 0

        # Parse the input file lines
        self._parse_input_file(lines)

    @staticmethod
    def _get_blocks(lines: list):
        """
        Obtain code blocks from the given list of lines
        :param lines: list of lines containing auralisation input code
        :return: a dictionary containing the code blocks. dict(str: list(str))
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

            # Put non-block lines in the dictionary as well
            elif not (line.startswith(';') or line.startswith('\n')):
                key, value, *_ = line.split(' ')
                blocks[key] = value

        return blocks

    def _parse_input_file(self, lines: list):
        """
        Parse the input file into the blocks and then parse those.
        :param lines: list of lines containing auralisation input code
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
        :param lines: list of lines containing auralisation input code
        """
        self.conditions = dict()
        for line in lines[1:-1]:
            if not (line.startswith(';') or line.startswith('\n')):
                key, value, *_ = line.split(' ')

                if key in ('wsp', 'groundtemp', 'groundpres', 'hub_height', 'rotor_radius', 'z0_wsp', 'z_wsp'):
                    self.conditions[key] = float(value)

                elif key in ():
                    self.conditions[key] = int(value)

                else:
                    self.conditions[key] = value

    def _parse_hawc2(self, lines: list):
        """
        Parse the HAWC2 block and create the case .htc file
        :param lines: list of lines containing auralisation input code
        """
        # Get the blocks from the HAWC2 block
        blocks = self._get_blocks(lines[1:-1])

        # Get the name and path of the base .htc file to work with
        self.htc_base_name = blocks['htc_name']
        self.htc_base_path = os.path.join(self.h2model_path, f'{self.htc_base_name}.htc')
        # Load the base htc file
        self.htc = HTCFile(filename=self.htc_base_path)

        # Add the case "wind" and "aero_noise" sections
        self.htc.add_section(HTCSection.from_lines(blocks['wind']))
        self.htc.aero.add_section(HTCSection.from_lines(blocks['aero_noise']))

        # Add necessary parameters to the "aero_noise" section
        self.htc.aero.aero_noise.add_line(name='output_filename', values=('aeroload', ),
                                          comments='')
        self.htc.aero.aero_noise.add_line(name='temperature', values=(self.conditions['groundtemp'],),
                                          comments='')
        self.htc.aero.aero_noise.add_line(name='atmospheric_pressure', values=(self.conditions['groundpres'],),
                                          comments='')
        self.htc.aero.aero_noise.add_line(name='octave_bandwidth', values=('24',),
                                          comments='')

        # Create the path for the case-specific htc file and save the htc file
        self.htc_path = os.path.join(self.h2model_path, f'{self.htc_base_name}_{self.case_name}.htc')

        # Extract the HAWC2 executable file location
        self.hawc2_path = os.path.join(blocks['hawc2_path'])

        # Extract the number of observer points
        self.n_obs = int(blocks['n_obs'])

    def _parse_source(self, lines: list):
        """
        Parse the source block
        :param lines: list of lines containing auralisation input code
        """
        self.propagation = dict()
        for line in lines[1:-1]:
            if not (line.startswith(';') or line.startswith('\n')):
                key, value, *_ = line.split(' ')

                if key in ('n_rays',):
                    self.conditions[key] = float(value)

                elif key in ():
                    self.conditions[key] = int(value)

                else:
                    self.conditions[key] = value

    def _parse_propagation(self, lines: list):
        """
        Parse the propagation block
        :param lines: list of lines containing auralisation input code
        """
        self.propagation = dict()
        for line in lines[1:-1]:
            if not (line.startswith(';') or line.startswith('\n')):
                key, value, *_ = line.split(' ')

                if key in ():
                    self.conditions[key] = float(value)

                elif key in ():
                    self.conditions[key] = int(value)

                else:
                    self.conditions[key] = value

    def _parse_reception(self, lines: list):
        """
        Parse the reception block
        :param lines: list of lines containing auralisation input code
        """
        pass

    def _parse_reconstruction(self, lines: list):
        """
        Parse the reconstruction block
        :param lines: list of lines containing auralisation input code
        """
        pass


class Case(CaseLoader):
    def __init__(self, project_path: str, case_file: str):
        """
        Class to manage an auralisation case defined by a .aur file.
        :param project_path: path of the overarcing auralisation project folder.
        :param case_file: file name of the .aur file inside the project folder.
        """
        # Call the CaseLoader
        super().__init__(project_path, case_file)

        ''' Preparations for HAWC2 '''
        # Set the variables to store the properties of the HAWC2 sphere
        self.obs_sphere = None
        self.h2result_path = os.path.join(self.h2model_path, 'res', self.case_name)

        ''' Setup of the models '''
        # Set the path for the atmosphere cache file
        self.atmosphere_path = os.path.join(self.project_path, f'atm/atm_{self.case_name}.dat')
        # Generate atmosphere if it does not exist yet
        if not os.path.isfile(self.atmosphere_path):
            self.atmosphere = hf.Atmosphere(self.conditions['z_wsp'], self.conditions['wsp'], self.conditions['z0_wsp'],
                                            1., self.conditions['groundtemp'], self.conditions['groundpres'],
                                            atm_path=self.atmosphere_path)
        # Otherwise, load the cache file
        else:
            self.atmosphere = hf.Atmosphere(self.conditions['z_wsp'], self.conditions['wsp'], self.conditions['z0_wsp'],
                                            atm_path=self.atmosphere_path)

    def generate_hawc2_sphere(self, ):
        """
        Generate a sphere of evenly spaced observer points
        """
        *coordinates, fail, pd = hf.uniform_spherical_grid(self.n_obs)
        coo = np.array(coordinates)

        offset = hf.Cartesian(0, 0, -self.conditions['hub_height'])

        self.obs_sphere = [hf.Cartesian(self.conditions['rotor_radius'] * np.cos(coo[1][idx]) * np.sin(coo[1][idx]),
                           self.conditions['rotor_radius'] * np.sin(coo[1][idx]) * np.sin(coo[1][idx]),
                           self.conditions['rotor_radius'] * np.cos(coo[1][idx])) + offset
                           for idx in range(coo.shape[1])]

        for pi, p in enumerate(self.obs_sphere):
            self.htc.aero.aero_noise.add_line(name='xyz_observer', values=p.vec, comments=f'Observer_{pi}')

    def run_hawc2(self, ):
        """
        Run the HAWC2 simulations for this case.
        """
        ''' Preprocessing '''
        # Make sure an observer sphere is generated
        if self.obs_sphere is None:
            self.generate_hawc2_sphere()
        # Make sure the HAWC2 path is valid
        if not os.path.isfile(self.hawc2_path):
            raise FileNotFoundError('Invalid file path given for HAWC2.')

        # Set the aero_noise simulation mode to 2, meaning to run and store the needed parameters
        self.htc.aero.aero_noise.add_line(name='noise_mode', values=('2', ), comments='Mode: Store')
        self.htc.save(self.htc_path)

        ''' Running HAWC2 '''
        # # Create and start a progress thread to continuously print the progress and .....
        # p_thread = hf.ProgressThread(2, 'Running HAWC2 simulation')
        # p_thread.start()
        # # Run the simulation
        # self.htc.simulate(self.hawc2_path)
        # # Set 1 to 2 in the progress thread
        # p_thread.update()

        # Prepare for noise simulation by setting the noise mode to calculate
        self.htc.aero.aero_noise.add_line(name='noise_mode', values=('3', ), comments='Mode: Calculate')
        self.htc.save(self.htc_path)
        # # Run the noise simulation
        # self.htc.simulate(self.hawc2_path)
        # # Stop the progress thread
        # p_thread.stop()

        ''' Postprocessing '''
        # Remove the temp htc file
        os.remove(case_obj.htc_path)

        # Create a directory to move noise results to if it does not exist yet
        if not os.path.isdir(self.h2result_path):
            os.mkdir(self.h2result_path)

        # Loop over all generated result files
        for fname in os.listdir(os.path.join(self.h2model_path, 'res')):
            # Create the path string for the current result file
            fpath = os.path.join(self.h2model_path, 'res', fname)
            # Store the base HAWC2 output files
            if fname.endswith('.sel') or fname.endswith('.dat'):
                sh.move(fpath, self.h2result_path)
            # Store the noise output files
            elif fname.endswith('.out'):
                # But only the relevant ones...
                if fname.startswith('aeroload_noise_psd'):
                    sh.move(fpath, self.h2result_path)
                # Remove the other ones
                else:
                    os.remove(fpath)


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
        self.cases = [Case(self.project_path, aur_file)
                      for aur_file in os.listdir(self.project_path) if aur_file.endswith('.aur')]

        if len(self.cases) <= 0:
            raise FileNotFoundError('No input files found in project folder.')

    def run_cases(self):
        """

        """
        for case in self.cases:
            case.run_hawc2()


if __name__ == '__main__':
    # project = Project(os.path.abspath('NTK'))
    proj_path = os.path.abspath('NTK')
    case_obj = Case(proj_path, 'ntk_05.5ms.aur')
    case_obj.run_hawc2()
    # print(case_obj.htc)










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
