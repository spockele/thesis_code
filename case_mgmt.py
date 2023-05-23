import os
import queue
import numpy as np
import shutil as sh
import matplotlib.pyplot as plt
import pandas as pd
import scipy.fft as spfft
import scipy.signal as spsig
import scipy.io as spio
from wetb.hawc2 import HTCFile
from wetb.hawc2.htc_contents import HTCSection

import helper_functions as hf
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


class CaseLoader:
    # Some predefined values for the HAWC2.aero.aero_noise module
    octave_bandwidth = '1'
    output_filename = 'aeroload'

    def __init__(self, project_path: str, case_file: str):
        """
        ================================================================================================================
        Class to load an auralisation case defined by a .aur file.
        ================================================================================================================
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
        self.conditions_dict = {}
        self.source_dict = {}
        self.propagation_dict = {}
        self.receiver_dict = {}
        self.reception_dict = {}
        self.reconstruction_dict = {}
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

        # Go over all loaded blocks and send to respective parser functions
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
            # Parse propagation model block
            elif block_name == 'propagation':
                self._parse_propagation(block)
            # Parse reception model block
            elif block_name == 'reception':
                self._parse_reception(block)
            # Parse reconstruction model block
            elif block_name == 'reconstruction':
                self._parse_reconstruction(block)

    def _parse_conditions(self, lines: list):
        """
        Parse the conditions block into the conditions dictionary
        :param lines: list of lines containing auralisation input code
        """
        for line in lines[1:-1]:
            if not (line.startswith(';') or line.startswith('\n')):
                key, value, *_ = line.split(' ')

                if key in ('rotor_radius', 'wsp', 'z_wsp', 'z0_wsp', 'groundtemp', 'groundpres', 'humidity', 'delta_t'):
                    self.conditions_dict[key] = float(value)

                elif key in ():
                    self.conditions_dict[key] = int(value)

                elif key == 'hub_pos':
                    x, y, z = value.split(',')
                    self.conditions_dict[key] = hf.Cartesian(float(x), float(y), float(z))

                else:
                    self.conditions_dict[key] = value

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
        self.htc.aero.aero_noise.add_line(name='temperature', values=(self.conditions_dict['groundtemp'],),
                                          comments='')
        self.htc.aero.aero_noise.add_line(name='atmospheric_pressure', values=(self.conditions_dict['groundpres'],),
                                          comments='')
        self.htc.aero.aero_noise.add_line(name='output_filename', values=(self.output_filename, ),
                                          comments='')
        self.htc.aero.aero_noise.add_line(name='octave_bandwidth', values=(self.octave_bandwidth,),
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
        blocks = self._get_blocks(lines[1:-1])

        for key, value in blocks.items():
            if key in ('blade_percent', 'radius_factor'):
                self.source_dict[key] = float(value)

            elif key in ('n_rays', 'n_threads'):
                self.source_dict[key] = int(value)

            else:
                self.source_dict[key] = value

    def _parse_propagation(self, lines: list):
        """
        Parse the propagation block
        :param lines: list of lines containing auralisation input code
        """
        blocks = self._get_blocks(lines[1:-1])
        for key, value in blocks.items():
            if key in ():
                self.propagation_dict[key] = float(value)

            elif key in ('n_threads', ):
                self.propagation_dict[key] = int(value)

            elif key == 'models':
                self.propagation_dict[key] = tuple(value.split(','))

            else:
                self.propagation_dict[key] = value

    def _parse_receiver(self, lines: list):
        """
        Parse a reception > receiver block
        :param lines: list of lines containing auralisation input code
        """
        blocks = self._get_blocks(lines[1:-1])

        parse_dict = {}
        for key, value in blocks.items():
            if key in ('rotation', ):
                parse_dict[key] = float(value)

            elif key in ('index', ):
                parse_dict[key] = int(value)

            elif key in ('pos', ):
                parse_dict['pos'] = (float(val) for val in value.split(','))

            else:
                parse_dict[key] = value

        self.receiver_dict[parse_dict['index']] = rm.Receiver(parse_dict)

    def _parse_reception(self, lines: list):
        """
        Parse the reception block
        :param lines: list of lines containing auralisation input code
        """
        blocks = self._get_blocks(lines[1:-1])

        for key, block in blocks.items():
            if key in ():
                self.reception_dict[key] = float(block)

            elif key in ():
                self.reception_dict[key] = int(block)

            elif key == 'receiver':
                self._parse_receiver(block)

            else:
                self.reception_dict[key] = block

    def _parse_reconstruction(self, lines: list):
        """
        Parse the reconstruction block
        :param lines: list of lines containing auralisation input code
        """
        blocks = self._get_blocks(lines[1:-1])
        for key, value in blocks.items():
            if key in ():
                self.reconstruction_dict[key] = float(value)

            elif key in ():
                self.reconstruction_dict[key] = int(value)

            else:
                self.reconstruction_dict[key] = value


class Case(CaseLoader):
    def __init__(self, project_path: str, case_file: str):
        """
        ================================================================================================================
        Class to manage an auralisation case defined by a .aur file.
        ================================================================================================================
        :param project_path: path of the overarcing auralisation project folder.
        :param case_file: file name of the .aur file inside the project folder.
        """
        # Call the CaseLoader
        super().__init__(project_path, case_file)

        ''' Preparations for HAWC2 '''
        # Set the variables to store the properties of the HAWC2 sphere
        self.h2result_sphere = None
        self.h2result_path = os.path.join(self.h2model_path, 'res', self.case_name)

        ''' Setup of the models '''
        # Set the path for the atmosphere cache file
        self.atmosphere_path = os.path.join(self.project_path, f'atm/atm_{self.case_name}.dat')
        # Generate atmosphere if it does not exist yet
        if not os.path.isfile(self.atmosphere_path):
            self.atmosphere = hf.Atmosphere(self.conditions_dict['z_wsp'], self.conditions_dict['wsp'],
                                            self.conditions_dict['humidity'], wind_z0=self.conditions_dict['z0_wsp'],
                                            delta_h=1., t_0m=self.conditions_dict['groundtemp'],
                                            p_0m=self.conditions_dict['groundpres'], atm_path=self.atmosphere_path)
        # Otherwise, load the cache file
        else:
            self.atmosphere = hf.Atmosphere(self.conditions_dict['z_wsp'], self.conditions_dict['wsp'],
                                            self.conditions_dict['humidity'], wind_z0=self.conditions_dict['z0_wsp'],
                                            atm_path=self.atmosphere_path)

    def generate_hawc2_sphere(self):
        """
        Generate a sphere of evenly spaced observer points
        """
        coordinates, fail, _ = hf.uniform_spherical_grid(self.n_obs)
        if fail:
            raise ValueError(f'Parameter n_obs = {self.n_obs} resulted in incomplete sphere. Try a different value.')

        scale = self.source_dict['radius_factor'] * self.conditions_dict['rotor_radius']
        self.h2result_sphere = [scale * coordinate + self.conditions_dict['hub_pos'] for coordinate in coordinates]

        for pi, p in enumerate(self.h2result_sphere):
            self.htc.aero.aero_noise.add_line(name='xyz_observer', values=p.vec, comments=f'Observer_{pi}')

    def _simulate_hawc2(self):
        """
        Run the HTCFile.simulate function with compensation for its stupidity
        """
        try:
            self.htc.simulate(self.hawc2_path)
        # If an error is thrown, just smile and wave
        except Exception as e:
            return e

    def run_hawc2(self):
        """
        Run the HAWC2 simulations for this case.
        """
        ''' Preprocessing '''
        # Make sure the HAWC2 path is valid
        if not os.path.isfile(self.hawc2_path):
            raise FileNotFoundError('Invalid file path given for HAWC2.')

        print(' -- HAWC2')
        # Make sure an observer sphere is generated
        if self.h2result_sphere is None:
            self.generate_hawc2_sphere()
            print('Generating observer sphere: Done!')

        # Set the aero_noise simulation mode to 2, meaning to run and store the needed parameters
        self.htc.aero.aero_noise.add_line(name='noise_mode', values=('2', ), comments='Mode: Store')
        self.htc.save(self.htc_path)

        ''' Running HAWC2 '''
        # Create and start a progress thread to continuously print the progress and .....
        p_thread = hf.ProgressThread(2, 'Running HAWC2 simulations')
        p_thread.start()

        # Run the base simulation
        self._simulate_hawc2()

        # Set 1 to 2 in the progress thread
        p_thread.update()
        # Prepare for noise simulation by setting the noise mode to calculate
        self.htc.aero.aero_noise.add_line(name='noise_mode', values=('3', ), comments='Mode: Calculate')
        self.htc.save(self.htc_path)

        # Run the noise simulation
        self._simulate_hawc2()

        # Stop the progress thread
        p_thread.stop()
        del p_thread

        ''' Postprocessing '''
        # Remove the temp htc file
        os.remove(self.htc_path)

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

    def run(self):
        """
        Run everything except HAWC2
        """
        source_model = sm.SourceModel(self.conditions_dict, self.source_dict, self.h2result_path, self.atmosphere)
        propagation_model = pm.PropagationModel(self.conditions_dict, self.propagation_dict, self.atmosphere)
        reception_model = rm.ReceptionModel(self.conditions_dict, self.reception_dict)

        receiver: rm.Receiver = self.receiver_dict[0]

        print(f' -- Running Propagation Model for receiver {0}')
        ray_queue: queue.Queue = source_model.run(receiver, self.propagation_dict['models'])
        ray_queue: queue.Queue = propagation_model.run(receiver, ray_queue)

        # propagation_model.pickle_ray_queue(ray_queue)
        # ray_queue = pm.PropagationModel.unpickle_ray_queue()

        print(f' -- Running Reception Model for receiver {0}')
        reception_model.run(receiver, ray_queue)

        spectrogram_path = os.path.join(self.project_path, 'spectrograms', f'spectrogram_{self.case_name}_rec{0}.csv')
        receiver.spectrogram.to_csv(spectrogram_path)

        # --------------------------------------------------------------------------------------------------------------
        # Histogram plot(s)
        # --------------------------------------------------------------------------------------------------------------
        t_rdb = [round(t, 10) for t in sorted(reception_model.rays.keys())]
        histogram = pd.DataFrame(0, index=np.arange(1, 99 + 2, 2), columns=t_rdb)
        for t in t_rdb:
            for ray in reception_model.rays[t]:
                spectrum = ray.spectrum['a'] * ray.spectrum['gaussian']
                energy = np.trapz(spectrum, spectrum.index)
                if energy > 0:
                    energy = 10 * np.log10(energy / hf.p_ref ** 2)

                    bin_e = 2 * int(energy // 2) + 1
                    if bin_e in histogram.index:
                        histogram.loc[bin_e, t] += 1
        plt.figure(10)
        ctr = plt.pcolor(histogram.columns, histogram.index, histogram)
        cbr = plt.colorbar(ctr)

        plt.xlabel('t (s)')
        plt.ylabel('Received OSPL of Sound Ray (dB) (binned per 2 dB)')
        cbr.set_label('Number of Received Sound Rays (-)')

        received: dict = receiver.received

        t = sorted(received.keys())
        n = np.array([len(received[t]) for t in sorted(received.keys())])

        plt.figure(11)
        plt.plot(t, n)
        plt.xlabel('t (s)')
        plt.ylabel('$N_{rays}$ (-)')

        # --------------------------------------------------------------------------------------------------------------
        # Sound reconstruction
        # --------------------------------------------------------------------------------------------------------------

        spectrogram = pd.read_csv(os.path.join(self.project_path, 'spectrograms', f'spectrogram_{self.case_name}_rec{0}.csv'),
                                  header=0, index_col=0).applymap(complex)
        spectrogram.columns = spectrogram.columns.astype(float)
        spectrogram.index = spectrogram.index.astype(float)

        n_base = 512
        n_fft = n_base * 8
        n_perseg = n_base * 2
        x_fft = 1j * np.zeros((spectrogram.columns.size, n_fft // 2))
        f_s_desired = n_base * 1e2
        f = spfft.fftfreq(n_fft, 1 / f_s_desired)[:n_fft // 2]
        f_octave = hf.octave_band_fc(1)
        for ti, t in enumerate(spectrogram.columns):
            x_spectrogram = spectrogram.loc[:, t].to_numpy()
            x_fft[ti] = np.sqrt(np.interp(f, f_octave, x_spectrogram)) * np.exp(
                1j * np.random.default_rng().uniform(0, 2 * np.pi, f.size))

        # --------------------------------------------------------------------------------------------------------------
        # Spectrograms and sound plots
        # --------------------------------------------------------------------------------------------------------------
        plt.figure(1)
        ctr = plt.pcolor(spectrogram.columns, f, 20 * np.log10(np.abs(x_fft.T) / hf.p_ref), vmin=0, vmax=40)
        cbar = plt.colorbar(ctr)
        plt.xlabel('t (s)')
        plt.ylabel('f (Hz)')
        cbar.set_label('PSL (dB / Hz)')

        t, x = spsig.istft(x_fft.T, f_s_desired, nfft=n_fft, nperseg=n_perseg, noverlap=n_perseg - n_base,
                           window=('tukey', .75))
        plt.figure(3)
        plt.plot(t, x)
        plt.xlabel('t (s)')
        plt.ylabel('p (Pa)')

        plt.figure(4)
        plt.plot(t, 20 * np.log10(np.abs(x) / hf.p_ref))
        plt.xlabel('t (s)')
        plt.ylabel('Pressure level (dB)')
        plt.ylim(-60, 80)

        plt.figure(2)
        f_stft, t_stft, x_stft = spsig.stft(x, f_s_desired)
        ctr = plt.pcolor(t_stft, f_stft, 20 * np.log10(np.abs(x_stft) / hf.p_ref), vmin=0, vmax=40)
        cbar = plt.colorbar(ctr)
        plt.xlabel('t (s)')
        plt.ylabel('f (Hz)')
        cbar.set_label('PSL (dB / Hz)')

        # plt.close('all')
        plt.show()

        # Longer sound files :)
        rotation_time = 60 / 27.1  # 1 / RPM

        x_rotation = x[t >= t[-1] - rotation_time]
        t_rotation = t[t >= t[-1] - rotation_time] - (t[-1] - rotation_time)

        x_rotation[t_rotation > t_rotation[-1] - (t[-1] - rotation_time)] += x[t < t[-1] - rotation_time]
        x_long = np.tile(x_rotation, 10)

        # Normalise to 80 dB (2e-1 Pa)
        wav_dat = (np.clip(x_long / 2e-1, -1, 1) * 32767).astype(np.int16)
        spio.wavfile.write(f'{self.case_name}_rec{0}.wav', int(f_s_desired), wav_dat)
