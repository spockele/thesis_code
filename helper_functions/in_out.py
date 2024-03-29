import numpy as np
import pandas as pd
import scipy.io as spio


"""
========================================================================================================================
===                                                                                                                  ===
=== I/O functions for data files used in this code                                                                   ===
===                                                                                                                  ===
========================================================================================================================

Copyright (c) 2023 Josephine Pockelé. Licensed under MIT license.

"""
__all__ = ["write_to_file", "read_from_file", "read_hawc2_aero_noise", "read_ntk_data", ]


def write_to_file(array, path: str):
    """
    Write a 2D array to a file
    :param array: A 2D numpy array or python list
    :param path: The file path to write to
    """
    lines = []
    for row in array:
        line = ''
        for num in row:
            line = line + f'{num},'

        lines.append(line[:-1] + '\n')

    f = open(path, 'w')
    f.writelines(lines)
    f.close()


def read_from_file(path: str):
    """
    Read a file made with write_to_file(_, path)
    :param path: The file path to read from
    :return: A 2D numpy array with the data in the file
    """
    # Read the file to raw data
    with open(path) as f:
        lines = f.readlines()
    # Read out the raw data
    out_list = [[float(num) for num in line.strip('\n').split(',')] for line in lines]
    # Return as numpy array
    return np.array(out_list)


def read_hawc2_aero_noise(path: str, scope: str = 'All'):
    """
    Read the sample HAWC2 aeroacoustic output file
    :param path: HAWC2 aeroacoustic output file
    :param scope: Selects the noise model result to load ('All', 'TI', 'TE', 'ST', 'TP')
    :return:    The observer position as numpy array of floats,
                pd.DataFrame of time series of hub position, wind speed and blade azimuths,
                List with pd.DataFrames of all PSD values at each timestep
    """
    if scope not in ('All', 'TI', 'TE', 'ST', 'TP'):
        raise ValueError(f'Invalid scope name: {scope}. Should be in {("All", "TI", "TE", "ST", "TP")}')

    scope_idxs = {'All': 0, 'TI': 1, 'TE': 2, 'ST': 3, 'TP': 4}

    # Read the data file into python
    with open(path) as f:
        lines = f.readlines()
    # Strip the lines of the newline
    lines = [line.strip('\n') for line in lines]

    n_freq = int(lines[4].split(' ')[-1])
    # Extract the observer position
    observer_pos = np.array([float(num) for num in lines[5].replace('  ', ' ').split(' ')[-3:]])
    # Prepare the time series DataFrame to be filled
    time_series_data = pd.DataFrame(columns=['t', 'hub_x', 'hub_y', 'hub_z', 'hub_u', 'psi_1', 'psi_2', 'psi_3', ], )
    time_series_data.set_index('t', inplace=True)

    # Create empty PSD dict
    psd = [{} for _ in range(4)]

    # Set the number of lines per time step
    n_lines_per_t = n_freq + 2
    # Loop over each time step
    for li, line in enumerate(lines[6::n_lines_per_t]):
        # Read the timestep info line and simplify it for processing
        info = line.replace('  ', ' ').split('# ')[1:]
        # Extract time
        t = float(info[0])
        # Extract rotor hub position
        hub_pos = [float(num) for num in info[1].split(' ')[2:-1]]
        # Extract hub wind speed
        hub_vel = float(info[2].split('  ')[1])
        # Extract azimuth of each blade
        blade_azim = [np.radians(float(num) + 90) for num in info[4].split(' ')[2:-1]]
        # Fill this info into the time series DataFrame
        time_series_data.loc[t] = [*hub_pos, hub_vel, *blade_azim]

        # Set the zero index of the PSD information
        psd_0_idx = 6 + li * n_lines_per_t + 1
        # Read the PSD block at this timestep
        psd_lines = np.array([[float(value) for value in psd_line.strip('  ').replace('  ', ' ').split(' ')]
                              for psd_line in lines[psd_0_idx:psd_0_idx + n_freq]])
        # Extract the frequency list for later
        f = psd_lines[:, 0]
        # Loop over the blades
        for n_blade in range(4):
            # Determine the blade's position in the psd block
            psd_idx = 5 * n_blade + 1
            # Put the correct psd into the dictionary of the blade
            psd[n_blade][t] = psd_lines[:, psd_idx + scope_idxs[scope]]

    # Create a DataFrame for each blade
    psd_out = [pd.DataFrame(psd[n_blade], index=f) for n_blade in range(4)]

    # Unwrap from the range [0,2pi] to avoid interpolation issues at the domain gap
    time_series_data.loc[:, 'psi_1'] = np.unwrap(time_series_data.loc[:, 'psi_1'])
    time_series_data.loc[:, 'psi_2'] = np.unwrap(time_series_data.loc[:, 'psi_2'])
    time_series_data.loc[:, 'psi_3'] = np.unwrap(time_series_data.loc[:, 'psi_3'])

    # Return all the extracted data ;|
    return observer_pos, time_series_data, psd_out

# TODO: hf.in_out implement alternative to librosa
# def wav_to_stft(path):
#     """
#     Read a WAV file and output the STFT and its attributes
#     :param path: path of the WAV file as a str
#     :return:    The sampling frequency as a float
#                 The timesteps of the STFT as a 1D np array
#                 The frequency bins of the STFT as a 1D np array
#                 The STFT of the left channel as a 2D np array
#                 The STFT of the right channel as a 2D np array
#     """
#     # Load the WAV fie with scipy io
#     freq, dat = spio.wavfile.read(path)
#     # Use librosa to determine the stft
#     fxx_0 = np.abs(librosa.stft(dat[:, 0] / np.max(np.abs(dat))))
#     fxx_1 = np.abs(librosa.stft(dat[:, 1] / np.max(np.abs(dat))))
#     # Determine the time and frequencies of the STFT
#     t_fxx = librosa.frames_to_time(np.arange(0, fxx_0.shape[1], dtype=int), sr=freq)
#     f_fxx = librosa.fft_frequencies(sr=freq)
#
#     return freq, t_fxx, f_fxx, fxx_0, fxx_1


# def wav_to_stft_mono(path):
#     """
#     Read a WAV file and output the STFT and its attributes
#     :param path: path of the WAV file as a str
#     :return:    The sampling frequency as a float
#                 The timesteps of the STFT as a 1D np array
#                 The frequency bins of the STFT as a 1D np array
#                 The STFT of the channel as a 2D np array
#     """
#     # Load the WAV file with scipy io
#     freq, dat = spio.wavfile.read(path)
#     # Use librosa to determine the stft
#     fxx_0 = np.abs(librosa.stft(dat / np.max(np.abs(dat))))
#     # Determine the time and frequencies of the STFT
#     t_fxx = librosa.frames_to_time(np.arange(0, fxx_0.shape[1], dtype=int), sr=freq)
#     f_fxx = librosa.fft_frequencies(sr=freq)
#
#     return freq, t_fxx, f_fxx, fxx_0


def read_ntk_data(path, calib_path):
    """
    Read the binary files from the 2015 NTK 500/41 turbine measurements
    :param path: path of the time series binary file
    :param calib_path: path of the file "calib.txt" with the calibration parameters
    :return: (number of sensors: int, sampling frequency: float, number of samples: int),
            data from the binary file in np.arrray with shape (n_samples, n_sensors)
    """
    # Read out the calibration file to get output parameters and calibration factors
    with open(calib_path) as f:
        lines = f.readlines()
        n_sensor, f_sampling, n_samples, _ = [int(float(num)) for num in lines[1].replace('d', 'e').split(' ')]
        calib = [float(num) for num in lines[3].strip('\n').replace('d', 'e').split(' ')[:-1]]

    # Read the binary files with the time series
    data = np.fromfile(path, dtype=np.float32)
    # Reshape the data and calibrate the values
    data = data.reshape(n_samples, n_sensor) / calib

    return (n_sensor, f_sampling, n_samples), data
