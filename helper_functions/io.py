import numpy as np
import pandas as pd
import scipy.io as spio
import librosa

from coordinate_systems import limit_angle


"""
"""


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


def read_hawc2_aero_noise(path: str):
    """
    Read the sample HAWC2 aeroacoustic output file
    :param path: HAWC2 aeroacoustic output file
    :return:    The observer position as numpy array of floats,
                pd.DataFrame of time series of hub position, wind speed and blade azimuths,
                Dict with pd.DataFrame of all PSD values at each timestep with time as keys
    """
    # Read the data file into python
    with open(path) as f:
        lines = f.readlines()
    # Strip the lines of the newline
    lines = [line.strip('\n') for line in lines]

    # Extract the observer position
    observer_pos = np.array([float(num) for num in lines[5].replace('  ', ' ').split(' ')[-3:]])
    # Prepare the time series DataFrame to be filled
    time_series_data = pd.DataFrame(columns=['t', 'hub_x', 'hub_y', 'hub_z', 'hub_u', 'psi_1', 'psi_2', 'psi_3', ], )
    time_series_data.set_index('t', inplace=True)

    # Create empty PSD dict
    psd = {}
    # Create PSD DataFrame for first time step
    psd_i = pd.DataFrame(columns=['f_c', 'PSD_All', 'PSD_TI', 'PSD_TE', 'PSD_ST', 'PSD_TP',
                                  'PSD_All_1', 'PSD_TI_1', 'PSD_TE_1', 'PSD_ST_1', 'PSD_TP_1',
                                  'PSD_All_2', 'PSD_TI_2', 'PSD_TE_2', 'PSD_ST_2', 'PSD_TP_2',
                                  'PSD_All_3', 'PSD_TI_3', 'PSD_TE_3', 'PSD_ST_3', 'PSD_TP_3'])
    psd_i.set_index('f_c', inplace=True)

    # Python is STOOOPID
    t = 0
    # Indicator to start the next timestep to True
    nxt = True
    # Loop over all the lines in the file
    for line in lines[6:]:
        # In case of having reached a new timestep
        if nxt:
            # Read the info line and simplify it for processing
            info = line.replace('  ', ' ').split('# ')[1:]
            # Extract all info at this timestep in this overly complex way :)
            t = float(info[0])
            hub_pos = [float(num) for num in info[1].split(' ')[3:-1]]
            hub_vel = float(info[2].split('  ')[1])
            blade_azim = [limit_angle(np.radians(float(num))) for num in info[4].split(' ')[2:-1]]

            # Fill this info into the DataFrame in an even better way :))
            time_series_data.loc[t] = [*hub_pos, hub_vel, *blade_azim]
            # Set the next timestep indicator to False since we will continue to extract the PSD stuff
            nxt = False

        # When an empty line is reached
        elif line == ' ':
            # Save PSD data for the timestep
            psd[t] = psd_i
            # We have reached the next timestep :)
            nxt = True
            # Create PSD DataFrame for next timestep
            psd_i = pd.DataFrame(columns=['f_c', 'PSD_All', 'PSD_TI', 'PSD_TE', 'PSD_ST', 'PSD_TP',
                                          'PSD_All_1', 'PSD_TI_1', 'PSD_TE_1', 'PSD_ST_1', 'PSD_TP_1',
                                          'PSD_All_2', 'PSD_TI_2', 'PSD_TE_2', 'PSD_ST_2', 'PSD_TP_2',
                                          'PSD_All_3', 'PSD_TI_3', 'PSD_TE_3', 'PSD_ST_3', 'PSD_TP_3'])
            psd_i.set_index('f_c', inplace=True)

        # All other lines are considered PSD data (THIS MIGHT BREAK WITH ACTUAL DATA...)
        else:
            # Split the numbers in this line from one another and fill them into the DataFrame of this timestep's PSD
            data = [float(num) for num in line.split('  ')[1:]]
            psd_i.loc[data[0]] = data[1:]

    # Return all the extracted data ;|
    return observer_pos, time_series_data, psd


def wav_to_stft(path):
    """
    Read a WAV file and output the STFT and its attributes
    :param path: path of the WAV file as a str
    :return:    The sampling frequency as a float
                The timesteps of the STFT as a 1D np array
                The frequency bins of the STFT as a 1D np array
                The STFT of the left channel as a 2D np array
                The STFT of the right channel as a 2D np array
    """
    # Load the WAV fie with scipy io
    freq, dat = spio.wavfile.read(path)
    # Use librosa to determine the stft
    fxx_0 = np.abs(librosa.stft(dat[:, 0] / np.max(np.abs(dat))))
    fxx_1 = np.abs(librosa.stft(dat[:, 1] / np.max(np.abs(dat))))
    # Determine the time and frequencies of the STFT
    t_fxx = librosa.frames_to_time(np.arange(0, fxx_0.shape[1], dtype=int), sr=freq)
    f_fxx = librosa.fft_frequencies(sr=freq)

    return freq, t_fxx, f_fxx, fxx_0, fxx_1


if __name__ == '__main__':
    raise RuntimeError("Do not run this file, it has no use.")
