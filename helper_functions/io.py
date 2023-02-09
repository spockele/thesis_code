import numpy as np
import pandas as pd

from .coordinate_systems import limit_angle


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
                Dict with pd.DataFrame of all psd values at each timestep with time as keys
    """
    with open(path) as f:
        lines = f.readlines()

    lines = [line.strip('\n') for line in lines]

    observer_pos = np.array([float(num) for num in lines[5].replace('  ', ' ').split(' ')[-3:]])

    time_series_data = pd.DataFrame(columns=['t', 'hub_x', 'hub_y', 'hub_z', 'hub_u', 'psi_1', 'psi_2', 'psi_3', ], )
    time_series_data.set_index('t', inplace=True)

    psd = {}
    psd_i = pd.DataFrame(columns=['f_c', 'PSD_All', 'PSD_TI', 'PSD_TE', 'PSD_ST', 'PSD_TP',
                                  'PSD_All_1', 'PSD_TI_1', 'PSD_TE_1', 'PSD_ST_1', 'PSD_TP_1',
                                  'PSD_All_2', 'PSD_TI_2', 'PSD_TE_2', 'PSD_ST_2', 'PSD_TP_2',
                                  'PSD_All_3', 'PSD_TI_3', 'PSD_TE_3', 'PSD_ST_3', 'PSD_TP_3'])
    psd_i.set_index('f_c', inplace=True)

    t = 0
    nxt = True
    for line in lines[6:]:
        if nxt:
            info = line.replace('  ', ' ').split('# ')[1:]
            t = float(info[0])
            hub_pos = [float(num) for num in info[1].split(' ')[3:-1]]
            hub_vel = float(info[2].split('  ')[1])

            blade_azim = [limit_angle(np.radians(float(num))) for num in info[4].split(' ')[2:-1]]

            time_series_data.loc[t] = [*hub_pos, hub_vel, *blade_azim]
            nxt = False

        elif line == ' ':
            psd[t] = psd_i
            nxt = True

        else:
            data = [float(num) for num in line.split('  ')[1:]]
            psd_i.loc[data[0]] = data[1:]

    return observer_pos, time_series_data, psd


if __name__ == '__main__':
    raise RuntimeError("Do not run this file, it has no use.")
