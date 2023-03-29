"""
Package with functions used in the main programme
"""
# ----------------------------------------------------------------------------------------------------------------------
# Import python modules for functions in this file
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Definition of constants
# ----------------------------------------------------------------------------------------------------------------------
c = 343  # Speed of Sound [m/s]
p_ref = 2e-5  # Sound reference pressure [Pa]
g = 9.80665  # Gravity [m/s2]
r_air = 287.  # Specific gas constant for air [J/kgK]
t_0 = 273.15  # Zero celsius in kelvin [K]
gamma_air = 1.4  # Specific heat ratio of air [-]


def limit_angle(angle):
    """
    Limit a radian angle between -pi and pi
    :param angle: input angle IN RADIANS
    :return: limited angle IN RADIANS
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


# ----------------------------------------------------------------------------------------------------------------------
# Import package functions
# ----------------------------------------------------------------------------------------------------------------------
from .coordinate_systems import Coordinates, Cartesian, Cylindrical, Spherical, HeadRelatedSpherical
from .in_out import wav_to_stft, read_from_file, read_hawc2_aero_noise, write_to_file, wav_to_stft_mono, read_ntk_data
from .data_structures import Heap
from .isa import Atmosphere
