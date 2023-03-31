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
    :param angle: input angle(s) IN RADIANS
    :return: limited angle(s) IN RADIANS
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def a_weighting(f):
    """
    A-weighting function Delta L_A
    :param f: frequency array
    :return: Array with corresponding values of Delta L_A
    """
    return -145.528 + 98.262 * np.log10(f) - 19.509 * np.log10(f) ** 2 + 0.975 * np.log10(f) ** 3


def uniform_spherical_grid(n_points: int):
    """
    Create a uniform grid in spherical coordinates where each point represents an equal fraction of the surface area
    of the unit sphere. Algorithm by Deserno, 2004.
    ------------------------------------------------------------------------------------------------
    !!! NOTE: output is not guaranteed to contain exact number of points input into the function !!!
    ------------------------------------------------------------------------------------------------
    :param n_points: desired number of points in the grid
    :return: the polar and azimuth angles in numpy arrays of length n_count
    """
    # Area represented per point of the grid
    point_area = 4 * np.pi / n_points
    # Length scale associated with this area
    dist = np.sqrt(point_area)
    # number of polar coordinates
    m_pol = round(np.pi / dist)
    # Angular distance between points in polar direction
    d_pol = np.pi / m_pol
    # Angular distance between points in azimuth direction
    d_azi = point_area / d_pol

    # Create counter and arrays for output
    n_count = 0
    polar, azimuth = np.empty((2, n_points))
    # Loop in the polar direction
    for mi in range(m_pol):
        # Determine polar angle
        pol = np.pi * (mi + .5) / m_pol
        # Determine number of azimuthal coordinates
        m_azi = round(2 * np.pi * np.sin(pol) / d_azi)
        # Loop over azimuth angles
        for ni in range(m_azi):
            # Add polar angle to output array
            polar[n_count] = pol
            # Determine azimuth angle and add to output array
            azimuth[n_count] = 2 * np.pi * ni / m_azi
            # Up the counter by 1
            n_count += 1

    # Output the output arrays, shortened to actual number of generated points
    return polar[:n_count + 1], azimuth[:n_count + 1]


# ----------------------------------------------------------------------------------------------------------------------
# Import package functions
# ----------------------------------------------------------------------------------------------------------------------
from .coordinate_systems import Coordinates, Cartesian, Cylindrical, Spherical, HeadRelatedSpherical
from .in_out import wav_to_stft, read_from_file, read_hawc2_aero_noise, write_to_file, wav_to_stft_mono, read_ntk_data
from .data_structures import Heap
from .isa import Atmosphere
from .hrtf import MitHrtf
