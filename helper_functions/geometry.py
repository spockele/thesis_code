from . import Cartesian
import numpy as np
import warnings


"""
========================================================================================================================
===                                                                                                                  ===
=== Some functions and classes to handle geometry                                                                    ===
===                                                                                                                  ===
========================================================================================================================
"""
__all__ = ["uniform_spherical_grid", "PerpendicularPlane3D", ]


def uniform_spherical_grid(n_points: int):
    """
    Create a uniform grid in spherical coordinates where each point represents an equal fraction of the surface area
    of the unit sphere. Algorithm by Deserno, 2004.
    ------------------------------------------------------------------------------------------------
    !!! NOTE: output is not guaranteed to contain exact number of points input into the function !!!
    ------------------------------------------------------------------------------------------------
    :param n_points: desired number of points in the grid
    :return:    > list with n_count Cartesian points forming a unit sphere,
                > boolean indicating failure
                > inter-point distance
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
    # Create fail indicator if grid wants to make too many points
    fail = False
    # Loop in the polar direction
    try:
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

    except IndexError:
        warnings.warn("Uniform spherical grid attempted to generate too many points. No grid is output. fail = True",
                      RuntimeWarning)
        fail = True

    # Create Cartesian point list
    points = [Cartesian(np.cos(azimuth[i]) * np.sin(polar[i]),
                        np.sin(azimuth[i]) * np.sin(polar[i]),
                        np.cos(polar[i])) for i in range(n_count)]

    # Output the output arrays, shortened to actual number of generated points
    return points, fail, dist


class PerpendicularPlane3D:
    def __init__(self, p1: Cartesian, p2: Cartesian) -> None:
        """
        Plane defined by 2 points defining its normal vector.
        :param p1: first Cartesian point (m, m, m) of normal vector
        :param p2: second Cartesian point (m, m, m) of normal vector, also point defining position of plane
        """
        self.p1 = p1
        self.p2 = p2

    def distance_to_point(self, point: Cartesian) -> float:
        """
        Find the distance from this plane to the given point
        :param point: a point defined in Cartesian coordinates (m, m, m)
        :return: the distance (m)
        """
        normal_vector = self.p2 - self.p1
        normal_vector = normal_vector / normal_vector.len()

        diff_vector = point - self.p2

        return abs(sum((normal_vector * diff_vector).vec))
