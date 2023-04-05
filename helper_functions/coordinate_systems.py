import numpy as np
from . import limit_angle

"""
Definitions of the used coordinate systems
"""


class Coordinates:
    def __init__(self, vec):
        """
        Super class for all coordinate systems to universally type set them.
        :param vec: the coordinate vector as determined in the child class.
        """
        self.vec = vec

    def __str__(self):
        return f'[{self.vec[0]}, {self.vec[1]}, {self.vec[2]}]'


class Cartesian(Coordinates):
    def __init__(self, x: float, y: float, z: float):
        """
        Class to store a point in CARTESIAN coordinates.
        :param x: Coordinate on the x-axis.
        :param y: Coordinate on the y-axis.
        :param z: Coordinate on the z-axis.
        """
        super().__init__(np.array((x, y, z)))

    def __repr__(self):
        return f'<Cartesian: {str(self)}>'

    def __add__(self, other):
        if isinstance(other, type(self)):
            return Cartesian(*(self.vec + other.vec))
        elif issubclass(type(other), NonCartesian):
            return Cartesian(*(self.vec + other.to_cartesian().vec))
        else:
            raise TypeError('Cannot add this data type to a Cartesian coordinate.')

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return Cartesian(*(self.vec - other.vec))
        elif issubclass(type(other), NonCartesian):
            return Cartesian(*(self.vec - other.to_cartesian().vec))
        else:
            raise TypeError('Cannot subtract this data type from a Cartesian coordinate.')

    def to_spherical(self, origin):
        """
        Convert self to a Spherical coordinate.
        :param origin: origin point around which to set the spherical coordinates.
        """
        x, y, z = (self - origin).vec
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)

        return Spherical(r, theta, phi, origin)

    def to_cylindrical(self, origin):
        """
        Convert self to a Cylindrical coordinate.
        :param origin: origin point around which to set the cylindrical coordinates.
        """
        x, y, z = (self - origin).vec
        r = np.sqrt(x ** 2 + y ** 2)
        psi = np.arctan2(-z, x)

        return Cylindrical(r, psi, y, origin)

    def to_hr_spherical(self, origin, rotation):
        """
        Convert self to a head-related spherical coordinate.
        :param origin: origin point around which to set the spherical coordinates.
        :param rotation: Rotation of the head in the azimuth angle. IN RADIANS.
        """
        x, y, z = (self - origin).vec
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        azimuth = np.arctan2(x, y) - rotation
        polar = np.arctan2(np.sqrt(x ** 2 + y ** 2), -z)

        return HeadRelatedSpherical(r, azimuth, polar, origin, rotation)


class NonCartesian(Coordinates):
    def __init__(self, vec: np.array, origin: Cartesian):
        """
        Super class for the non-Cartesian coordinate systems containing all their conversions
        ONLY THE _to_cartesian() FUNCTION IS TO BE OVERWRITTEN, AS WELL AS THE __init__().
        :param vec: the coordinate vector as determined in the child class.
        :param origin: Reference point around which these coordinates are determined.
        """
        super().__init__(vec)
        self.origin = origin

    def __add__(self, other):
        """
        Addition of other to self, which is then referred to self.origin.
        """
        if issubclass(type(other), NonCartesian):
            cart1 = self.to_cartesian()
            cart2 = other.to_cartesian()
            cart = cart1 + cart2

            return cart.to_spherical(self.origin)

        elif isinstance(other, Cartesian):
            cart1 = self.to_cartesian()
            cart = cart1 + other

            return cart.to_spherical(self.origin)

        else:
            raise TypeError('Cannot add this data type to a Spherical coordinate.')

    def __sub__(self, other):
        """
        Subtraction of other from self, which is then referred to self.origin.
        """
        if issubclass(type(other), NonCartesian):
            cart1 = self.to_cartesian()
            cart2 = other.to_cartesian()
            cart = cart1 - cart2

            return cart.to_spherical(self.origin)

        elif isinstance(other, Cartesian):
            cart1 = self.to_cartesian()
            cart = cart1 - other

            return cart.to_spherical(self.origin)

        else:
            raise TypeError('Cannot subtract this data type from a Spherical coordinate.')

    def _to_cartesian(self):
        """
        TO BE OVERWRITTEN: Convert self to local Cartesian coordinate.
        :return: (x, y, z) as unpackable array of floats
        """
        return self.vec

    def to_cartesian(self):
        """
        Convert self to a Cartesian coordinate.
        """
        x, y, z = self._to_cartesian()
        return Cartesian(x, y, z) + self.origin

    def to_spherical(self, origin: Cartesian = None):
        """
        Convert self to a Spherical coordinate.
        :param origin: origin point around which to set the spherical coordinates. If None: around own origin point.
        """
        if origin is None:
            origin = self.origin

        return self.to_cartesian().to_spherical(origin)

    def to_cylindrical(self, origin: Cartesian = None):
        """
        Convert self to a Cylindrical coordinate.
        :param origin: origin point around which to set the cylindrical coordinates.  If None: around own origin point.
        """
        if origin is None:
            origin = self.origin

        return self.to_cartesian().to_cylindrical(origin)

    def to_hr_spherical(self, origin: Cartesian = None, rotation: float = None):
        """
        Convert self to a head-related spherical coordinate.
        :param origin: origin point around which to set the spherical coordinates.
        :param rotation: Rotation of the head in the azimuth angle. IN RADIANS.
        """
        if origin is None:
            origin = self.origin

        if rotation is None:
            rotation = 0.

        return self.to_cartesian().to_hr_spherical(origin, rotation)


class Spherical(NonCartesian):
    def __init__(self, r: float, theta: float, phi: float, origin: Cartesian):
        """
        Class to store a point in SPHERICAL coordinates around "origin"
        :param r: Radial coordinate.
        :param theta: Angle in xy-plane IN RADIANS.
        :param phi: Angle in the plane; perpendicular to xy-plane, and through origin and point.
                    Zero when parallel to z-axis. IN RADIANS.
        :param origin: Reference point around which these coordinates are determined.
        """
        super().__init__(np.array((r, limit_angle(theta), limit_angle(phi))), origin)

    def __repr__(self):
        return f'<Spherical: {str(self)}, around {str(self.origin)}>'

    def _to_cartesian(self):
        """
        Convert self to a local Cartesian coordinate.
        """
        r, theta, phi = self.vec
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.cos(phi)
        z = r * np.sin(phi)

        return x, y, z


class Cylindrical(NonCartesian):
    def __init__(self, r: float, psi: float, y: float, origin: Cartesian):
        """
        Class to store a point in Cylindrical coordinates around "origin"
        :param r: Radial coordinate.
        :param psi: Angle in xz-plane IN RADIANS.
        :param y: Coordinate in the y-axis.
        :param origin: Reference point around which these coordinates are determined.
        """
        super().__init__(np.array((r, limit_angle(psi), y)), origin)

    def __str__(self):
        return f'[{self.vec[0]}, {self.vec[1]}, {self.vec[2]}]'

    def _to_cartesian(self):
        """
        Convert self to a local Cartesian coordinate.
        """
        r, psi, y = self.vec
        x = r * np.cos(psi)
        z = - r * np.sin(psi)

        return x, y, z


class HeadRelatedSpherical(NonCartesian):
    def __init__(self, r, azimuth, polar, origin, rotation):
        """
        Class to store a point in Head-Related SPHERICAL coordinates around "origin"
        :param r: Radial coordinate.
        :param azimuth: Angle in xy-plane IN RADIANS.
        :param polar: Angle in the plane; perpendicular to xy-plane, and through origin and point.
                    Zero when parallel to z-axis. IN RADIANS.
        :param origin: Reference point around which these coordinates are determined.
        :param rotation: Rotation of the head in the azimuth angle. IN RADIANS.
        """
        super().__init__(np.array((r, limit_angle(azimuth), limit_angle(polar))), origin)
        self.rotation = limit_angle(rotation)

    def __repr__(self):
        return f'<HR-Spherical: {str(self)}, around {str(self.origin)} with rotation {self.rotation}>'

    def _to_cartesian(self):
        """
        Convert self to a Cartesian coordinate.
        """
        r, az, po = self.vec
        x = r * np.sin(az + self.rotation) * np.cos(po)
        y = r * np.cos(az + self.rotation) * np.cos(po)
        z = - r * np.sin(po)

        return x, y, z


if __name__ == '__main__':
    raise RuntimeError('Thou shalt not run this module on its own!')
