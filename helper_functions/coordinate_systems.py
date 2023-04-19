import numpy as np
from . import limit_angle

"""
========================================================================================================================
Definitions of the used coordinate systems
========================================================================================================================
> Coordinates -> common class for all these systems, mainly for type-setting purposes.

>> Cartesian -> direct child of Coordinates to store Cartesian coordinates.
    Contains conversions to all others

>> NonCartesian -> the common class for all systems that are not Cartesian.
    Contains the operator and conversion logic for all its child classes
    CHILD CLASSES SHOULD OVERWRITE INTERNAL _to_cartesian(self) FUNCTION TO GENERATE LOCAL X,Y,Z VALUES

>>> Spherical -> child of NonCartesian to define project spherical system

>>> Cylindrical -> child of NonCartesian to define project cylindrical system

>>> HeadRelatedSpherical -> child of NonCartesian to define Head-Related Spherical system
     Biggest difference to Spherical is different definition of polar and azimuth angles
     and a direction for the head to point in
"""


class Coordinates:
    def __init__(self, vec: np.array) -> None:
        """
        Super class for all coordinate systems to universally type-set them.
        :param vec: the coordinate vector as determined in the child class.
        """
        self.vec = vec

    def __getitem__(self, idx: int) -> float:
        # Get the coordinate at idx
        # Check that idx is in the vector
        if 0 > idx or idx > 2:
            raise IndexError(f'Coordinate vector only has 3 elements: 0 <= index <= 2, {idx}')
        # Return the coordinate
        return self.vec[idx]

    def __setitem__(self, idx: int, value: float) -> None:
        # Set the coordinate at idx to value
        # Check that idx is in the vector
        if 0 > idx or idx > 2:
            raise IndexError(f'Coordinate vector only has 3 elements: 0 <= key <= 2, {idx}')
        # Set the coordinate
        self.vec[idx] = value

    def __str__(self) -> str:
        # String representation of the coordinate vector
        return f'[{self.vec[0]}, {self.vec[1]}, {self.vec[2]}]'


class Cartesian(Coordinates):
    def __init__(self, x: float, y: float, z: float):
        """
        Class to store a point in CARTESIAN coordinates.
        :param x: Coordinate on the x-axis (m).
        :param y: Coordinate on the y-axis (m).
        :param z: Coordinate on the z-axis (m).
        """
        super().__init__(np.array((x, y, z)))

    def __repr__(self) -> str:
        # Nice class representation here :)
        return f'<Cartesian: {str(self)}>'

    def __add__(self, other):
        """
        Addition of other to self.
        """
        if isinstance(other, type(self)):
            return Cartesian(*(self.vec + other.vec))
        elif issubclass(type(other), NonCartesian):
            return Cartesian(*(self.vec + other.to_cartesian().vec))
        else:
            return Cartesian(*(self.vec + other))

    def __sub__(self, other):
        """
        Subtraction of other from self.
        """
        if isinstance(other, type(self)):
            return Cartesian(*(self.vec - other.vec))
        elif issubclass(type(other), NonCartesian):
            return Cartesian(*(self.vec - other.to_cartesian().vec))
        else:
            return Cartesian(*(self.vec - other))

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        """
        Subtraction of self from other.
        """
        if not (isinstance(other, type(self)) or issubclass(type(other), NonCartesian)):
            return Cartesian(*(other - self.vec))

    def __mul__(self, other):
        """
        Multiplication of other with self. Coordinates are multiplied elementwise.
        """
        if isinstance(other, Cartesian):
            return Cartesian(*(self.vec * other.vec))
        elif issubclass(type(other), NonCartesian):
            return Cartesian(*(self.vec * other.to_cartesian().vec))
        else:
            return Cartesian(*(self.vec * other))

    def __truediv__(self, other):
        """
        Division of self by other. Coordinates are divided elementwise.
        """
        if isinstance(other, Cartesian):
            return Cartesian(*(self.vec / other.vec))
        elif issubclass(type(other), NonCartesian):
            return Cartesian(*(self.vec / other.to_cartesian().vec))
        else:
            return Cartesian(*(self.vec / other))

    def __rmul__(self, other):
        # Multiplication is commutative when not defined above.
        return self.__mul__(other)

    def __rtruediv__(self, other):
        """
        Division of other by self. Coordinates are divided elementwise.
        """
        if isinstance(other, Cartesian):
            return Cartesian(*(other.vec / self.vec))
        elif issubclass(type(other), NonCartesian):
            return Cartesian(*(other.to_cartesian().vec / self.vec))
        else:
            return Cartesian(*(other / self.vec))

    def len(self) -> float:
        """
        Determine the vector length of this coordinate
        :return: the vector length (m)
        """
        return np.sqrt(np.sum(self.vec ** 2))

    def to_spherical(self, origin) -> Coordinates:
        """
        Convert self to a Spherical coordinate.
        :param origin: origin point around which to set the spherical coordinates.
        """
        x, y, z = (self - origin).vec
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))

        return Spherical(r, theta, phi, origin)

    def to_cylindrical(self, origin) -> Coordinates:
        """
        Convert self to a Cylindrical coordinate.
        :param origin: origin point around which to set the cylindrical coordinates.
        """
        x, y, z = (self - origin).vec
        r = np.sqrt(x ** 2 + y ** 2)
        psi = np.arctan2(-z, x)

        return Cylindrical(r, psi, y, origin)

    def to_hr_spherical(self, origin, rotation: float) -> Coordinates:
        """
        Convert self to a head-related spherical coordinate.
        :param origin: origin point around which to set the spherical coordinates.
        :param rotation: Rotation of the head in the azimuth angle. IN RADIANS.
        """
        x, y, z = (self - origin).vec
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        azimuth = np.arctan2(x, y) - rotation
        polar = np.arctan2(-z, np.sqrt(x ** 2 + y ** 2))

        return HeadRelatedSpherical(r, azimuth, polar, origin, rotation)


class NonCartesian(Coordinates):
    def __init__(self, vec: np.array, origin: Cartesian) -> None:
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

        else:
            cart1 = self.to_cartesian()
            cart = cart1 + other

            return cart.to_spherical(self.origin)

    def __sub__(self, other):
        """
        Subtraction of other from self, which is then referred to self.origin.
        """
        if issubclass(type(other), NonCartesian):
            cart1 = self.to_cartesian()
            cart2 = other.to_cartesian()
            cart = cart1 - cart2

            return cart.to_spherical(self.origin)

        else:
            cart1 = self.to_cartesian()
            cart = cart1 - other

            return cart.to_spherical(self.origin)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if not issubclass(type(other), NonCartesian):
            cart1 = self.to_cartesian()
            cart = other - cart1

            return cart.to_spherical(self.origin)

    def __mul__(self, other):
        """
        Multiplication of other with self. Coordinates are multiplied elementwise in Cartesian form.
        """
        if issubclass(type(other), NonCartesian):
            cart1 = self.to_cartesian()
            cart2 = other.to_cartesian()
            cart = cart1 * cart2

            return cart.to_spherical(self.origin)

        else:
            cart1 = self.to_cartesian()
            cart = cart1 * other

            return cart.to_spherical(self.origin)

    def __truediv__(self, other):
        """
        Division of self by other. Coordinates are Divided elementwise in Cartesian form.
        """
        if issubclass(type(other), NonCartesian):
            cart1 = self.to_cartesian()
            cart2 = other.to_cartesian()
            cart = cart1 / cart2

            return cart.to_spherical(self.origin)

        else:
            cart1 = self.to_cartesian()
            cart = cart1 / other

            return cart.to_spherical(self.origin)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        """
        Division of other by self. Coordinates are Divided elementwise in Cartesian form.
        """
        if not issubclass(type(other), NonCartesian):
            cart1 = self.to_cartesian()
            cart = other / cart1

            return cart.to_spherical(self.origin)

    def len(self) -> float:
        """
        Determine the CARTESIAN vector length of this coordinate
        :return: the vector length (m)
        """
        return self.to_cartesian().len()

    def _to_cartesian(self) -> np.array:
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
        :param origin: Origin point around which to set the cylindrical coordinates.  If None: around own origin point.
        :param rotation: Rotation of the head in the azimuth angle. IN RADIANS. If None: no head rotation is added
        """
        if origin is None:
            origin = self.origin

        if rotation is None:
            rotation = 0.

        return self.to_cartesian().to_hr_spherical(origin, rotation)


class Spherical(NonCartesian):
    def __init__(self, r: float, theta: float, phi: float, origin: Cartesian) -> None:
        """
        Class to store a point in SPHERICAL coordinates around "origin"
        :param r: Radial coordinate.
        :param theta: Angle in xy-plane IN RADIANS.
        :param phi: Angle in the plane; perpendicular to xy-plane, and through origin and point.
                    Zero when parallel to z-axis. IN RADIANS.
        :param origin: Reference point around which these coordinates are determined.
        """
        super().__init__(np.array((r, limit_angle(theta), limit_angle(phi))), origin)

    def __repr__(self) -> str:
        # Nice class representation here :)
        return f'<Spherical: {str(self)}, around {str(self.origin)}>'

    def _to_cartesian(self) -> (float, float, float):
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

    def __repr__(self):
        # Nice class representation here :)
        return f'<Cylindrical: {str(self)}, around {str(self.origin)}>'

    def _to_cartesian(self) -> (float, float, float):
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
        # Nice class representation here :)
        return f'<HR-Spherical: {str(self)}, around {str(self.origin)} with rotation {self.rotation}>'

    def _to_cartesian(self) -> (float, float, float):
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
