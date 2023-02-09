import numpy as np


"""
"""


def limit_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class Cartesian:
    def __init__(self, x: float, y: float, z: float):
        """
        Class to store a point in cartesian coordinates
        :param x: Coordinate on the x-axis
        :param y: Coordinate on the y-axis
        :param z: Coordinate on the z-axis
        """
        self.vec = np.array((x, y, z))

    def __str__(self):
        return f'[{self.vec[0]}, {self.vec[1]}, {self.vec[2]}]'

    def __repr__(self):
        return f'<Cartesian: {str(self)}>'

    def __add__(self, other):
        if isinstance(other, type(self)):
            return Cartesian(*(self.vec + other.vec))
        elif isinstance(other, Spherical):
            return Cartesian(*(self.vec + other.to_cartesian().vec))
        else:
            raise TypeError('Cannot add this data type with a Cartesian coordinate.')

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return Cartesian(*(self.vec - other.vec))
        elif isinstance(other, Spherical):
            return Cartesian(*(self.vec - other.to_cartesian().vec))
        else:
            raise TypeError('Cannot add this data type with a Cartesian coordinate.')

    def to_spherical(self, origin):
        x, y, z = (self - origin).vec
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)

        return Spherical(r, theta, phi, origin)


class Spherical:
    def __init__(self, r: float, theta: float, phi: float, origin: Cartesian):
        """
        Class to store a point in spherical coordinates around "origin"
        :param r: Radial coordinate
        :param theta: Angle in xy-plane
        :param phi: Angle in the plane; perpendicular to xy-plane, and through origin and point.
                    Zero when parallel to z-axis.
        :param origin: Reference point around which these coordinates are determined.
        """
        self.vec = np.array((r, limit_angle(theta), limit_angle(phi)))
        self.origin = origin

    def __str__(self):
        return f'[{self.vec[0]}, {self.vec[1]}, {self.vec[2]}]'

    def __repr__(self):
        return f'<Spherical: {str(self)}, around {str(self.origin)}>'

    def __add__(self, other):
        """
        Addition of other to self, which is then referred to self.origin.
        """
        if isinstance(other, type(self)):
            cart1 = self.to_cartesian()
            cart2 = other.to_cartesian()
            cart = cart1 + cart2

            return cart.to_spherical(self.origin)

        elif isinstance(other, Cartesian):
            cart1 = self.to_cartesian()
            cart = cart1 + other

            return cart.to_spherical(self.origin)

        else:
            raise TypeError('Cannot add this data type with a Spherical coordinate.')

    def __sub__(self, other):
        """
        Subtraction of other from self, which is then referred to self.origin.
        """
        if isinstance(other, type(self)):
            cart1 = self.to_cartesian()
            cart2 = other.to_cartesian()
            cart = cart1 - cart2

            return cart.to_spherical(self.origin)

        elif isinstance(other, Cartesian):
            cart1 = self.to_cartesian()
            cart = cart1 - other

            return cart.to_spherical(self.origin)

        else:
            raise TypeError('Cannot add this data type with a Spherical coordinate.')

    def to_cartesian(self):
        r, theta, phi = self.vec
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)

        return Cartesian(x, y, z) + self.origin


if __name__ == '__main__':
    raise RuntimeError('Thou shalt not run this module on its own!')
