from . import Cartesian


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

        return sum((normal_vector * diff_vector).vec)


if __name__ == '__main__':
    raise RuntimeError('Thou shalt not run this module on its own!')
