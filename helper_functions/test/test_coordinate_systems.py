import unittest
import numpy as np

from .. import coordinate_systems as cs


class TestCoordinates(unittest.TestCase):
    """
    Tests the base Coordinates class
    """
    def setUp(self) -> None:
        self.coordinates = cs.Coordinates(np.array((1., 1., 1.)))

    def tearDown(self) -> None:
        del self.coordinates

    def test_operators(self):
        """
        Tests all operators: __init__ __getitem__ __setitem__ __str__
        """
        # Test __init__ and __getitem__
        self.assertEqual(self.coordinates[0], 1.)
        self.assertEqual(self.coordinates[1], 1.)
        self.assertEqual(self.coordinates[2], 1.)
        # Test __setitem__
        self.coordinates[0] = 2.
        self.assertEqual(self.coordinates[0], 2.)
        self.coordinates[1] = 2.
        self.assertEqual(self.coordinates[1], 2.)
        self.coordinates[2] = 2.
        self.assertEqual(self.coordinates[2], 2.)
        # Test __str__
        self.assertEqual(str(self.coordinates), '[2.0, 2.0, 2.0]')


class TestCartesian(unittest.TestCase):
    def setUp(self) -> None:
        self.cartesian_0 = cs.Cartesian(0., 0., 0.)
        self.cartesian_1 = cs.Cartesian(1., 1., 1.)
        self.cartesian_2 = cs.Cartesian(2., 2., 2.)

        self.non_cartesian = cs.NonCartesian(np.array((1., 1., 1.)), cs.Cartesian(0, 0, 0))

    def tearDown(self) -> None:
        del self.cartesian_0
        del self.cartesian_1
        del self.cartesian_2

    def test_operators(self):
        """
        Test the operators __init__ __repr__
        """
        # Test __init__
        self.assertEqual(self.cartesian_1[0], 1.)
        self.assertEqual(self.cartesian_1[1], 1.)
        self.assertEqual(self.cartesian_1[2], 1.)

        # Test __repr__
        self.assertEqual(self.cartesian_0.__repr__(), '<Cartesian: [0.0, 0.0, 0.0]>')

    def test_add_sub(self):
        """
        Test the operators: __add__ __sub__
        """
        # Test __sub__
        self.assertEqual((self.cartesian_2 - self.cartesian_1)[0], 1.)
        self.assertEqual((self.cartesian_2 - self.cartesian_1)[1], 1.)
        self.assertEqual((self.cartesian_2 - self.cartesian_1)[2], 1.)

        self.assertEqual((self.cartesian_2 - 1)[0], 1.)
        self.assertEqual((self.cartesian_2 - 1)[1], 1.)
        self.assertEqual((self.cartesian_2 - 1)[2], 1.)

        self.assertEqual((self.cartesian_2 - self.non_cartesian)[0], 1.)
        self.assertEqual((self.cartesian_2 - self.non_cartesian)[1], 1.)
        self.assertEqual((self.cartesian_2 - self.non_cartesian)[2], 1.)

        self.assertEqual((1- self.cartesian_2)[0], -1.)
        self.assertEqual((1- self.cartesian_2)[1], -1.)
        self.assertEqual((1- self.cartesian_2)[2], -1.)

        # Test __add__
        self.assertEqual((self.cartesian_2 + self.cartesian_1)[0], 3.)
        self.assertEqual((self.cartesian_2 + self.cartesian_1)[1], 3.)
        self.assertEqual((self.cartesian_2 + self.cartesian_1)[2], 3.)

        self.assertEqual((self.cartesian_2 + 1)[0], 3.)
        self.assertEqual((self.cartesian_2 + 1)[1], 3.)
        self.assertEqual((self.cartesian_2 + 1)[2], 3.)

        self.assertEqual((1 + self.cartesian_2)[0], 3.)
        self.assertEqual((1 + self.cartesian_2)[1], 3.)
        self.assertEqual((1 + self.cartesian_2)[2], 3.)

        self.assertEqual((self.cartesian_2 + self.non_cartesian)[0], 3.)
        self.assertEqual((self.cartesian_2 + self.non_cartesian)[1], 3.)
        self.assertEqual((self.cartesian_2 + self.non_cartesian)[2], 3.)

    def test_mul_truediv(self):
        """
        Test the operators: __mul__ __truediv__ __rmul__ __rtruediv__
        """
        # Test __mul__
        self.assertEqual((self.cartesian_2 * self.cartesian_1)[0], 2.)
        self.assertEqual((self.cartesian_2 * self.cartesian_1)[1], 2.)
        self.assertEqual((self.cartesian_2 * self.cartesian_1)[2], 2.)
        self.assertEqual((self.cartesian_2 * self.non_cartesian)[0], 2.)
        self.assertEqual((self.cartesian_2 * self.non_cartesian)[1], 2.)
        self.assertEqual((self.cartesian_2 * self.non_cartesian)[2], 2.)
        self.assertEqual((self.cartesian_2 * 2)[0], 4.)
        self.assertEqual((self.cartesian_2 * 2)[1], 4.)
        self.assertEqual((self.cartesian_2 * 2)[2], 4.)
        # Test __truediv__
        self.assertEqual((self.cartesian_2 / self.cartesian_1)[0], 2.)
        self.assertEqual((self.cartesian_2 / self.cartesian_1)[1], 2.)
        self.assertEqual((self.cartesian_2 / self.cartesian_1)[2], 2.)
        self.assertEqual((self.cartesian_2 / self.non_cartesian)[0], 2.)
        self.assertEqual((self.cartesian_2 / self.non_cartesian)[1], 2.)
        self.assertEqual((self.cartesian_2 / self.non_cartesian)[2], 2.)
        self.assertEqual((self.cartesian_2 / 2)[0], 1.)
        self.assertEqual((self.cartesian_2 / 2)[1], 1.)
        self.assertEqual((self.cartesian_2 / 2)[2], 1.)
        # Test __rmul__
        self.assertEqual((2 * self.cartesian_1)[0], 2.)
        self.assertEqual((2 * self.cartesian_1)[1], 2.)
        self.assertEqual((2 * self.cartesian_1)[2], 2.)
        # Test __rtruediv__
        self.assertEqual((2 / self.cartesian_1)[0], 2.)
        self.assertEqual((2 / self.cartesian_1)[1], 2.)
        self.assertEqual((2 / self.cartesian_1)[2], 2.)

    def test_len(self):
        """
        Test the vector length function
        """
        self.assertEqual(self.cartesian_1.len(), np.sqrt(3))
        self.assertEqual(self.cartesian_2.len(), np.sqrt(12))

    def test_to_spherical(self):
        """
        Test conversion to spherical coordinates
        """
        # Manual calculations
        r1 = 1.7321
        th1 = .7854
        ph1 = .6155
        # Tests the conversion for r = 1
        sph1 = self.cartesian_1.to_spherical(self.cartesian_0)
        self.assertEqual(round(sph1[0], 4), r1)
        self.assertEqual(round(sph1[1], 4), th1)
        self.assertEqual(round(sph1[2], 4), ph1)
        # Tests the conversion for r = 2
        sph2 = self.cartesian_2.to_spherical(self.cartesian_0)
        self.assertEqual(round(sph2[0], 3), round(2 * r1, 3))
        self.assertEqual(round(sph2[1], 4), th1)
        self.assertEqual(round(sph2[2], 4), ph1)

    def test_to_cylindrical(self):
        """
        Test conversion to cylindrical coordinates
        """
        # Manual calculations
        r1 = 1.4142
        ps1 = -.7854
        y1 = 1.
        # Tests the conversion
        cyl1 = self.cartesian_1.to_cylindrical(self.cartesian_0)
        self.assertEqual(round(cyl1[0], 4), r1)
        self.assertEqual(round(cyl1[1], 4), ps1)
        self.assertEqual(round(cyl1[2], 4), y1)
        # Tests the conversion
        cyl2 = self.cartesian_2.to_cylindrical(self.cartesian_0)
        self.assertEqual(round(cyl2[0], 3), round(2 * r1, 3))
        self.assertEqual(round(cyl2[1], 4), ps1)
        self.assertEqual(round(cyl2[2], 4), 2 * y1)

    def test_to_hr_spherical(self):
        """
        Test conversion to HR spherical coordinates
        """
        # Manual calculations
        r1 = 1.7321
        th1 = .7854
        ph1 = -.6155
        # Tests the conversion for r = 1
        sph1 = self.cartesian_1.to_hr_spherical(self.cartesian_0, 0)
        self.assertEqual(round(sph1[0], 4), r1)
        self.assertEqual(round(sph1[1], 4), th1)
        self.assertEqual(round(sph1[2], 4), ph1)
        # Tests the conversion for r = 2
        sph2 = self.cartesian_2.to_hr_spherical(self.cartesian_0, 0)
        self.assertEqual(round(sph2[0], 3), round(2 * r1, 3))
        self.assertEqual(round(sph2[1], 4), th1)
        self.assertEqual(round(sph2[2], 4), ph1)


class TestNonCartesian(unittest.TestCase):
    def setUp(self) -> None:
        pass



if __name__ == '__main__':
    unittest.main()
