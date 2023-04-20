import unittest
import numpy as np

import helper_functions as hf


class TestCoordinates(unittest.TestCase):
    """
    Tests the base Coordinates class
    """
    def setUp(self) -> None:
        self.coordinates = hf.Coordinates(np.array((1., 1., 1.)))

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
        self.assertRaises(IndexError, self.coordinates.__getitem__, 3,)
        self.assertRaises(IndexError, self.coordinates.__getitem__, -4,)
        # Test __setitem__
        self.coordinates[0] = 2.
        self.assertEqual(self.coordinates[0], 2.)
        self.coordinates[1] = 2.
        self.assertEqual(self.coordinates[1], 2.)
        self.coordinates[2] = 2.
        self.assertEqual(self.coordinates[2], 2.)
        self.assertRaises(IndexError, self.coordinates.__setitem__, 3, 1.)
        self.assertRaises(IndexError, self.coordinates.__setitem__, -4, 1.)
        # Test __str__
        self.assertEqual(str(self.coordinates), '[2.0, 2.0, 2.0]')


class TestCartesian(unittest.TestCase):
    def setUp(self) -> None:
        self.cartesian_0 = hf.Cartesian(0., 0., 0.)
        self.cartesian_1 = hf.Cartesian(1., 1., 1.)
        self.cartesian_2 = hf.Cartesian(2., 2., 2.)

        self.non_cartesian = hf.NonCartesian(np.array((1., 1., 1.)), hf.Cartesian(0, 0, 0))

    def tearDown(self) -> None:
        del self.cartesian_0
        del self.cartesian_1
        del self.cartesian_2
        del self.non_cartesian

    def test_operators(self):
        """
        Test the operators __init__ __repr__ __neq__
        """
        # Test __init__
        self.assertEqual(self.cartesian_1[0], 1.)
        self.assertEqual(self.cartesian_1[1], 1.)
        self.assertEqual(self.cartesian_1[2], 1.)

        # Test __neg__
        neg = -self.cartesian_1
        self.assertEqual(neg[0], -1.)
        self.assertEqual(neg[1], -1.)
        self.assertEqual(neg[2], -1.)

        # Test __repr__
        self.assertEqual(self.cartesian_0.__repr__(), '<Cartesian: [0.0, 0.0, 0.0]>')

    def test_equality(self):
        """
        Test the operators: __eq__ __ne__
        """
        self.assertTrue(self.cartesian_0 == self.cartesian_0)
        self.assertFalse(self.cartesian_0 == self.cartesian_1)
        self.assertTrue(self.cartesian_0 != self.cartesian_1)
        self.assertTrue(self.cartesian_1 == self.non_cartesian)
        self.assertFalse(self.cartesian_0 == 1.)

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
        # Test __radd__
        self.assertEqual((1 - self.cartesian_2)[0], -1.)
        self.assertEqual((1 - self.cartesian_2)[1], -1.)
        self.assertEqual((1 - self.cartesian_2)[2], -1.)

        # Test __add__
        self.assertEqual((self.cartesian_2 + self.cartesian_1)[0], 3.)
        self.assertEqual((self.cartesian_2 + self.cartesian_1)[1], 3.)
        self.assertEqual((self.cartesian_2 + self.cartesian_1)[2], 3.)
        self.assertEqual((self.cartesian_2 + 1)[0], 3.)
        self.assertEqual((self.cartesian_2 + 1)[1], 3.)
        self.assertEqual((self.cartesian_2 + 1)[2], 3.)
        self.assertEqual((self.cartesian_2 + self.non_cartesian)[0], 3.)
        self.assertEqual((self.cartesian_2 + self.non_cartesian)[1], 3.)
        self.assertEqual((self.cartesian_2 + self.non_cartesian)[2], 3.)
        # Test __rsub__
        self.assertEqual((1 + self.cartesian_2)[0], 3.)
        self.assertEqual((1 + self.cartesian_2)[1], 3.)
        self.assertEqual((1 + self.cartesian_2)[2], 3.)

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
        self.non_cartesian_0 = hf.NonCartesian(np.array((0., 0., 0.)), hf.Cartesian(0, 0, 0))
        self.non_cartesian_1 = hf.NonCartesian(np.array((1., 1., 1.)), hf.Cartesian(0, 0, 0))
        self.non_cartesian_2 = hf.NonCartesian(np.array((2., 2., 2.)), hf.Cartesian(0, 0, 0))

        self.cartesian = hf.Cartesian(1., 1., 1.)
        self.cartesian_0 = hf.Cartesian(0., 0., 0.)

    def tearDown(self) -> None:
        del self.non_cartesian_0
        del self.non_cartesian_1
        del self.non_cartesian_2
        del self.cartesian
        del self.cartesian_0

    def test_operators(self):
        """
        Test the operators __init__ __neg__
        """
        # Test __init__
        self.assertEqual(self.non_cartesian_1[0], 1.)
        self.assertEqual(self.non_cartesian_1[1], 1.)
        self.assertEqual(self.non_cartesian_1[2], 1.)

        # Test __neg__
        neg = -self.non_cartesian_1
        self.assertEqual(neg[0], -1.)
        self.assertEqual(neg[1], -1.)
        self.assertEqual(neg[2], -1.)

    def test_equality(self):
        """
        Test the operators: __eq__ __ne__
        """
        self.assertTrue(self.non_cartesian_0 == self.non_cartesian_0)
        self.assertFalse(self.non_cartesian_0 == self.non_cartesian_1)
        self.assertTrue(self.non_cartesian_0 != self.non_cartesian_1)
        self.assertTrue(self.non_cartesian_1 == self.cartesian)
        self.assertFalse(self.non_cartesian_0 == 1.)

    def test_origin(self):
        """
        Test the assignment of the origin for the transform to Cartesian
        """
        self.assertEqual(hf.NonCartesian(np.array((0, 0, 0)), self.cartesian).to_cartesian(), self.cartesian)

    def test_add_sub(self):
        """
        Test the operators: __add__ __sub__
        """
        # Test __sub__
        self.assertEqual(round((self.non_cartesian_2 - self.non_cartesian_1)[0]), 1.)
        self.assertEqual(round((self.non_cartesian_2 - self.non_cartesian_1)[1]), 1.)
        self.assertEqual(round((self.non_cartesian_2 - self.non_cartesian_1)[2]), 1.)
        self.assertEqual(round((self.non_cartesian_2 - 1)[0]), 1.)
        self.assertEqual(round((self.non_cartesian_2 - 1)[1]), 1.)
        self.assertEqual(round((self.non_cartesian_2 - 1)[2]), 1.)
        self.assertEqual(round((self.non_cartesian_2 - self.cartesian)[0]), 1.)
        self.assertEqual(round((self.non_cartesian_2 - self.cartesian)[1]), 1.)
        self.assertEqual(round((self.non_cartesian_2 - self.cartesian)[2]), 1.)
        # Test __radd__
        self.assertEqual(round((1 - self.non_cartesian_2).to_cartesian()[0]), -1.)
        self.assertEqual(round((1 - self.non_cartesian_2).to_cartesian()[1]), -1.)
        self.assertEqual(round((1 - self.non_cartesian_2).to_cartesian()[2]), -1.)

        # Test __add__
        self.assertEqual(round((self.non_cartesian_2 + self.non_cartesian_1)[0]), 3.)
        self.assertEqual(round((self.non_cartesian_2 + self.non_cartesian_1)[1]), 3.)
        self.assertEqual(round((self.non_cartesian_2 + self.non_cartesian_1)[2]), 3.)
        self.assertEqual(round((self.non_cartesian_2 + 1)[0]), 3.)
        self.assertEqual(round((self.non_cartesian_2 + 1)[1]), 3.)
        self.assertEqual(round((self.non_cartesian_2 + 1)[2]), 3.)
        self.assertEqual(round((self.non_cartesian_2 + self.cartesian)[0]), 3.)
        self.assertEqual(round((self.non_cartesian_2 + self.cartesian)[1]), 3.)
        self.assertEqual(round((self.non_cartesian_2 + self.cartesian)[2]), 3.)
        # Test __rsub__
        self.assertEqual(round((1 + self.non_cartesian_2)[0]), 3.)
        self.assertEqual(round((1 + self.non_cartesian_2)[1]), 3.)
        self.assertEqual(round((1 + self.non_cartesian_2)[2]), 3.)

    def test_mul_truediv(self):
        """
        Test the operators: __mul__ __truediv__ __rmul__ __rtruediv__
        """
        # Test __mul__
        self.assertEqual(round((self.non_cartesian_2 * self.non_cartesian_1)[0]), 2.)
        self.assertEqual(round((self.non_cartesian_2 * self.non_cartesian_1)[1]), 2.)
        self.assertEqual(round((self.non_cartesian_2 * self.non_cartesian_1)[2]), 2.)
        self.assertEqual(round((self.non_cartesian_2 * self.cartesian)[0]), 2.)
        self.assertEqual(round((self.non_cartesian_2 * self.cartesian)[1]), 2.)
        self.assertEqual(round((self.non_cartesian_2 * self.cartesian)[2]), 2.)
        self.assertEqual(round((self.non_cartesian_2 * 2)[0]), 4.)
        self.assertEqual(round((self.non_cartesian_2 * 2)[1]), 4.)
        self.assertEqual(round((self.non_cartesian_2 * 2)[2]), 4.)
        # Test __truediv__
        self.assertEqual(round((self.non_cartesian_2 / self.non_cartesian_1)[0]), 2.)
        self.assertEqual(round((self.non_cartesian_2 / self.non_cartesian_1)[1]), 2.)
        self.assertEqual(round((self.non_cartesian_2 / self.non_cartesian_1)[2]), 2.)
        self.assertEqual(round((self.non_cartesian_2 / self.cartesian)[0]), 2.)
        self.assertEqual(round((self.non_cartesian_2 / self.cartesian)[1]), 2.)
        self.assertEqual(round((self.non_cartesian_2 / self.cartesian)[2]), 2.)
        self.assertEqual(round((self.non_cartesian_2 / 2)[0]), 1.)
        self.assertEqual(round((self.non_cartesian_2 / 2)[1]), 1.)
        self.assertEqual(round((self.non_cartesian_2 / 2)[2]), 1.)
        # Test __rmul__
        self.assertEqual(round((2 * self.non_cartesian_1)[0]), 2.)
        self.assertEqual(round((2 * self.non_cartesian_1)[1]), 2.)
        self.assertEqual(round((2 * self.non_cartesian_1)[2]), 2.)
        # Test __rtruediv__
        self.assertEqual(round((2 / self.non_cartesian_1)[0]), 2.)
        self.assertEqual(round((2 / self.non_cartesian_1)[1]), 2.)
        self.assertEqual(round((2 / self.non_cartesian_1)[2]), 2.)

    def test_len(self):
        """
        Test the vector length function
        """
        self.assertEqual(self.non_cartesian_1.len(), np.sqrt(3))
        self.assertEqual(self.non_cartesian_2.len(), np.sqrt(12))

    def test_to_spherical(self):
        """
        Test conversion to spherical coordinates
        """
        # Manual calculations
        r1 = 1.7321
        th1 = .7854
        ph1 = .6155
        # Tests the conversion for r = 1
        sph1 = self.non_cartesian_1.to_spherical(self.cartesian_0)
        self.assertEqual(round(sph1[0], 4), r1)
        self.assertEqual(round(sph1[1], 4), th1)
        self.assertEqual(round(sph1[2], 4), ph1)
        # Tests the conversion for r = 2
        sph2 = self.non_cartesian_2.to_spherical(self.cartesian_0)
        self.assertEqual(round(sph2[0], 3), round(2 * r1, 3))
        self.assertEqual(round(sph2[1], 4), th1)
        self.assertEqual(round(sph2[2], 4), ph1)

        # Tests the conversion for r = 1 w/o origin
        sph1 = self.non_cartesian_1.to_spherical()
        self.assertEqual(round(sph1[0], 4), r1)
        self.assertEqual(round(sph1[1], 4), th1)
        self.assertEqual(round(sph1[2], 4), ph1)
        # Tests the conversion for r = 2 w/o origin
        sph2 = self.non_cartesian_2.to_spherical()
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
        # Tests the conversion for r = 1
        cyl1 = self.non_cartesian_1.to_cylindrical(self.cartesian_0)
        self.assertEqual(round(cyl1[0], 4), r1)
        self.assertEqual(round(cyl1[1], 4), ps1)
        self.assertEqual(round(cyl1[2], 4), y1)
        # Tests the conversion for r = 2
        cyl2 = self.non_cartesian_2.to_cylindrical(self.cartesian_0)
        self.assertEqual(round(cyl2[0], 3), round(2 * r1, 3))
        self.assertEqual(round(cyl2[1], 4), ps1)
        self.assertEqual(round(cyl2[2], 4), 2 * y1)

        # Tests the conversion for r = 1 w/o origin
        cyl1 = self.non_cartesian_1.to_cylindrical()
        self.assertEqual(round(cyl1[0], 4), r1)
        self.assertEqual(round(cyl1[1], 4), ps1)
        self.assertEqual(round(cyl1[2], 4), y1)
        # Tests the conversion for r = 2 w/o origin
        cyl2 = self.non_cartesian_2.to_cylindrical()
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
        # Tests the conversion for r = 1 w/rotation
        sph1 = self.non_cartesian_1.to_hr_spherical(self.cartesian_0, 0)
        self.assertEqual(round(sph1[0], 4), r1)
        self.assertEqual(round(sph1[1], 4), th1)
        self.assertEqual(round(sph1[2], 4), ph1)
        # Tests the conversion for r = 2 w/rotation
        sph2 = self.non_cartesian_2.to_hr_spherical(self.cartesian_0, 0)
        self.assertEqual(round(sph2[0], 3), round(2 * r1, 3))
        self.assertEqual(round(sph2[1], 4), th1)
        self.assertEqual(round(sph2[2], 4), ph1)

        # Tests the conversion for r = 1
        sph1 = self.non_cartesian_1.to_hr_spherical(self.cartesian_0)
        self.assertEqual(round(sph1[0], 4), r1)
        self.assertEqual(round(sph1[1], 4), th1)
        self.assertEqual(round(sph1[2], 4), ph1)
        # Tests the conversion for r = 2
        sph2 = self.non_cartesian_2.to_hr_spherical(self.cartesian_0)
        self.assertEqual(round(sph2[0], 3), round(2 * r1, 3))
        self.assertEqual(round(sph2[1], 4), th1)
        self.assertEqual(round(sph2[2], 4), ph1)

        # Tests the conversion for r = 1 w/o origin
        sph1 = self.non_cartesian_1.to_hr_spherical()
        self.assertEqual(round(sph1[0], 4), r1)
        self.assertEqual(round(sph1[1], 4), th1)
        self.assertEqual(round(sph1[2], 4), ph1)
        # Tests the conversion for r = 2 w/o origin
        sph2 = self.non_cartesian_2.to_hr_spherical()
        self.assertEqual(round(sph2[0], 3), round(2 * r1, 3))
        self.assertEqual(round(sph2[1], 4), th1)
        self.assertEqual(round(sph2[2], 4), ph1)


class TestSpherical(unittest.TestCase):
    def setUp(self) -> None:
        self.spherical_o = hf.Spherical(0., 0., 0., hf.Cartesian(1., 1., 1.))
        self.spherical_0 = hf.Spherical(0., 0., 0., hf.Cartesian(0., 0., 0.))
        self.spherical_1 = hf.Spherical(1., 1., 1., hf.Cartesian(0., 0., 0.))
        self.spherical = hf.Spherical(1., 0., 0., hf.Cartesian(0., 0., 0.))
        self.cartesian = hf.Cartesian(1., 0., 0.)
        self.cartesian_1 = hf.Cartesian(1., 1., 1.)

    def tearDown(self) -> None:
        del self.spherical_0
        del self.spherical_1

    def test_operators(self):
        """
        Test the operators __init__ __repr__
        """
        # Test __init__
        self.assertEqual(self.spherical_1[0], 1.)
        self.assertEqual(self.spherical_1[1], 1.)
        self.assertEqual(self.spherical_1[2], 1.)
        self.assertEqual(self.spherical_1.origin, hf.Cartesian(0., 0., 0.))

        # Test origin point
        self.assertEqual(self.spherical_o.to_cartesian(), hf.Cartesian(1., 1., 1.))

        # Test __repr__
        self.assertEqual(self.spherical_0.__repr__(), '<Spherical: [0.0, 0.0, 0.0], around [0.0, 0.0, 0.0]>')

    def test_to_self(self):
        """
        Test the working of spherical _to_self
        """
        self.assertEqual(self.spherical._to_self(self.cartesian), self.spherical)
        self.assertEqual(self.spherical_o._to_self(self.cartesian_1), self.spherical_o)

    def test_conversion(self):
        """
        Test conversion to Cartesian by double conversion
        """
        self.assertEqual(self.spherical_1.to_cartesian().to_spherical(self.spherical_1.origin), self.spherical_1)
        self.assertEqual(self.spherical_o.to_cartesian().to_spherical(self.spherical_o.origin), self.spherical_o)


class TestCylindrical(unittest.TestCase):
    def setUp(self) -> None:
        self.spherical_o = hf.Cylindrical(0., 0., 0., hf.Cartesian(1., 1., 1.))
        self.spherical_0 = hf.Cylindrical(0., 0., 0., hf.Cartesian(0., 0., 0.))
        self.spherical_1 = hf.Cylindrical(1., 1., 1., hf.Cartesian(0., 0., 0.))
        self.spherical = hf.Cylindrical(1., 0., 0., hf.Cartesian(0., 0., 0.))
        self.cartesian = hf.Cartesian(1., 0., 0.)
        self.cartesian_1 = hf.Cartesian(1., 1., 1.)

    def tearDown(self) -> None:
        del self.spherical_0
        del self.spherical_1

    def test_operators(self):
        """
        Test the operators __init__ __repr__
        """
        # Test __init__
        self.assertEqual(self.spherical_1[0], 1.)
        self.assertEqual(self.spherical_1[1], 1.)
        self.assertEqual(self.spherical_1[2], 1.)
        self.assertEqual(self.spherical_1.origin, hf.Cartesian(0., 0., 0.))

        # Test origin point
        self.assertEqual(self.spherical_o.to_cartesian(), hf.Cartesian(1., 1., 1.))

        # Test __repr__
        self.assertEqual(self.spherical_0.__repr__(), '<Cylindrical: [0.0, 0.0, 0.0], around [0.0, 0.0, 0.0]>')

    def test_to_self(self):
        """
        Test the working of spherical _to_self
        """
        self.assertEqual(self.spherical._to_self(self.cartesian), self.spherical)
        self.assertEqual(self.spherical_o._to_self(self.cartesian_1), self.spherical_o)

    def test_conversion(self):
        """
        Test conversion to Cartesian by double conversion
        """
        self.assertEqual(self.spherical_1.to_cartesian().to_cylindrical(self.spherical_1.origin), self.spherical_1)
        self.assertEqual(self.spherical_o.to_cartesian().to_cylindrical(self.spherical_o.origin), self.spherical_o)


class TestHeadRelatedSpherical(unittest.TestCase):
    def setUp(self) -> None:
        self.spherical_r = hf.HeadRelatedSpherical(1., 1., 1., hf.Cartesian(0., 0., 0.), 1.)
        self.spherical_o = hf.HeadRelatedSpherical(0., 0., 0., hf.Cartesian(1., 1., 1.), 0.)
        self.spherical_0 = hf.HeadRelatedSpherical(0., 0., 0., hf.Cartesian(0., 0., 0.), 0.)
        self.spherical_1 = hf.HeadRelatedSpherical(1., 1., 1., hf.Cartesian(0., 0., 0.), 0.)
        self.spherical = hf.HeadRelatedSpherical(1., 0., 0., hf.Cartesian(0., 0., 0.), 0.)
        self.cartesian = hf.Cartesian(0., 1., 0.)
        self.cartesian_1 = hf.Cartesian(1., 1., 1.)

    def tearDown(self) -> None:
        del self.spherical_0
        del self.spherical_1

    def test_operators(self):
        """
        Test the operators __init__ __repr__
        """
        # Test __init__
        self.assertEqual(self.spherical_1[0], 1.)
        self.assertEqual(self.spherical_1[1], 1.)
        self.assertEqual(self.spherical_1[2], 1.)
        self.assertEqual(self.spherical_1.origin, hf.Cartesian(0., 0., 0.))

        # Test origin point
        self.assertEqual(self.spherical_o.to_cartesian(), hf.Cartesian(1., 1., 1.))
        self.assertEqual(self.spherical_r.to_cartesian().to_hr_spherical(self.spherical_r.origin, 0.0),
                         hf.HeadRelatedSpherical(1., 2., 1., hf.Cartesian(0., 0., 0.), 0.))

        # Test __repr__
        self.assertEqual(self.spherical_0.__repr__(),
                         '<HR-Spherical: [0.0, 0.0, 0.0], around [0.0, 0.0, 0.0] with rotation 0.0>')

    def test_to_self(self):
        """
        Test the working of spherical _to_self
        """
        self.assertEqual(self.spherical._to_self(self.cartesian), self.spherical)
        self.assertEqual(self.spherical_o._to_self(self.cartesian_1), self.spherical_o)
        self.assertEqual(self.spherical_r._to_self(self.cartesian),
                         hf.HeadRelatedSpherical(1., -1., 0., hf.Cartesian(0., 0., 0.), 1.))

    def test_conversion(self):
        """
        Test conversion to Cartesian by double conversion
        """
        self.assertEqual(self.spherical_1.to_cartesian().to_hr_spherical(self.spherical_1.origin,
                                                                         self.spherical_1.rotation), self.spherical_1)
        self.assertEqual(self.spherical_r.to_cartesian().to_hr_spherical(self.spherical_r.origin,
                                                                         self.spherical_r.rotation), self.spherical_r)
        self.assertEqual(self.spherical_o.to_cartesian().to_hr_spherical(self.spherical_o.origin,
                                                                         self.spherical_o.rotation), self.spherical_o)


if __name__ == '__main__':
    unittest.main()
