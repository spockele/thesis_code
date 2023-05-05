import unittest
from unittest.mock import MagicMock
import numpy as np
import warnings

import helper_functions as hf


"""
========================================================================================================================
===                                                                                                                  ===
=== Unit tests for the geometry module                                                                               ===
===                                                                                                                  ===
========================================================================================================================
"""


class TestGeometry(unittest.TestCase):
    def test_uniform_spherical_grid(self):
        """
        Checks the calls for the uniform grid generator
        """
        warnings.warn = MagicMock()
        self.assertTrue(hf.uniform_spherical_grid(1)[2])
        warnings.warn.assert_called_once()

        self.assertAlmostEqual(hf.uniform_spherical_grid(4)[2], np.sqrt(np.pi))
        self.assertFalse(hf.uniform_spherical_grid(4)[1])


class TestPerpendicularPlane3D(unittest.TestCase):
    def setUp(self) -> None:
        self.p0 = hf.Cartesian(0, 0, 0)
        self.p1 = hf.Cartesian(1, 0, 0)
        self.p2 = hf.Cartesian(1, 1, 1)

        self.plane_1 = hf.PerpendicularPlane3D(self.p1, self.p0)
        self.plane_2 = hf.PerpendicularPlane3D(self.p2, self.p0)

    def tearDown(self) -> None:
        del self.plane_1
        del self.plane_2

    def test_distance_to_point(self):
        """
        Test calculation of distance from plane to a point in space
        """
        self.assertEqual(self.plane_1.distance_to_point(self.p0), 0.)
        self.assertEqual(self.plane_1.distance_to_point(self.p1), 1.)
        self.assertEqual(self.plane_1.distance_to_point(self.p2), 1.)
        self.assertEqual(round(self.plane_2.distance_to_point(self.p2), 6), round(np.sqrt(3), 6))
        self.assertEqual(self.plane_2.distance_to_point(self.p0), 0.)
