import unittest
import numpy as np

import helper_functions as hf


"""
========================================================================================================================
===                                                                                                                  ===
=== Unit tests for the geometry module                                                                               ===
===                                                                                                                  ===
========================================================================================================================
"""


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
