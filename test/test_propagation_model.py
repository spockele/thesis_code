import unittest
from unittest.mock import MagicMock
import numpy as np
import queue
import sys

import pandas as pd

import helper_functions as hf
import propagation_model as pm
import reception_model as rm


class TestRay(unittest.TestCase):
    def setUp(self) -> None:
        # Create general starting conditions
        self.p0 = hf.Cartesian(0, 0, 0)
        self.c0 = hf.Cartesian(1, 0, 0)
        self.s0 = 0.
        self.bw = np.pi / 36
        # Create simple atmosphere parameters
        atmosphere = np.ones((5, 2))
        atmosphere[0] = (0, 100000)

        # Create simple atmosphere
        hf.isa.read_from_file = MagicMock(return_value=atmosphere.copy())
        self.atm_simple = hf.Atmosphere(1, 0, 50., )
        # Create simple sound ray
        self.soundray_simple = pm.Ray(self.p0, self.c0, self.s0, self.bw)

        # Create windy atmosphere
        self.atm_wind = hf.Atmosphere(1, 1, 50., wind_z0=1/np.e)
        # Create windy sound ray
        self.soundray_wind = pm.Ray(hf.Cartesian(0, 0, -1), hf.Cartesian(0, 0, 0), self.s0, self.bw)

        # Create extreme speed of sound gradient
        atmosphere[4] = (1, 100001)
        # Create complex atmosphere
        hf.isa.read_from_file = MagicMock(return_value=atmosphere.copy())
        self.atm_complex = hf.Atmosphere(1, 0, 50., )
        # Create complex sound ray
        self.soundray_complex = pm.Ray(self.p0, self.c0, self.s0, self.bw)

    def tearDown(self) -> None:
        del self.atm_simple
        del self.soundray_simple
        del self.atm_wind
        del self.soundray_wind
        del self.atm_complex
        del self.soundray_complex

        del self.p0
        del self.c0
        del self.s0
        del self.bw

    def test_update_ray_velocity(self):
        """
        Test the ray velocity update with simple cases
        """
        # Nothing happens
        vel, direction = self.soundray_simple.update_ray_velocity(1, self.atm_simple)
        self.assertEqual(vel, hf.Cartesian(1, 0, 0))
        self.assertEqual(direction, hf.Cartesian(1, 0, 0))
        # Push into wind direction
        vel, direction = self.soundray_wind.update_ray_velocity(1, self.atm_wind)
        self.assertEqual(vel, hf.Cartesian(0, 1, 0))
        self.assertEqual(direction, hf.Cartesian(0, 0, 0))
        # Bend of direction by gradient
        vel, direction = self.soundray_complex.update_ray_velocity(1, self.atm_complex)
        self.assertEqual(vel, hf.Cartesian(1, 0, 1) / hf.Cartesian(1, 0, 1).len())
        self.assertEqual(direction, hf.Cartesian(1, 0, 1))

    def test_update_ray_position(self):
        """
        Test the position update of the soundrays
        """
        # Nothing happens
        pos, delta_s = self.soundray_simple.update_ray_position(1, hf.Cartesian(0, 0, 0), hf.Cartesian(0, 0, 0))
        self.assertEqual(pos, hf.Cartesian(0, 0, 0))
        self.assertEqual(delta_s, 0.)
        # Very Simple
        pos, delta_s = self.soundray_simple.update_ray_position(1, self.c0, self.c0)
        self.assertEqual(pos, hf.Cartesian(1, 0, 0))
        self.assertEqual(delta_s, 1.)
        # Slightly more complex
        pos, delta_s = self.soundray_simple.update_ray_position(1, hf.Cartesian(1, 1, -1), hf.Cartesian(1, 1, -1))
        self.assertEqual(pos, hf.Cartesian(1, 1, -1))
        self.assertEqual(delta_s, np.sqrt(3))

    def test_ray_step(self):
        """
        Test the ray velocity update with simple cases
        """
        # Nothing happens
        vel, direction, pos, delta_s = self.soundray_simple.ray_step(1, self.atm_simple)
        self.assertEqual(vel, hf.Cartesian(1, 0, 0))
        self.assertEqual(direction, hf.Cartesian(1, 0, 0))
        self.assertEqual(pos, hf.Cartesian(1, 0, 0))
        self.assertEqual(delta_s, 1.)
        # Push into wind direction
        vel, direction, pos, delta_s = self.soundray_wind.ray_step(1, self.atm_wind)
        self.assertEqual(vel, hf.Cartesian(0, 1, 0))
        self.assertEqual(direction, hf.Cartesian(0, 0, 0))
        self.assertEqual(pos, hf.Cartesian(0, 1, -1))
        self.assertEqual(delta_s, 1.)
        # Bend of direction by gradient
        vel, direction, pos, delta_s = self.soundray_complex.ray_step(1, self.atm_complex)
        self.assertEqual(vel, hf.Cartesian(1, 0, -1) / hf.Cartesian(1, 0, -1).len())
        self.assertEqual(direction, hf.Cartesian(1, 0, -1))
        self.assertEqual(pos, hf.Cartesian(1, 0, -1) / hf.Cartesian(1, 0, -1).len())
        self.assertAlmostEqual(delta_s, 1.)

    def test_check_reception(self):
        """
        Simple tests for the check_reception functions
        """
        # Create dummy receivers
        receiver_0 = rm.Receiver({'pos': (1., 0., 0.), 'rotation': 0., 'index': 0})
        receiver_1 = rm.Receiver({'pos': (2., 0., 0.), 'rotation': 0., 'index': 0})
        receiver_2 = rm.Receiver({'pos': (-1., 0., 0.), 'rotation': 0., 'index': 0})
        # Check with only one position saved should always be False
        self.assertFalse(self.soundray_simple.check_reception(receiver_0, 1.5))
        self.assertFalse(self.soundray_simple.check_reception(receiver_1, 1.5))
        self.assertFalse(self.soundray_simple.check_reception(receiver_2, 1.5))
        # Check with second position should return good stuff
        self.soundray_simple.pos = np.append(self.soundray_simple.pos, [hf.Cartesian(1.5, 0, 0), ])
        self.assertTrue(self.soundray_simple.check_reception(receiver_0, 1.5))
        self.assertFalse(self.soundray_simple.check_reception(receiver_1, 1.5))
        self.assertFalse(self.soundray_simple.check_reception(receiver_2, 1.5))


class TestSoundRay(unittest.TestCase):
    def setUp(self) -> None:
        # Create general starting conditions
        self.p0 = hf.Cartesian(0, 0, 0)
        self.c0 = hf.Cartesian(1, 0, 0)
        self.s0 = 0.
        self.bw = np.pi / 36
        # Create simple atmosphere parameters
        atmosphere = np.ones((5, 2))
        atmosphere[0] = (0, 100000)

        spectrum = pd.DataFrame(1, columns=['a'], index=hf.octave_band_fc(1))
        models = ('spherical', 'atmospheric')

        # Create simple atmosphere
        hf.isa.read_from_file = MagicMock(return_value=atmosphere.copy())
        self.atm_simple = hf.Atmosphere(1, 0, 50., )
        # Create simple sound ray
        self.soundray_simple = pm.SoundRay(self.p0, self.c0, self.s0, hf.Cartesian(0, 0, 0), self.bw, spectrum, models)

        # Create windy atmosphere
        self.atm_wind = hf.Atmosphere(1, 1, 50., wind_z0=1 / np.e)
        # Create windy sound ray
        self.soundray_wind = pm.SoundRay(hf.Cartesian(0, 0, -1), hf.Cartesian(0, 0, 0), self.s0, hf.Cartesian(0, 0, 0),
                                         self.bw, spectrum, models)

        # Create extreme speed of sound gradient
        atmosphere[4] = (1, 100001)
        # Create complex atmosphere
        hf.isa.read_from_file = MagicMock(return_value=atmosphere.copy())
        self.atm_complex = hf.Atmosphere(1, 0, 50., )
        # Create complex sound ray
        self.soundray_complex = pm.SoundRay(self.p0, self.c0, self.s0, hf.Cartesian(0, 0, 0), self.bw, spectrum, models)

    def tearDown(self) -> None:
        del self.atm_simple
        del self.soundray_simple
        del self.atm_wind
        del self.soundray_wind
        del self.atm_complex
        del self.soundray_complex

        del self.p0
        del self.c0
        del self.s0
        del self.bw

    def test_gaussian_reception(self):
        """
        Test for the Gaussian beam reception model
        """
        # Setup some fake case
        self.soundray_simple.pos = np.append(self.soundray_simple.pos, [hf.Cartesian(1, 0, 0), ])
        self.soundray_simple.s = np.append(self.soundray_simple.s, [1., ])
        f = hf.octave_band_fc(1)
        # Test for simple receiver position
        rec = rm.Receiver({'pos': (0.5, 1, 0), 'index': 0, 'rotation': 0.})
        expected = np.clip(np.exp(-1. / ((self.bw * .5)**2 + 1/(np.pi * f))), 0, 1)
        self.soundray_simple.gaussian_factor(rec)
        self.assertTrue(np.all(np.round(expected, -9) == np.round(self.soundray_simple.spectrum['gaussian'], -9)))
        # Check for less simple receiver position
        rec = rm.Receiver({'pos': (0.5, 1, 1), 'index': 0, 'rotation': 0.})
        expected = np.clip(np.exp(-np.sqrt(2) / ((self.bw * .5) ** 2 + 1 / (np.pi * f))), 0, 1)
        self.soundray_simple.gaussian_factor(rec)
        self.assertTrue(np.all(np.round(expected, -9) == np.round(self.soundray_simple.spectrum['gaussian'], -9)))
