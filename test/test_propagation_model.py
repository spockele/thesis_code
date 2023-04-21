import unittest
from unittest.mock import MagicMock
import numpy as np

import helper_functions as hf
import propagation_model as pm


class TestSoundRay(unittest.TestCase):
    def setUp(self) -> None:
        # Create general starting conditions
        self.p0 = hf.Cartesian(0, 0, 0)
        self.c0 = hf.Cartesian(1, 0, 0)
        self.t0 = 0.
        self.s0 = 0.
        self.bw = np.pi / 36
        # Create simple simple atmosphere parameters
        atmosphere = np.ones((5, 2))
        atmosphere[0] = (0, 100000)

        # Create simple simple atmosphere
        hf.isa.read_from_file = MagicMock(return_value=atmosphere.copy())
        self.atm_simple = hf.Atmosphere(1, 0, )
        # Create simple sound ray
        self.soundray_simple = pm.SoundRay(self.p0, self.c0, self.t0, self.s0, self.bw, self.atm_simple)

        # Create windy atmosphere
        self.atm_wind = hf.Atmosphere(1, 1, wind_z0=1/np.e)
        # Create windy sound ray
        self.soundray_wind = pm.SoundRay(hf.Cartesian(0, 0, -1), hf.Cartesian(0, 0, 0), self.t0, self.s0, self.bw, self.atm_wind)

        # Create extreme speed of sound gradient
        atmosphere[4] = (1, 100001)
        # Create complex atmosphere
        hf.isa.read_from_file = MagicMock(return_value=atmosphere.copy())
        self.atm_complex = hf.Atmosphere(1, 0, )
        # Create complex sound ray
        self.soundray_complex = pm.SoundRay(self.p0, self.c0, self.t0, self.s0, self.bw, self.atm_complex)

    def tearDown(self) -> None:
        del self.atm_simple
        del self.soundray_simple
        del self.atm_wind
        del self.soundray_wind
        del self.atm_complex
        del self.soundray_complex

        del self.p0
        del self.c0
        del self.t0
        del self.s0
        del self.bw

    def test_update_ray_velocity(self):
        """
        Test the ray velocity update with simple cases
        """
        # Nothing happens
        vel, direction = self.soundray_simple.update_ray_velocity(1)
        self.assertEqual(vel, hf.Cartesian(1, 0, 0))
        self.assertEqual(direction, hf.Cartesian(1, 0, 0))
        # Push into wind direction
        vel, direction = self.soundray_wind.update_ray_velocity(1)
        self.assertEqual(vel, hf.Cartesian(0, 1, 0))
        self.assertEqual(direction, hf.Cartesian(0, 0, 0))
        # Bend of direction by gradient
        vel, direction = self.soundray_complex.update_ray_velocity(1)
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
        vel, direction, pos, delta_s = self.soundray_simple.ray_step(1)
        self.assertEqual(vel, hf.Cartesian(1, 0, 0))
        self.assertEqual(direction, hf.Cartesian(1, 0, 0))
        self.assertEqual(pos, hf.Cartesian(1, 0, 0))
        self.assertEqual(delta_s, 1.)
        # Push into wind direction
        vel, direction, pos, delta_s = self.soundray_wind.ray_step(1)
        self.assertEqual(vel, hf.Cartesian(0, 1, 0))
        self.assertEqual(direction, hf.Cartesian(0, 0, 0))
        self.assertEqual(pos, hf.Cartesian(0, 1, -1))
        self.assertEqual(delta_s, 1.)
        # Bend of direction by gradient
        vel, direction, pos, delta_s = self.soundray_complex.ray_step(1)
        self.assertEqual(vel, hf.Cartesian(1, 0, -1) / hf.Cartesian(1, 0, -1).len())
        self.assertEqual(direction, hf.Cartesian(1, 0, -1))
        self.assertEqual(pos, hf.Cartesian(1, 0, -1) / hf.Cartesian(1, 0, -1).len())
        self.assertAlmostEqual(delta_s, 1.)
