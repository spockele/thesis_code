import unittest
from unittest.mock import MagicMock
import numpy as np
import queue
import sys

import helper_functions as hf
import propagation_model as pm


class TestPropagationThread(unittest.TestCase):
    def setUp(self) -> None:
        # Create a soundray
        self.soundray = pm.Ray(hf.Cartesian(0, 0, 0), hf.Cartesian(0, 0, 0), 0, 0, hf.Atmosphere(1, 0))

        # Create an in and out queue
        inq = queue.Queue()
        outq = queue.Queue()
        inq.put(self.soundray)
        # Create a testing thread
        self.prop_thread = pm.PropagationThread(inq, outq, 1, hf.Cartesian(0, 0, 0), hf.ProgressThread(0, ''))

        # Create another in and out queue
        inq2 = queue.Queue()
        outq2 = queue.Queue()
        inq2.put(self.soundray)
        # Create another testing thread
        self.prop_thread_2 = pm.PropagationThread(inq2, outq2, 1, hf.Cartesian(0, 0, 0), hf.ProgressThread(1, 'test'))

    def tearDown(self) -> None:
        del self.soundray
        del self.prop_thread
        del self.prop_thread_2

    def test_run(self):
        """
        Tests the propagationThread's run feature
        """
        # Mock the soundray propagation to avoid weird
        self.soundray.propagate = MagicMock()
        # Run the first propagation thread
        self.prop_thread.start()
        # Check whether the run does what it must
        self.assertEqual(self.prop_thread.out_queue.qsize(), 1)
        self.soundray.propagate.assert_called_once_with(1, hf.Cartesian(0, 0, 0), 1.)
        self.assertEqual(self.prop_thread.p_thread.step, 2)
        # Run the second propagation thread
        self.prop_thread_2.start()
        # Check whether the run does what it must
        self.assertEqual(self.prop_thread_2.p_thread.step, 2)

    def test_interrupt(self):
        # Mock the soundray propagation and terminal output to avoid weird
        self.soundray.propagate = MagicMock()
        sys.stdout.write = MagicMock()

        # Add an extra soundray to the queue
        self.prop_thread.in_queue.put(self.soundray)

        # Mock the mainthread alive check
        mock_before = pm.threading.main_thread().is_alive
        pm.threading.main_thread().is_alive = MagicMock(return_value=False)

        # Start the propagation thread, knowing the main thread is dead
        self.prop_thread.start()

        # Check the interruption worked or not
        self.assertEqual(self.prop_thread.out_queue.qsize(), 0)
        self.assertEqual(self.prop_thread.in_queue.qsize(), 2)
        sys.stdout.write.assert_called()

        # Reset mock of the alive check
        pm.threading.main_thread().is_alive = mock_before


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

        # Create simple atmosphere
        hf.isa.read_from_file = MagicMock(return_value=atmosphere.copy())
        self.atm_simple = hf.Atmosphere(1, 0, )
        # Create simple sound ray
        self.soundray_simple = pm.Ray(self.p0, self.c0, self.s0, self.bw, self.atm_simple)

        # Create windy atmosphere
        self.atm_wind = hf.Atmosphere(1, 1, wind_z0=1/np.e)
        # Create windy sound ray
        self.soundray_wind = pm.Ray(hf.Cartesian(0, 0, -1), hf.Cartesian(0, 0, 0), self.s0, self.bw, self.atm_wind)

        # Create extreme speed of sound gradient
        atmosphere[4] = (1, 100001)
        # Create complex atmosphere
        hf.isa.read_from_file = MagicMock(return_value=atmosphere.copy())
        self.atm_complex = hf.Atmosphere(1, 0, )
        # Create complex sound ray
        self.soundray_complex = pm.Ray(self.p0, self.c0, self.s0, self.bw, self.atm_complex)

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

    def test_check_reception(self):
        """
        Simple tests for the check_reception functions
        """
        # Check with only one position saved should always be False
        self.assertFalse(self.soundray_simple.check_reception(hf.Cartesian(1, 0, 0), 1.5))
        self.assertFalse(self.soundray_simple.check_reception(hf.Cartesian(2, 0, 0), 1.5))
        self.assertFalse(self.soundray_simple.check_reception(hf.Cartesian(-1, 0, 0), 1.5))
        # Check with second position should return good stuff
        self.soundray_simple.pos = np.append(self.soundray_simple.pos, [hf.Cartesian(1.5, 0, 0), ])
        self.assertTrue(self.soundray_simple.check_reception(hf.Cartesian(1, 0, 0), 1.5))
        self.assertFalse(self.soundray_simple.check_reception(hf.Cartesian(2, 0, 0), 1.5))
        self.assertFalse(self.soundray_simple.check_reception(hf.Cartesian(-1, 0, 0), 1.5))

    def test_gaussian_reception(self):
        """
        Test for the Gaussian beam reception model
        """
        # Setup some fake case
        self.soundray_simple.pos = np.append(self.soundray_simple.pos, [hf.Cartesian(1, 0, 0), ])
        self.soundray_simple.s = np.append(self.soundray_simple.s, [1., ])
        f = np.linspace(1, 51.2e3, 100)
        # Test for simple receiver position
        rec = hf.Cartesian(0.5, 1, 0)
        expected = np.clip(np.exp(-1. / ((self.bw * .5)**2 + 1/(np.pi * f))), 0, 1)
        actual = self.soundray_simple.gaussian_reception(f, rec)
        self.assertTrue(np.all(np.round(expected, -9) == np.round(actual, -9)))
        # Check for less simple receiver position
        rec = hf.Cartesian(.5, 1., 1.)
        expected = np.clip(np.exp(-np.sqrt(2) / ((self.bw * .5) ** 2 + 1 / (np.pi * f))), 0, 1)
        actual = self.soundray_simple.gaussian_reception(f, rec)
        self.assertTrue(np.all(np.round(expected, -9) == np.round(actual, -9)))
