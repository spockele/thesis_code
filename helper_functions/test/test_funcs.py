import unittest
from unittest.mock import MagicMock

import numpy as np

import helper_functions as hf


class TestFuncs(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_limit_angle(self):
        """
        Tests the limit_angle() function
        """
        # Test edge and centre cases
        self.assertAlmostEqual(0., hf.limit_angle(0.))
        self.assertAlmostEqual(np.pi, hf.limit_angle(np.pi))
        self.assertAlmostEqual(-np.pi, hf.limit_angle(-np.pi))
        # Test inside limits
        self.assertAlmostEqual(np.pi / 2, hf.limit_angle(np.pi / 2))
        self.assertAlmostEqual(-np.pi / 2, hf.limit_angle(-np.pi / 2))
        # Test outside limits
        self.assertAlmostEqual(- np.pi / 2, hf.limit_angle(3 * np.pi / 2))
        self.assertAlmostEqual(np.pi / 2, hf.limit_angle(-3 * np.pi / 2))
        self.assertAlmostEqual(0., hf.limit_angle(2 * np.pi))
        self.assertAlmostEqual(0., hf.limit_angle(-2 * np.pi))

    def test_uniform_spherical_grid(self):
        """
        Checks the calls for the uniform grid generator
        """
        hf.warnings.warn = MagicMock()
        self.assertTrue(hf.uniform_spherical_grid(1)[2])
        hf.warnings.warn.assert_called_once()

        self.assertAlmostEqual(hf.uniform_spherical_grid(4)[3], np.sqrt(np.pi))
        self.assertFalse(hf.uniform_spherical_grid(4)[2])

    def test_a_weighting(self):
        """
        Check A-weighting L_a against table of hand-calculated values
        """
        f, l_a = hf.read_from_file('./a-weights_f-L.csv').T
        l_a_func = hf.a_weighting(f)

        for il, l in enumerate(l_a):
            self.assertAlmostEqual(l_a[il], l_a_func[il])


class TestProgressThread(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_progress_thread = hf.ProgressThread(10, 'Testing')

    def tearDown(self) -> None:
        del self.mock_progress_thread

    def test_run_update(self):
        """
        Test the ProgressThread run, stop, and update functionality
        """
        # Mock the terminal output so no printing occurs
        hf.funcs.sys.stdout.write = MagicMock()
        hf.funcs.sys.stdout.flush = MagicMock()
        # Mock the main_thread().is_alive() function to always be True
        hf.threading.main_thread().is_alive = MagicMock(return_value=True)

        # Start the progress thread
        self.mock_progress_thread.start()
        # Wait just a pinch
        hf.time.sleep(.1)

        # Check some calls
        hf.funcs.sys.stdout.write.assert_called()
        hf.funcs.sys.stdout.flush.assert_called()

        # Update the progress thread
        self.mock_progress_thread.update()
        # Stop the thread and wait a pinch
        self.mock_progress_thread.stop()
        hf.time.sleep(.25)

        # Check values for functioning
        self.assertFalse(self.mock_progress_thread.work)
        self.assertFalse(self.mock_progress_thread.is_alive())
        self.assertEqual(2, self.mock_progress_thread.step)

    def test_interrupt(self):
        """
        Test code interruption of the ProgressThread
        """
        # Mock the terminal output so no printing occurs
        hf.funcs.sys.stdout.write = MagicMock()
        hf.funcs.sys.stdout.flush = MagicMock()
        # Mock the main_thread().is_alive() function to always be True
        hf.threading.main_thread().is_alive = MagicMock(return_value=True)

        # Start the progress thread
        self.mock_progress_thread.start()
        # Simulate a system interrupt by changing the mock value
        hf.threading.main_thread().is_alive = MagicMock(return_value=False)
        # Give it a hot second
        hf.time.sleep(.25)

        # Check that the interrupt worked as expected
        self.assertTrue(self.mock_progress_thread.work)
        self.assertFalse(self.mock_progress_thread.is_alive())
