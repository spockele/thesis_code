import unittest
from unittest.mock import MagicMock
import numpy as np

import helper_functions as hf


class TestAtmosphere(unittest.TestCase):
    def setUp(self) -> None:
        # Hand calculated ISA values
        self.heights = np.array((0., 11000., 20000., 32000., 47000., 51000., 71000., 86000.))
        self.temps = np.array((288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 184.65))
        self.press = np.array((1.01e+05, 2.26e+04, 5.47e+03, 8.67e+02, 1.11e+02, 6.68e+01, 3.95e+00, 3.02e-01))
        self.denss = self.press / (hf.r_air * self.temps)
        self.soss = np.sqrt(hf.gamma_air * hf.r_air * self.temps)
        self.values = np.vstack((self.heights, self.temps, self.press, self.denss, self.soss))
        # Create mock for read_from_file function to insert these values
        hf.isa.read_from_file = MagicMock(return_value=self.values)
        # Create mock atmosphere for testing
        self.atmosphere_mock = hf.Atmosphere(1., 0.)

        # Create a mocked atmosphere with wind
        self.atmosphere_wind = hf.Atmosphere(1., 1., wind_z0=1/np.e)

    def test_generation(self):
        """
        Test the generator of the ISA class
        """
        # Mock some functions
        hf.isa.np.arange = MagicMock(return_value=self.heights)
        hf.isa.write_to_file = MagicMock()
        # Generate the atmosphere
        atmosphere = hf.Atmosphere(1., 0., delta_h=1)
        # Check that mocked functions got called
        hf.isa.np.arange.assert_called_once()
        hf.isa.write_to_file.assert_called_once()

        # Check the generated atmosphere to 3 digits
        for ia, alt in enumerate(atmosphere.alt):
            self.assertEqual(round(self.atmosphere_mock.temperature[ia], -3),
                             round(atmosphere.temperature[ia], -3))
            self.assertEqual(round(self.atmosphere_mock.pressure[ia], -3),
                             round(atmosphere.pressure[ia], -3))
            self.assertEqual(round(self.atmosphere_mock.density[ia], -3),
                             round(atmosphere.density[ia], -3))
            self.assertEqual(round(self.atmosphere_mock.speed_of_sound[ia], -3),
                             round(atmosphere.speed_of_sound[ia], -3))

        # Check wind profile parameters
        self.assertEqual(atmosphere.z0, .03)
        self.assertEqual(atmosphere.ws_z0, 0.)
        self.assertEqual(self.atmosphere_wind.ws_z0, 1.)
        self.assertEqual(self.atmosphere_wind.z0, 1/np.e)

    def test_get_temperature(self):
        """
        Tests of Atmosphere().get_temperature()
        """
        # Check return value at one height per ISA layer
        for i in range(7):
            height = self.heights[i] + 1000
            expected_temps = self.temps[i] + ((height - self.heights[i]) * (self.temps[i+1] - self.temps[i]) /
                                              (self.heights[i+1] - self.heights[i]))
            self.assertEqual(self.atmosphere_mock.get_temperature(height), expected_temps)

    def test_get_pressure(self):
        """
        Tests of Atmosphere().get_pressure()
        """
        # Check return value at one height per ISA layer
        for i in range(7):
            height = self.heights[i] + 1000
            expected_press = self.press[i] + ((height - self.heights[i]) * (self.press[i+1] - self.press[i]) /
                                              (self.heights[i+1] - self.heights[i]))
            self.assertEqual(self.atmosphere_mock.get_pressure(height), expected_press)

    def test_get_density(self):
        """
        Tests of Atmosphere().get_density()
        """
        # Check return value at one height per ISA layer
        for i in range(7):
            height = self.heights[i] + 1000
            expected_denss = self.denss[i] + ((height - self.heights[i]) * (self.denss[i+1] - self.denss[i]) /
                                              (self.heights[i+1] - self.heights[i]))
            self.assertEqual(self.atmosphere_mock.get_density(height), expected_denss)

    def test_get_speed_of_sound(self):
        """
        Tests of Atmosphere().get_speed_of_sound()
        """
        # Check return value at one height per ISA layer
        for i in range(7):
            height = self.heights[i] + 1000
            expected_soss = self.soss[i] + ((height - self.heights[i]) * (self.soss[i+1] - self.soss[i]) /
                                            (self.heights[i+1] - self.heights[i]))
            self.assertEqual(self.atmosphere_mock.get_speed_of_sound(height), expected_soss)

    def test_get_speed_of_sound_gradient(self):
        """
        Tests of Atmosphere().get_speed_of_sound_gradient()
        """
        # Check return value at one height per ISA layer
        for i in range(7):
            expected_grad = (self.soss[i+1] - self.soss[i]) / (self.heights[i+1] - self.heights[i])
            self.assertEqual(self.atmosphere_mock.get_speed_of_sound_gradient(self.heights[i]), expected_grad)

    def test_get_wind_speed(self):
        """
        Tests of Atmosphere().get_get_wind_speed()
        """
        # Check return value at easy values
        for i in range(7):
            h = np.exp(i)
            self.assertEqual(self.atmosphere_wind.get_wind_speed(h), i+1.)

        self.assertEqual(self.atmosphere_wind.get_wind_speed(0), 0.)

    def test_get_conditions(self):
        """
        Check that Atmosphere().get_conditions() makes the right calls
        :return:
        """
        # Mock the functions
        self.atmosphere_mock.get_temperature = MagicMock()
        self.atmosphere_mock.get_pressure = MagicMock()
        self.atmosphere_mock.get_density = MagicMock()
        self.atmosphere_mock.get_speed_of_sound = MagicMock()
        self.atmosphere_mock.get_wind_speed = MagicMock()
        # Run one case
        h = 1
        self.atmosphere_mock.get_conditions(h)
        # Check the calls
        self.atmosphere_mock.get_temperature.assert_called_once_with(h)
        self.atmosphere_mock.get_pressure.assert_called_once_with(h)
        self.atmosphere_mock.get_density.assert_called_once_with(h)
        self.atmosphere_mock.get_speed_of_sound.assert_called_once_with(h)
        self.atmosphere_mock.get_wind_speed.assert_called_once_with(h)

    def test_plot(self):
        """
        Check the plt calls of Atmosphere().plot()
        """
        for tf in (True, False):
            # Create mocks
            hf.isa.plt.figure = MagicMock()
            hf.isa.plt.plot = MagicMock()
            hf.isa.plt.savefig = MagicMock()
            hf.isa.plt.show = MagicMock()
            # Run case and ensure pyplot is closed
            self.atmosphere_mock.plot(to_file=tf)
            hf.isa.plt.close()
            # Check function calls to pyplot
            hf.isa.plt.figure.assert_called()
            hf.isa.plt.plot.assert_called()
            if tf:
                hf.isa.plt.savefig.assert_called()
            else:
                hf.isa.plt.savefig.assert_not_called()
            hf.isa.plt.show.assert_called()
