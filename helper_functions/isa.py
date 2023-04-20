import numpy as np
import matplotlib.pyplot as plt

from . import t_0, r_air, g, gamma_air, write_to_file, read_from_file


"""
Definition of the ISO standard atmosphere (*ISO 2533-1975*)
"""


class Atmosphere:
    # Define the layer top altitudes and the temperature gradients per layer
    a = (-6.5e-3, 0., 1.0e-3, 2.8e-3, 0., -2.8e-3, -2.0e-3)
    h = (11000., 20000., 32000., 47000., 51000., 71000., 86000.)

    def __init__(self, z_0, ws_0, wind_z0: float = None, delta_h: float = None):
        """
        Class containing the ISO Standard Atmosphere (ISO 2533-1975) and a logarithmic wind profile.
        :param z_0: Reference height for the wind profile
        :param ws_0: Wind speed at reference height z_0
        :param wind_z0: Roughness height for the wind profile
        :param delta_h: Float indicating the delta altitude for generation. ISA will not generate if None
        """
        # Load the ISA
        if delta_h is None:
            isa = read_from_file('./helper_functions/data/isa.dat')
            self.alt, self.temperature, self.pressure, self.density, self.speed_of_sound = isa
        # Generate the ISA
        else:
            # Predefine the values per metre
            self.alt = np.arange(0, self.h[-1] + delta_h, delta_h)
            self.temperature = np.zeros(self.alt.shape)
            self.pressure = np.zeros(self.alt.shape)
            self.density = np.zeros(self.alt.shape)

            self.temperature[0], self.pressure[0] = 15. + t_0, 101325.

            layer = 0
            for hi, h in enumerate(self.alt[1:]):
                layer += 1 if h > self.h[layer] else 0

                self.temperature[hi + 1] = self.temperature[hi] + self.a[layer] * (h - self.alt[hi])

                if self.a[layer] == 0.:
                    e = -(g * (h - self.alt[hi]) / (r_air * self.temperature[hi]))
                    self.pressure[hi + 1] = self.pressure[hi] * np.exp(e)

                else:
                    e = -(g / (self.a[layer] * r_air))
                    self.pressure[hi + 1] = self.pressure[hi] * (self.temperature[hi + 1] / self.temperature[hi]) ** e

            self.density = self.pressure / (r_air * self.temperature)
            self.speed_of_sound = np.sqrt(gamma_air * r_air * self.temperature)

            stack = np.vstack((self.alt, self.temperature, self.pressure, self.density, self.speed_of_sound))
            write_to_file(stack, './helper_functions/data/isa.dat')

        # Set the wind profile references
        if wind_z0 is None:
            self.z0 = .03
        else:
            self.z0 = wind_z0

        self.ws_z0 = ws_0 / np.log(z_0 / self.z0)

    def get_temperature(self, altitude):
        """
        Interpolate temperature from table
        :param altitude: Altitude(s) to interpolate at
        :return: The temperature(s)
        """
        return np.interp(altitude, self.alt, self.temperature)

    def get_pressure(self, altitude):
        """
        Interpolate pressure from table
        :param altitude: Altitude(s) to interpolate at
        :return: The pressure(s)
        """
        return np.interp(altitude, self.alt, self.pressure)

    def get_density(self, altitude):
        """
        Interpolate density from table
        :param altitude: Altitude(s) to interpolate at
        :return: The density(s)
        """
        return np.interp(altitude, self.alt, self.density)

    def get_speed_of_sound(self, altitude):
        """
        Interpolate speed of sound from table
        :param altitude: Altitude(s) to interpolate at
        :return: The speed(s) of sound
        """
        return np.interp(altitude, self.alt, self.speed_of_sound)

    def get_speed_of_sound_gradient(self, altitude: float):
        """
        Determine the gradient of c at altitude
        :param altitude: Altitude to determine gradient at
        :return: The speed of sound gradient
        """
        idx = np.argwhere(self.alt <= altitude)[-1], np.argwhere(self.alt > altitude)[0]
        return self.speed_of_sound[idx[1]][0] - self.speed_of_sound[idx[0]][0]

    def get_wind_speed(self, altitude):
        """
        Get the wind speed from the logarithmic profile
        :param altitude: Altitude(s) at which to determine the wind speed
        :return: The wind speed(s)
        """
        if altitude < self.z0:
            return 0

        return self.ws_z0 * np.log(altitude / self.z0)

    def get_conditions(self, altitude):
        """
        Obtain all atmospheric conditions
        :param altitude: Altitude(s) to obtain conditions at
        :return: The condition(s)
        """
        temperature = self.get_temperature(altitude)
        pressure = self.get_pressure(altitude)
        density = self.get_density(altitude)
        speed_of_sound = self.get_speed_of_sound(altitude)
        wind_speed = self.get_speed_of_sound(altitude)

        return temperature, pressure, density, speed_of_sound, wind_speed

    def plot(self, to_file=True):
        """
        Create plots of all atmospheric parameters
        :param to_file: Boolean to set outputting the plots to pdf files
        """
        plt.figure(1)
        plt.plot(self.temperature, self.alt / 1e3, color='k', label="ISA")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Altitude (km)")
        plt.xlim(180, 300)
        plt.ylim(0, 87)
        plt.grid()
        plt.tight_layout()

        plt.figure(2)
        plt.plot(self.pressure / 1e5, self.alt / 1e3, color='k', label="ISA")
        plt.xlabel("Pressure (hPa)")
        plt.ylabel("Altitude (km)")
        plt.xlim(-.05, 1.05)
        plt.ylim(0, 87)
        plt.grid()
        plt.tight_layout()

        plt.figure(3)
        plt.plot(self.density, self.alt / 1e3, color='k', label="ISA")
        plt.xlabel("Density (kg/m$^3$)")
        plt.ylabel("Altitude (km)")
        plt.xlim(-.05, 1.25)
        plt.ylim(0, 87)
        plt.grid()
        plt.tight_layout()

        plt.figure(4)
        plt.plot(self.speed_of_sound, self.alt / 1e3, color='k',)
        plt.xlabel("Speed of Sound (m/s)")
        plt.ylabel("Altitude (km)")
        plt.xlim(270, 350)
        plt.ylim(0, 87)
        plt.grid()
        plt.tight_layout()

        plt.figure(5)
        plt.plot(self.get_wind_speed(self.alt[self.alt > 0]), self.alt[self.alt > 0] / 1e3, color='k')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Altitude (km)')
        plt.xlim(4, 18)
        plt.ylim(0, 11)
        plt.grid()
        plt.tight_layout()

        if to_file:
            plt.figure(1)
            plt.savefig('./plots/isa_temperature.pdf')
            plt.figure(2)
            plt.savefig('./plots/isa_pressure.pdf')
            plt.figure(3)
            plt.savefig('./plots/isa_density.pdf')
            plt.figure(4)
            plt.savefig('./plots/isa_speed_of_sound.pdf')
            plt.figure(5)
            plt.savefig('./plots/isa_wind_speed.pdf')

        plt.show()


if __name__ == '__main__':
    raise RuntimeError('Thou shalt not run this module on its own!')
