import numpy as np
import matplotlib.pyplot as plt

from . import t_0, r_air, g, gamma_air


class Atmosphere:
    def __init__(self):
        # Define the layer top altitudes and the temperature gradients per layer
        self.a = (-0.0065, 0.0000, 0.0010, 0.0028, 0.0000, -0.0028, -0.0020)
        self.h = (11000., 20000., 32000., 47000., 51000., 71000., 86000.)

        # Predefine the values per metre
        self.alt = np.arange(0, self.h[-1] + 1, 1)
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

    def conditions(self, altitude):
        temperature = np.interp(altitude, self.alt, self.temperature)
        pressure = np.interp(altitude, self.alt, self.pressure)
        density = np.interp(altitude, self.alt, self.density)

        return temperature, pressure, density

    def speed_of_sound(self, altitude):
        temperature, *_ = self.conditions(altitude)

        return np.sqrt(gamma_air * r_air * temperature)

    def plot(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()

        p1, = ax1.plot(self.temperature, self.alt / 1e3, color='tab:blue', label="Temperature")
        p2, = ax2.plot(self.pressure / 1e5, self.alt / 1e3, color='tab:orange', label="Pressure")
        p3, = ax2.plot(self.density, self.alt / 1e3, color='tab:green', label="Density")

        ax1.set(xlabel="Temperature (K)", ylabel="Altitude (km)")
        ax2.set(xlabel="Pressure (hPa) / Density (kg/m$^3$)")

        ax1.set_xlim(150, 290)
        ax1.set_ylim(-5, 90)
        ax1.set_yticks(np.arange(0, 90, 5))
        ax2.set_xlim(-.1, 1.3)

        plt.legend(handles=[p1, p2, p3])
        ax1.grid()
        plt.show()


if __name__ == '__main__':
    raise RuntimeError('Thou shalt not run this module on its own!')
