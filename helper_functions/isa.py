import numpy as np
import matplotlib.pyplot as plt

from . import t_0, r_air, g, gamma_air, write_to_file, read_from_file


class Atmosphere:
    def __init__(self, generate=False):
        # Define the layer top altitudes and the temperature gradients per layer
        self.a = (-0.0065, 0.0000, 0.0010, 0.0028, 0.0000, -0.0028, -0.0020)
        self.h = (11000., 20000., 32000., 47000., 51000., 71000., 86000.)

        if generate:
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
            self.speed_of_sound = np.sqrt(gamma_air * r_air * self.temperature)

            stack = np.vstack((self.alt, self.temperature, self.pressure, self.density, self.speed_of_sound))
            write_to_file(stack, './helper_functions/isa.dat')

        else:
            isa = read_from_file('./helper_functions/isa.dat')
            self.alt, self.temperature, self.pressure, self.density, self.speed_of_sound = isa

    def conditions(self, altitude):
        temperature = np.interp(altitude, self.alt, self.temperature)
        pressure = np.interp(altitude, self.alt, self.pressure)
        density = np.interp(altitude, self.alt, self.density)
        speed_of_sound = np.interp(altitude, self.alt, self.speed_of_sound)

        return temperature, pressure, density, speed_of_sound

    def plot(self):
        fig1, (ax1, ax3) = plt.subplots(1, 2, sharey='all')
        ax2 = ax1.twiny()

        ax4 = ax3.twiny()

        p1, = ax1.plot(self.temperature, self.alt / 1e3, color='k', label="Temperature")
        p11, = ax1.plot([t_0, t_0], [-5, 90], linestyle=":", color='0.5', label="$0^{\\circ}C$")
        p2, = ax2.plot(self.pressure / 1e5, self.alt / 1e3, color='k', linestyle="--", label="Pressure")
        p4, = ax4.plot(self.density, self.alt / 1e3, color='k', linestyle="--", label="Density")
        p3, = ax3.plot(self.speed_of_sound, self.alt / 1e3, color='k', label="Speed of sound")

        ax1.set(xlabel="Temperature (K)", ylabel="Altitude (km)")
        ax2.set(xlabel="Pressure (hPa)")
        ax4.set(xlabel="Density (kg/m$^3$)")
        ax3.set(xlabel="Speed of Sound (m/s)")

        ax1.set_xlim(150, 290)
        ax1.set_ylim(-5, 90)
        ax1.set_yticks(np.arange(0, 90, 5))
        ax2.set_xlim(-.1, 1.3)

        ax4.set_xlim(-.1, 1.32)
        ax3.set_xlim(270, 341)

        plt.tight_layout()

        ax1.legend(handles=[p1, p11, p2, ], loc=1)
        ax1.grid()
        ax3.legend(handles=[p3, p4, ], loc=1)
        ax3.grid()
        ax4.grid()
        plt.show()


if __name__ == '__main__':
    raise RuntimeError('Thou shalt not run this module on its own!')
