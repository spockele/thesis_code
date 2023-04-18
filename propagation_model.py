import numpy as np
import matplotlib.pyplot as plt

import helper_functions as hf

"""
The very cool propagation model of this thesis :)
"""


class SoundRay:
    def __init__(self, pos_0: hf.Cartesian, vel_0: hf.Cartesian, t_0: float, s_0: float, beam_width: float,
                 atmosphere: hf.Atmosphere):
        """
        Class for the propagation sound ray model.

        :param pos_0: initial position in cartesian coordinates (m, m, m)
        :param vel_0: initial velocity in cartesian coordinates (m/s, m/s, m/s)
        :param t_0: initial time (s)
        :param s_0: initial beam length (m)
        :param beam_width: initial beam width angle (rad)
        """
        # Set initial conditions
        self.pos = np.array([pos_0, ])
        self.vel = np.array([vel_0, ])
        self.dir = np.array([vel_0, ])
        self.t = np.array([t_0, ])
        self.s = np.array([s_0])
        # Set fixed parameters
        self.bw = beam_width
        self.atmosphere = atmosphere

    def propagate(self, delta_t, ):
        # Get height from last position
        h = -self.pos[-1][2]
        # Determine direction change
        dc_dh = self.atmosphere.get_speed_of_sound_gradient(h)
        ddir_dt = self.vel[-1].len() * hf.Cartesian(0, 0, dc_dh)
        direction: hf.Cartesian = self.dir[-1] + ddir_dt * delta_t

        # Get wind speed and speed of sound at height
        u = self.atmosphere.get_wind_speed(h)
        c = self.atmosphere.get_speed_of_sound(h)

        # Determine new ray velocity v = u + c * direction
        vel: hf.Cartesian = u * hf.Cartesian(0, 1, 0) + c * direction / direction.len()
        # Determine new position with forward euler stepping
        pos_new = self.pos[-1] + vel * delta_t
        # Check for reflections and if so: invert z-coordinate and z-velocity and z-direction
        if pos_new[2] >= 0:
            pos_new[2] = -pos_new[2]
            vel[2] = -vel[2]
            direction[2] = -direction[2]

        # Store velocity, direction and velocity
        self.vel = np.append(self.vel, (vel,))
        self.dir = np.append(self.dir, (direction,))
        self.pos = np.append(self.pos, (pos_new, ))
        # Propagate time
        self.t = np.append(self.t, self.t[-1] + delta_t)
        # Propagate travelled distance
        delta_s = (vel * delta_t).len()
        self.s = np.append(self.s, self.s[-1] + delta_s)

    def pos_array(self):
        arr = np.empty((self.pos.size, 3))
        for pi, pos in enumerate(self.pos):
            arr[pi] = pos.vec

        return arr


if __name__ == '__main__':
    atm = hf.Atmosphere(35.5, 10.5, )
    phi, theta = hf.uniform_spherical_grid(25)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    offset = hf.Cartesian(0, 0, -35.5)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    startpt = [hf.Cartesian(x[i], y[i], z[i]) for i in range(len(x))]

    for p_init in startpt:
        c_init = p_init * atm.get_speed_of_sound(0) / p_init.len()
        soundray = SoundRay(p_init + offset, c_init, 0, 0, 0, atm)

        while soundray.t[-1] < 100:
            soundray.propagate(1e-2)

        pos = soundray.pos_array()

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
    plt.show()
