import numpy as np
import matplotlib.pyplot as plt

import helper_functions as hf

"""
The very cool propagation model of this thesis :)
"""


class SoundRay:
    def __init__(self, pos_0: hf.Cartesian, vel_0: hf.Cartesian, t_0: float, s_0: float, beam_width: float,
                 atmosphere: hf.Atmosphere) -> None:
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

    def update_ray_velocity(self, delta_t: float) -> (hf.Cartesian, hf.Cartesian):
        """
        Update the ray velocity and direction forward by time step delta_t
        :param delta_t: time step (s)
        :return: Updated Cartesian velocity (m/s, m/s, m/s) and direction (m/s, m/s, m/s) vectors of the sound ray
        """
        # Get height from last position
        height = -self.pos[-1][2]
        # Determine direction change
        speed_of_sound_gradient = self.atmosphere.get_speed_of_sound_gradient(height)
        direction_change: hf.Cartesian = self.vel[-1].len() * hf.Cartesian(0, 0, speed_of_sound_gradient)
        direction: hf.Cartesian = self.dir[-1] + direction_change * delta_t

        # Get wind speed and speed of sound at height
        wind_speed = self.atmosphere.get_wind_speed(height)
        speed_of_sound = self.atmosphere.get_speed_of_sound(height)

        # Determine new ray velocity v = u + c * direction
        vel: hf.Cartesian = wind_speed * hf.Cartesian(0, 1, 0) + speed_of_sound * direction / direction.len()

        # Store new velocity and direction
        self.vel = np.append(self.vel, (vel,))
        self.dir = np.append(self.dir, (direction,))

        return vel, direction

    def update_ray_position(self, delta_t: float, vel: hf.Cartesian, direction: hf.Cartesian) -> (hf.Cartesian, float):
        """
        Update the ray position forward by time step delta_t and determine path step delta_s
        :param delta_t: time step (s)
        :param vel: Cartesian velocity vector (m/s, m/s, m/s) for current time step
        :param direction: Cartesian direction vector (m/s, m/s, m/s) for current time step
        :return: Updated position (m, m, m) for current time step and path step (m)
        """
        # Determine new position with forward euler stepping
        pos_new = self.pos[-1] + self.vel[-1] * delta_t
        # Check for reflections and if so: invert z-coordinate and z-velocity and z-direction
        if pos_new[2] >= 0:
            pos_new[2] = -pos_new[2]
            vel[2] = -vel[2]
            direction[2] = -direction[2]

        # Store new position
        self.pos = np.append(self.pos, (pos_new, ))

        # Propagate travelled distance
        delta_s = (self.vel[-1] * delta_t).len()
        self.s = np.append(self.s, self.s[-1] + delta_s)

        return pos_new, delta_s

    def ray_step(self, delta_t: float, ) -> (hf.Cartesian, hf.Cartesian, hf.Cartesian, float):
        """
        Logic for time stepping the sound rays forward one time step
        :param delta_t: time step (s)
        :return: Updated Cartesian velocity (m/s, m/s, m/s) and direction (m/s, m/s, m/s) vectors of the sound ray,
                 updated sound ray position (m, m, m), and the ray path step (m)
        """
        vel, direction = self.update_ray_velocity(delta_t)
        pos, delta_s = self.update_ray_position(delta_t, vel, direction)

        # Propagate time
        self.t = np.append(self.t, self.t[-1] + delta_t)

        return vel, direction, pos, delta_s

    def pos_array(self) -> np.array:
        """
        Convert awful position history to an easily readable array of shape (self.pos.size, 3)
        :return: numpy array with full position history unpacked
        """
        arr = np.empty((self.pos.size, 3))
        for pi, pos in enumerate(self.pos):
            arr[pi] = pos.vec

        return arr


if __name__ == '__main__':
    atm = hf.Atmosphere(35.5, 10.5, )
    phi, theta, fail = hf.uniform_spherical_grid(29)

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    offset = hf.Cartesian(0, 0, -35.5)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    startpt = [hf.Cartesian(x[i], y[i], z[i]) for i in range(len(x))]

    for p_init in startpt:
        c_init = p_init * atm.get_speed_of_sound(0) / p_init.len()
        soundray = SoundRay(p_init + offset, c_init, 0, 0, 0, atm)

        while soundray.t[-1] < 1:
            soundray.ray_step(1e-3)

        pos_arr = soundray.pos_array()

        ax.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
    plt.show()
