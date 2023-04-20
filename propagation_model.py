import numpy as np
import matplotlib.pyplot as plt
import threading
import queue

import helper_functions as hf

"""
The very cool propagation model of this thesis :)
"""


class PropagationThread(threading.Thread):
    def __init__(self, in_queue: queue.Queue, out_queue: queue.Queue, delta_t: float, receiver: hf.Cartesian,
                 progress: hf.ProgressThread = None) -> None:
        """

        :param in_queue:
        :param delta_t:
        """
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue

        self.delta_t = delta_t
        self.receiver = receiver

        if progress is None:
            self.p_thread = hf.ProgressThread(1)
        else:
            self.p_thread = progress

    def run(self) -> None:
        while not self.in_queue.empty():
            ray: SoundRay = self.in_queue.get()
            ray.propagate(self.delta_t, self.receiver)
            self.out_queue.put(ray)

            self.p_thread.update()
            if not threading.main_thread().is_alive():
                print(f'Stopped {self} after Interupt of MainThread')
                break


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

    def check_reception(self, receiver: hf.Cartesian, delta_s: float):
        """
        Check if the ray went past a receiver point.
        :param receiver: the receiver point in Cartesian coordinates (m, m, m).
        :param delta_s: the last ray path step (m).
        :return: boolean indicating whether the receiver has been passed.
        """
        # Cannot check if less than two points
        if self.pos.size < 2:
            return False
        # Determine perpendicular planes at last 2 points
        plane1 = hf.PerpendicularPlane3D(self.pos[-1], self.pos[-2])
        plane2 = hf.PerpendicularPlane3D(self.pos[-2], self.pos[-1])
        # Determine distances between planes and receiver point
        dist1 = plane1.distance_to_point(receiver)
        dist2 = plane2.distance_to_point(receiver)
        # Check condition for point being between planes
        if dist1 <= delta_s and dist2 <= delta_s:
            # If yes, the ray has passed the receiver: SUCCESS!!!
            return True
        # If all else fails: this ray has not yet passed, probably
        return False

    def gaussian_reception(self, frequency: np.array, receiver: hf.Cartesian):
        """

        :param frequency:
        :param receiver:
        :return:
        """
        plane = hf.PerpendicularPlane3D(self.pos[-1], self.pos[-2])
        dist1 = plane.distance_to_point(receiver)
        dist2 = (receiver - self.pos[-2]).len()

        n_sq = dist2**2 - dist1**2

        s = self.s[-1]

        return np.clip(np.exp(-n_sq / ((self.bw * s)**2 + 1/(np.pi * frequency))), 0, 1)

    def propagate(self, delta_t: float, receiver: hf.Cartesian):
        """

        :param delta_t:
        :param receiver:
        :return:
        """
        received = False
        kill = False

        while not (received or kill):
            vel, direction, pos, delta_s = self.ray_step(delta_t)
            received = self.check_reception(receiver, delta_s)

            if self.t[-1] > .25:
                kill = True

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
    phi, theta, fail, pd = hf.uniform_spherical_grid(2048)

    x = 20.5 * np.cos(theta) * np.sin(phi)
    y = 20.5 * np.sin(theta) * np.sin(phi)
    z = 20.5 * np.cos(phi)

    offset = hf.Cartesian(0, 0, -35.5)
    rec = hf.Cartesian(0, -35.5 - 20.5, -1.7)
    # rec = hf.Cartesian(0, -500, -1.7)

    f = np.linspace(1, 44.1e3, 512)
    spec = np.empty((len(x), 512))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    startpt = [hf.Cartesian(x[i], y[i], z[i]) for i in range(len(x))]

    prop_queue = queue.Queue()
    prop_done = queue.Queue()

    for pi, p_init in enumerate(startpt):
        c_init = p_init * atm.get_speed_of_sound(0) / p_init.len()
        soundray = SoundRay(p_init + offset, c_init, 0, 0, pd, atm)

        prop_queue.put(soundray)

    p_thread = hf.ProgressThread(prop_queue.qsize())
    p_thread.start()
    threads = (PropagationThread(prop_queue, prop_done, .01, rec, p_thread) for i in range(64))
    [thread.start() for thread in threads]
    while threading.active_count() > 2:
        pass

    p_thread.stop()

    i = 0
    while not prop_done.empty():
        soundray = prop_done.get()

        spec[i] = soundray.gaussian_reception(f, rec)
        i += 1

        pos_arr = soundray.pos_array()

        ax.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')

    ax.scatter(*rec.vec)
    plt.show()

    plt.plot(f, spec.T)
    plt.show()

    # atm = hf.Atmosphere(1, 0, )
    # c = atm.get_speed_of_sound(0)
    # p_init = hf.Cartesian(0, 0, -10)
    # c_init = c * hf.Cartesian(1, 1, 0) / hf.Cartesian(1, 1, 0).len()
    # soundray = SoundRay(p_init, c_init, 0, 0, 0, atm)
    #
    # soundray.propagate(.01, hf.Cartesian(200, 0, 0))
    # print(soundray.pos[-1])
