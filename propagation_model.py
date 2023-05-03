import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import queue
import time

import helper_functions as hf


"""
========================================================================================================================
===                                                                                                                  ===
===                                                                                                                  ===
===                                                                                                                  ===
========================================================================================================================
"""


class PropagationThread(threading.Thread):
    def __init__(self, in_queue: queue.Queue, out_queue: queue.Queue, delta_t: float, receiver: hf.Cartesian,
                 p_thread: hf.ProgressThread, t_lim: float = 1.) -> None:
        """
        ================================================================================================================
        Subclass of threading.Thread to allow multiprocessing of the SoundRay.propagate function
        ================================================================================================================
        :param in_queue:
        :param out_queue:
        :param delta_t:
        :param receiver
        :param p_thread
        :param t_lim
        """
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue

        self.delta_t = delta_t
        self.receiver = receiver

        self.t_lim = t_lim

        self.p_thread = p_thread

    def run(self) -> None:
        """
        Propagates all SoundRays in the in_queue
        """
        # Loop as long as the input queue has SoundRays
        while threading.main_thread().is_alive() and not self.in_queue.empty():
            # Take a SoundRay from the queue
            ray: Ray = self.in_queue.get()
            # Propagate this Ray with given parameters
            ray.propagate(self.delta_t, self.receiver, self.t_lim)
            # When that is done, put the ray in the output queue
            self.out_queue.put(ray)
            # Update the progress thread so the counter goes up
            self.p_thread.update()

        # Check for ctrl+c type situations...
        if not threading.main_thread().is_alive():
            print(f'Stopped {self} after Interupt of MainThread')


class Ray:
    def __init__(self, pos_0: hf.Cartesian, vel_0: hf.Cartesian, s_0: float, beam_width: float,
                 atmosphere: hf.Atmosphere, t_0: float = 0.) -> None:
        """
        ================================================================================================================
        Class for the propagation sound ray model.
        ================================================================================================================
        :param pos_0: initial position in cartesian coordinates (m, m, m)
        :param vel_0: initial velocity in cartesian coordinates (m/s, m/s, m/s)
        :param s_0: initial beam length (m)
        :param beam_width: initial beam width angle (rad)
        :param atmosphere:
        :param t_0:
        """
        # Set initial conditions
        self.pos = np.array([pos_0, ])
        self.vel = np.array([vel_0, ])
        self.dir = np.array([vel_0, ])
        self.t = np.array([t_0, ])
        self.s = np.array([s_0])
        self.received = False
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
        if direction.len() > 0:
            vel: hf.Cartesian = wind_speed * hf.Cartesian(0, 1, 0) + speed_of_sound * direction / direction.len()
        else:
            vel: hf.Cartesian = wind_speed * hf.Cartesian(0, 1, 0)

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
        pos_new = self.pos[-1] + vel * delta_t
        # Check for reflections and if so: invert z-coordinate and z-velocity and z-direction
        if pos_new[2] >= 0:
            pos_new[2] = -pos_new[2]
            vel[2] = -vel[2]
            direction[2] = -direction[2]

        # Calculate ray path travel
        delta_s = (vel * delta_t).len()

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
        # Store new velocity and direction
        self.vel = np.append(self.vel, (vel,))
        self.dir = np.append(self.dir, (direction,))
        # Store new position
        self.pos = np.append(self.pos, (pos, ))
        # Propagate travelled distance
        self.s = np.append(self.s, self.s[-1] + delta_s)

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

        s = self.s[-2] + dist1

        return np.clip(np.exp(-n_sq / ((self.bw * s)**2 + 1/(np.pi * frequency))), 0, 1)

    def propagate(self, delta_t: float, receiver: hf.Cartesian, t_lim: float = 1.):
        """

        :param delta_t:
        :param receiver:
        :param t_lim:
        :return:
        """
        self.received = False
        kill = self.pos[0][2] >= 0

        while not (self.received or kill):
            vel, direction, pos, delta_s = self.ray_step(delta_t)
            self.received = self.check_reception(receiver, delta_s)

            if self.t[-1] > t_lim:
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


class SoundRay(Ray):
    def __init__(self, pos_0: hf.Cartesian, vel_0: hf.Cartesian, s_0: float, beam_width: float,
                 atmosphere: hf.Atmosphere, amplitude_spectrum: pd.DataFrame, t_0: float = 0., label: str = None):
        super().__init__(pos_0, vel_0, s_0, beam_width, atmosphere, t_0)

        self.label = label
        self.amplitude = amplitude_spectrum
        self.phase = pd.DataFrame(index=self.amplitude.index)

    def copy(self):
        """
        Create a copy of this Soundray
        """
        return SoundRay(self.pos[0], self.vel[0], self.s[0], self.bw, self.atmosphere, self.amplitude,
                        self.t[0], self.label)


class PropagationModel:
    def __init__(self, aur_conditions_dict: dict, aur_propagation_dict: dict, aur_receiver_dict: dict, ray_list: list):
        """
        ================================================================================================================

        ================================================================================================================
        :param aur_conditions_dict:
        :param aur_propagation_dict:
        :param ray_list:
        """
        self.conditions_dict = aur_conditions_dict
        self.params = aur_propagation_dict
        self.receivers = aur_receiver_dict
        self.ray_list = ray_list

    def run_receiver(self, receiver_idx: int, receiver_pos: hf.Cartesian, emission_duration: float):
        in_queue = queue.Queue()
        out_queue = queue.Queue()

        p_thread_0 = hf.ProgressThread(len(self.ray_list), 'Reading sound rays')
        p_thread_0.start()
        ray_dataframe: pd.DataFrame
        for ray_dataframe in self.ray_list:
            p_thread_0.update()
            for t in ray_dataframe.index:
                for blade in ray_dataframe.columns:
                    in_queue.put(ray_dataframe.loc[t, blade].copy())

        p_thread_0.stop()

        t_limit = 2 * receiver_pos.dist(self.conditions_dict['hub_pos']) / hf.c + emission_duration

        p_thread = hf.ProgressThread(in_queue.qsize(), f'Propagating to receiver {receiver_idx}')
        p_thread.start()
        threads = [PropagationThread(in_queue, out_queue, self.params['delta_t'], receiver_pos, p_thread, t_limit)
                   for _ in range(self.params['n_threads'])]

        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        p_thread.stop()

        return out_queue

    def run(self, emission_duration: float, which: int = -1, ):
        """
        :param emission_duration:
        :param which:
        """
        if which == -1:
            raise NotImplementedError('Multiple observer running not implemented yet!')
            # for receiver_idx, receiver_pos in self.receivers.items():
            #     self.run_receiver(receiver_idx, receiver_pos, emission_duration)

        elif which >= 0:
            return self.run_receiver(which, self.receivers[which], emission_duration)

        else:
            raise ValueError("Parameter 'which' should be: which >= 0 or which == -1")


if __name__ == '__main__':
    atm = hf.Atmosphere(35.5, 10.5, )
    phi, theta, fail, pd = hf.uniform_spherical_grid(2048)

    r = 41 / 2
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)

    offset = hf.Cartesian(0, 0, -35.5)
    # rec = hf.Cartesian(0, -35.5 - 20.5, -1.7)
    rec = hf.Cartesian(0, -500, -1.7)

    tlim = abs(2 * rec[1] / hf.c)

    f = np.linspace(1, 44.1e3, 512)
    spec = np.empty((len(x), 512))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    startpt = [hf.Cartesian(x[i], y[i], z[i]) for i in range(len(x)) if y[i] <= 0]

    prop_queue = queue.Queue()
    prop_done = queue.Queue()

    for pi, p_init in enumerate(startpt):
        c_init = p_init * atm.get_speed_of_sound(0) / p_init.len()
        soundray = Ray(p_init + offset, c_init, 0, pd, atm)

        prop_queue.put(soundray)

    p_thread = hf.ProgressThread(prop_queue.qsize(), "Propagating Rays")
    p_thread.start()
    threads = [PropagationThread(prop_queue, prop_done, .01, rec, p_thread, t_lim=tlim) for _ in range(32)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]

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

    # plt.plot(f, spec.T)
    # plt.show()

    # atm = hf.Atmosphere(1, 0, )
    # c = atm.get_speed_of_sound(0)
    # p_init = hf.Cartesian(0, 0, -10)
    # c_init = c * hf.Cartesian(1, 1, 0) / hf.Cartesian(1, 1, 0).len()
    # soundray = SoundRay(p_init, c_init, 0, 0, 0, atm)
    #
    # soundray.propagate(.01, hf.Cartesian(200, 0, 0))
    # print(soundray.pos[-1])
