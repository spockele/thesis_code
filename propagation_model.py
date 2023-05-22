import os
import queue
import warnings
import threading
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import compress_pickle as pickle

from matplotlib.widgets import Slider

import helper_functions as hf
import reception_model as rm


"""
========================================================================================================================
===                                                                                                                  ===
=== The propagation model for this auralisation tool                                                                 ===
===                                                                                                                  ===
========================================================================================================================
"""


class PropagationThread(threading.Thread):
    def __init__(self, in_queue: queue.Queue, out_queue: queue.Queue, delta_t: float, receiver: rm.Receiver,
                 atmosphere: hf.Atmosphere, p_thread: hf.ProgressThread, t_lim: float = 1.) -> None:
        """
        ================================================================================================================
        Subclass of threading.Thread to allow multiprocessing of the SoundRay.propagate function
        ================================================================================================================
        :param in_queue: queue.Queue instance containing non-propagated SoundRays
        :param out_queue: queue.Queue instance where propagated SoundRays will be put
        :param delta_t: time step (s)
        :param receiver: the receiver point in Cartesian coordinates (m, m, m).
        :param atmosphere:
        :param p_thread: ProgressThread instance to track progress of the propagation model
        :param t_lim: propagation time limit (s)
        """
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue

        self.delta_t = delta_t
        self.receiver = receiver
        self.atmosphere = atmosphere

        self.t_lim = t_lim

        self.p_thread = p_thread

    def run(self) -> None:
        """
        Propagates all SoundRays in the in_queue
        """
        # Loop as long as the input queue has SoundRays
        while threading.main_thread().is_alive() and not self.in_queue.empty():
            # Take a SoundRay from the queue
            ray: SoundRay = self.in_queue.get()
            # Propagate this Ray with given parameters
            ray.propagate(self.delta_t, self.receiver, self.atmosphere, self.t_lim)
            # When that is done, put the ray in the output queue
            self.out_queue.put(ray)
            # Update the progress thread so the counter goes up
            self.p_thread.update()

        # Check for ctrl+c type situations...
        if not threading.main_thread().is_alive():
            print(f'Stopped {self} after Interupt of MainThread')


class Ray:
    def __init__(self, pos_0: hf.Cartesian, vel_0: hf.Cartesian, s_0: float, t_0: float = 0.) -> None:
        """
        ================================================================================================================
        Class for the propagation sound ray model.
        ================================================================================================================
        :param pos_0: initial position in cartesian coordinates (m, m, m)
        :param vel_0: initial velocity in cartesian coordinates (m/s, m/s, m/s)
        :param s_0: initial beam length (m)
        :param t_0: the start time of the ray propagation (s)
        """
        # Set initial conditions
        self.pos = np.array([pos_0, ])
        self.vel = np.array([vel_0, ])
        self.dir = np.array([vel_0, ])
        self.t = np.array([t_0, ])
        self.s = np.array([s_0])
        self.received = False

    def __repr__(self):
        return f'<Ray at {self.pos[-1]}, received={self.received}>'

    def update_ray_velocity(self, delta_t: float, atmosphere: hf.Atmosphere) -> (hf.Cartesian, hf.Cartesian):
        """
        Update the ray velocity and direction forward by time step delta_t
        :param delta_t: time step (s)
        :param atmosphere:
        :return: Updated Cartesian velocity (m/s, m/s, m/s) and direction (m/s, m/s, m/s) vectors of the sound ray
        """
        # Get height from last position
        height = -self.pos[-1][2]
        # Determine direction change
        speed_of_sound_gradient = atmosphere.get_speed_of_sound_gradient(height)
        direction_change: hf.Cartesian = self.vel[-1].len() * hf.Cartesian(0, 0, speed_of_sound_gradient)
        direction: hf.Cartesian = self.dir[-1] + direction_change * delta_t

        # Get wind speed and speed of sound at height
        wind_speed = atmosphere.get_wind_speed(height)
        speed_of_sound = atmosphere.get_speed_of_sound(height)

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

    def ray_step(self, delta_t: float, atmosphere: hf.Atmosphere) -> (hf.Cartesian, hf.Cartesian, hf.Cartesian, float):
        """
        Logic for time stepping the sound rays forward one time step
        :param delta_t: time step (s)
        :param atmosphere:
        :return: Updated Cartesian velocity (m/s, m/s, m/s) and direction (m/s, m/s, m/s) vectors of the sound ray,
                 updated sound ray position (m, m, m), and the ray path step (m)
        """
        vel, direction = self.update_ray_velocity(delta_t, atmosphere)
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

    def check_reception(self, receiver: rm.Receiver, delta_s: float):
        """
        Check if the ray went past a receiver point.
        :param receiver: The receiver point in Cartesian coordinates (m, m, m).
        :param delta_s: The last ray path step (m).
        :return: Boolean indicating whether the receiver has been passed.
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

    def propagate(self, delta_t: float, receiver: rm.Receiver, atmosphere: hf.Atmosphere, t_lim: float = 1.):
        """
        Propagate the Ray until received or kill condition is reached
        :param delta_t: time step (s)
        :param receiver: the receiver point in Cartesian coordinates (m, m, m)
        :param atmosphere:
        :param t_lim: propagation time limit (s)
        """
        self.received = False
        kill = self.pos[0][2] >= 0

        while not (self.received or kill):
            vel, direction, pos, delta_s = self.ray_step(delta_t, atmosphere)
            self.received = self.check_reception(receiver, delta_s)

            if self.t[-1] - self.t[0] > t_lim:
                kill = True

    def pos_array(self) -> np.array:
        """
        Convert position history to an easily readable array of shape (self.pos.size, 3)
        :return: numpy array with full position history unpacked
        """
        arr = np.zeros((self.pos.size, 3))
        for pi, pos in enumerate(self.pos):
            arr[pi] = pos.vec

        return arr


class SoundRay(Ray):
    def __init__(self, pos_0: hf.Cartesian, vel_0: hf.Cartesian, s_0: float, beam_width: float,
                 amplitude_spectrum: pd.DataFrame, t_0: float = 0., label: str = None):
        """
        ================================================================================================================
        Class for the propagation sound ray model. With the sound spectral effects.
        ================================================================================================================
        :param pos_0: initial position in cartesian coordinates (m, m, m)
        :param vel_0: initial velocity in cartesian coordinates (m/s, m/s, m/s)
        :param s_0: initial beam length (m)
        :param beam_width: initial beam width angle (rad)
        :param t_0: the start time of the ray propagation (s)
        :param label: a string label for SoundRay
        """
        super().__init__(pos_0, vel_0, s_0, t_0)

        self.label = label
        self.spectrum = pd.DataFrame(amplitude_spectrum)
        self.spectrum['p'] = 0.
        self.spectrum['gaussian'] = 0.
        self.spectrum.columns = ['a', 'p', 'gaussian']
        self.bw = beam_width

    def copy(self):
        """
        Create a not-yet-propagated copy of this SoundRay
        """
        return SoundRay(self.pos[0], self.vel[0], self.s[0], self.bw, self.spectrum['a'], self.t[0], self.label)

    def gaussian_factor(self, receiver: rm.Receiver):
        """
        Calculate the Gaussian beam reception transfer function
        :param receiver: an instance of rm.Receiver
        """
        # Determine the perpendicular plane just before the receiver
        plane = hf.PerpendicularPlane3D(self.pos[-1], self.pos[-2])
        # Determine distance from that plane to the receiver
        dist1 = plane.distance_to_point(receiver)
        # Determine the distance between the ray-plane intersection and the receiver
        dist2 = self.pos[-2].dist(receiver)
        # Determine the distance between the ray and the receiver
        n_sq = dist2**2 - dist1**2
        # Determine the distance along the ray to the intersecting line
        s = self.s[-2] + dist1

        # Determine the filter and clip to between 0 and 1
        self.spectrum['gaussian'] = np.clip(np.exp(-n_sq / ((self.bw * s)**2 + 1/(np.pi * self.spectrum.index))), 0, 1)

    def receive(self, receiver: rm.Receiver):
        """
        Function that adds the SoundRay to the receiver
        :param receiver: instance of rm.Receiver
        """
        # Determine the Gaussian beam attenuation spectrum
        self.gaussian_factor(receiver)
        # Create the received sound
        received_sound = rm.ReceivedSound(self.t[-1], receiver.rotation, self.dir[-1], self.spectrum)
        # Receive sound with the receiver
        receiver.receive(received_sound)


class PropagationModel:
    def __init__(self, aur_conditions_dict: dict, aur_propagation_dict: dict, atmosphere: hf.Atmosphere):
        """
        ================================================================================================================
        Class that manages the whole propagation model
        ================================================================================================================
        :param aur_conditions_dict: conditions_dict from the Case class
        :param aur_propagation_dict: propagation_dict from the Case class
        :param atmosphere:
        """
        self.conditions_dict = aur_conditions_dict
        self.params = aur_propagation_dict
        self.atmosphere = atmosphere

    def run(self, receiver_key: int, receiver_pos: rm.Receiver, in_queue: queue.Queue):
        """
        Run the propagation model for one receiver.
        :param receiver_key: Key of the receiver in self.receivers.
        :param receiver_pos: The receiver point in Cartesian coordinates (m, m, m).
        :param in_queue:
        :return: A queue.Queue instance containing all propagated SoundRays.
        """
        # Initialise the output queue.Queue()s
        out_queue = queue.Queue()

        # Set the time limit to limit compute time
        t_limit = 3 * receiver_pos.dist(self.conditions_dict['hub_pos']) / hf.c

        # Start a ProgressThread to follow the propagation
        p_thread = hf.ProgressThread(in_queue.qsize(), f'Propagating to receiver {receiver_key}')
        p_thread.start()
        # Create the PropagationThreads
        threads = [PropagationThread(in_queue, out_queue, self.conditions_dict['delta_t'],
                                     receiver_pos, self.atmosphere, p_thread, t_limit)
                   for _ in range(self.params['n_threads'])]
        # Start the threads and hold until all are done
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        # Stop the ProgressThread
        p_thread.stop()
        del p_thread

        return out_queue

    @staticmethod
    def pickle_ray_queue(ray_queue: queue.Queue) -> None:
        """
        TODO: PropagationModel.pickle_ray_queue > write docstring and comments
        TODO: PropagationModel.pickle_ray_queue > look into pickle of whole queue instead of 1 per ray
        :param ray_queue:
        :return:
        """
        ray_cache_path = os.path.abspath('ray_cache')
        if not os.path.isdir(ray_cache_path):
            os.mkdir(ray_cache_path)

        warnings.warn("Pickle files may take up a lot of storage! Requires up to ~1MB per ray!")

        p_thread = hf.ProgressThread(ray_queue.qsize(), 'Pickle-ing sound rays')
        p_thread.start()

        ray: SoundRay
        for ray in ray_queue.queue:
            ray_file = open(os.path.join(ray_cache_path, f'SoundRay_{p_thread.step}.pickle.gz'), 'wb')
            pickle.dump(ray, ray_file)
            ray_file.close()

            p_thread.update()

        p_thread.stop()
        del p_thread

    @staticmethod
    def unpickle_ray_queue() -> queue.Queue:
        """
        TODO: PropagationModel.unpickle_ray_queue > write docstring and comments
        :return:
        """
        ray_cache_path = os.path.abspath('ray_cache')
        if not os.path.isdir(ray_cache_path):
            raise NotADirectoryError(f'No ./ray_cache directory found. Cannot un-pickle ray queue')

        ray_queue = queue.Queue()
        ray_paths = [ray_path for ray_path in os.listdir(ray_cache_path) if '.pickle' in ray_path]

        p_thread = hf.ProgressThread(len(ray_paths), 'Un-pickle-ing sound rays')
        p_thread.start()

        for ray_path in ray_paths:
            ray_file = open(os.path.join(ray_cache_path, ray_path), 'rb')
            ray_queue.put(pickle.load(ray_file))
            ray_file.close()
            p_thread.update()

        p_thread.stop()
        del p_thread

        return ray_queue

    @staticmethod
    def interactive_ray_plot(ray_queue: queue.Queue, receiver: rm.Receiver, time_series: pd.DataFrame):
        """
        TODO: PropagationModel.interactive_ray_plot > write docstring and comments
        :param ray_queue:
        :param receiver:
        :return:
        """
        # raise NotImplementedError('Yeah, nah mate :/')
        rays = {}
        ray: SoundRay
        for ray in ray_queue.queue:
            if ray.received and ray.label != 'blade_0':
                t = round(ray.t[-1], 10)
                if t in rays.keys():
                    rays[t].append(ray)
                else:
                    rays[t] = [ray, ]

        # create the main plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        def update_plot(t_plt: float):
            """
            TODO: PropagationModel.interactive_ray_plot.update_plot > write docstring and comments
            :param t_plt:
            :return:
            """
            ray_lst = rays[t_plt]

            ax.clear()
            ax.set_aspect('equal')
            ax.set_xlim(-75, 75)
            ax.set_ylim(-75, 75)
            ax.set_zlim(0, 150)

            ax.set_xlabel('-x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('-z (m)')
            ax.scatter(-receiver[0], receiver[1], -receiver[2])
            for r in ray_lst:
                pos_array = r.pos_array()

                spectrum = r.spectrum['a'] * r.spectrum['gaussian']
                energy = np.trapz(spectrum, spectrum.index)
                if energy > 0:
                    energy = 10 * np.log10(energy / hf.p_ref ** 2)
                    energy_bin = 10 * int(energy // 10) + 5
                    cmap_lvl = np.float(np.argwhere(levels == np.clip(energy_bin, 5, 95))) / (levels.size - 1)

                    color = cmap(cmap_lvl)

                    ax.plot(-pos_array[:, 0], pos_array[:, 1], -pos_array[:, 2], color=color)

            for key in time_series.columns:
                if 'blade' in key and t_plt in time_series.index:
                    x, y, z = time_series.loc[t_plt, key].vec
                    ax.scatter(-x, y, -z, s=5, color='k', marker='8')

        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0., bottom=0.2, right=0.85, top=1.)

        # Make a horizontal slider to control the time.
        ax_time = fig.add_axes([0.11, 0.1, 0.65, 0.05])
        valstep = list(sorted(rays.keys()))
        slider = Slider(
            ax=ax_time,
            label='Time (s)',
            valmin=valstep[0],
            valmax=valstep[-1],
            valstep=valstep,
            valinit=valstep[0],
        )

        levels = np.arange(5, 95 + 10, 10)
        ticks = np.arange(0, 100 + 10, 10)
        cmap = mpl.colormaps['viridis'].resampled(10)

        ax_cbar = fig.add_axes([.85, 0.1, 0.05, 0.8])
        norm = mpl.colors.BoundaryNorm(ticks, cmap.N)
        plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                     cax=ax_cbar,
                     extend='both',
                     orientation='vertical',
                     label='Received OSPL of Sound Ray (dB) (Binned per 10 dB)'
                     )

        update_plot(valstep[0])
        slider.on_changed(update_plot)
        plt.show()
