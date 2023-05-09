import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import queue
import time

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
                 p_thread: hf.ProgressThread, t_lim: float = 1.) -> None:
        """
        ================================================================================================================
        Subclass of threading.Thread to allow multiprocessing of the SoundRay.propagate function
        ================================================================================================================
        :param in_queue: queue.Queue instance containing non-propagated SoundRays
        :param out_queue: queue.Queue instance where propagated SoundRays will be put
        :param delta_t: time step (s)
        :param receiver: the receiver point in Cartesian coordinates (m, m, m).
        :param p_thread: ProgressThread instance to track progress of the propagation model
        :param t_lim: propagation time limit (s)
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
        :param atmosphere: atmosphere defined in hf.Atmosphere()
        :param t_0: the start time of the ray propagation (s)
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

    def check_reception(self, receiver: rm.Receiver, delta_s: float):
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

    def propagate(self, delta_t: float, receiver: rm.Receiver, t_lim: float = 1.):
        """
        Propagate the Ray until received or kill condition is reached
        :param delta_t: time step (s)
        :param receiver: the receiver point in Cartesian coordinates (m, m, m)
        :param t_lim: propagation time limit (s)
        """
        self.received = False
        kill = self.pos[0][2] >= 0

        while not (self.received or kill):
            vel, direction, pos, delta_s = self.ray_step(delta_t)
            self.received = self.check_reception(receiver, delta_s)

            if self.t[-1] - self.t[0] > t_lim:
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
        """
        ================================================================================================================
        Class for the propagation sound ray model. With the sound spectral effects.
        ================================================================================================================
        :param pos_0: initial position in cartesian coordinates (m, m, m)
        :param vel_0: initial velocity in cartesian coordinates (m/s, m/s, m/s)
        :param s_0: initial beam length (m)
        :param beam_width: initial beam width angle (rad)
        :param atmosphere: atmosphere defined in hf.Atmosphere()
        :param t_0: the start time of the ray propagation (s)
        :param label: a string label for SoundRay
        """
        super().__init__(pos_0, vel_0, s_0, beam_width, atmosphere, t_0)

        self.label = label
        self.spectrum = pd.DataFrame(amplitude_spectrum)
        self.spectrum['p'] = 0.
        self.spectrum['gaussian'] = 0.
        self.spectrum.columns = ['a', 'p', 'gaussian']

    def copy(self):
        """
        Create a copy of this Soundray
        """
        return SoundRay(self.pos[0], self.vel[0], self.s[0], self.bw, self.atmosphere, self.spectrum['a'],
                        self.t[0], self.label)

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
        self.gaussian_factor(receiver)
        received_sound = rm.ReceivedSound(self.t[-1], receiver.rotation, self.dir[-1], self.spectrum)
        receiver.receive(received_sound)


class PropagationModel:
    def __init__(self, aur_conditions_dict: dict, aur_propagation_dict: dict, aur_receiver_dict: dict, ray_queue: queue.Queue):
        """
        ================================================================================================================
        Class that manages the whole propagation model
        ================================================================================================================
        :param aur_conditions_dict: conditions_dict from the Case class
        :param aur_propagation_dict: propagation_dict from the Case class
        :param ray_queue: queue with SoundRay instances to propagate
        """
        self.conditions_dict = aur_conditions_dict
        self.params = aur_propagation_dict
        self.receivers = aur_receiver_dict
        self.ray_queue = ray_queue

    def run_receiver(self, receiver_key: int, receiver_pos: rm.Receiver):
        """
        Run the propagation model for one receiver
        :param receiver_key: key of the receiver in self.receivers
        :param receiver_pos: the receiver point in Cartesian coordinates (m, m, m)
        :return: a queue.Queue instance containing all propagated SoundRays
        """
        # Initialise the output queue.Queue()s
        out_queue = queue.Queue()

        # Set the time limit to limit compute time
        t_limit = 3 * receiver_pos.dist(self.conditions_dict['hub_pos']) / hf.c

        # Start a ProgressThread to follow the propagation
        p_thread = hf.ProgressThread(self.ray_queue.qsize(), f'Propagating to receiver {receiver_key}')
        p_thread.start()
        # Create the PropagationThreads
        threads = [PropagationThread(self.ray_queue, out_queue, self.params['delta_t'], receiver_pos, p_thread, t_limit)
                   for _ in range(self.params['n_threads'])]
        # Start the threads and hold until all are done
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        # Stop the ProgressThread
        p_thread.stop()

        return out_queue

    def run(self, which: int = -1):
        """
        Run the propagation model
        :param which: key of the receiver to which to propagate the sound
        """
        # Run for all receivers
        if which == -1:
            raise NotImplementedError('Multiple observer running not implemented yet!')
            # for receiver_idx, receiver_pos in self.receivers.items():
            #     self.run_receiver(receiver_idx, receiver_pos)

        # Run for specified receiver
        elif which >= 0:
            return self.run_receiver(which, self.receivers[which])

        # We don't do negative numbers here
        else:
            raise ValueError("Parameter 'which' should be: which >= 0 or which == -1")
