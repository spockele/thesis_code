import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import compress_pickle as pickle

from typing import Self
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

import helper_functions as hf
from reception_model import Receiver


"""
========================================================================================================================
===                                                                                                                  ===
=== The propagation model for this auralisation tool                                                                 ===
===                                                                                                                  ===
========================================================================================================================
"""
__all__ = ['Ray', 'SoundRay', 'PropagationModel', ]


class Ray:
    def __init__(self, pos_0: hf.Cartesian, vel_0: hf.Cartesian, s_0: float, source_pos: hf.Cartesian,
                 t_0: float = 0.) -> None:
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

        self.source_pos = source_pos

    def __repr__(self):
        return f'<Ray at {self.pos[-1]}, received={self.received}>'

    def update_ray_velocity(self, delta_t: float, atmosphere: hf.Atmosphere) -> (hf.Cartesian, hf.Cartesian):
        """
        Update the ray velocity and direction forward by time step delta_t.
        :param delta_t: time step (s)
        :param atmosphere: the hf.Atmosphere to propagate the Ray through
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
        Update the ray position forward by time step delta_t and determine path step delta_s.
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
        Logic for time stepping the sound rays forward one time step.
        :param delta_t: time step (s)
        :param atmosphere: the hf.Atmosphere to propagate the Ray through
        :return: updated Cartesian velocity (m/s, m/s, m/s) and direction (m/s, m/s, m/s) vectors of the sound ray,
                 updated sound ray position (m, m, m), and the ray path step (m)
        """
        # Run the above functions to update velocity and position
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

    def check_reception(self, receiver: Receiver, delta_s: float) -> bool:
        """
        Check if the ray went past a receiver point.
        :param receiver: instance of rm.Receiver
        :param delta_s: the last ray path step (m)
        :return: boolean indicating whether the receiver has been passed
        """
        # Cannot check if less than two points
        if self.pos.size < 2:
            return False
        # Determine perpendicular planes with last 2 points
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

    def propagation_effects(self, atmosphere: hf.Atmosphere) -> None:
        """
        TO BE OVERWRITTEN: accommodation for sound propagation effects functionality in subclasses.
        :param atmosphere: the hf.Atmosphere to propagate the Ray through
        """
        pass

    def propagate(self, delta_t: float, receiver: Receiver, atmosphere: hf.Atmosphere, t_lim: float) -> None:
        """
        Propagate the Ray until received or kill condition is reached.
        :param delta_t: time step (s)
        :param receiver: instance of rm.Receiver
        :param atmosphere: the hf.Atmosphere to propagate the Ray through
        :param t_lim: propagation time limit (s)
        """
        # Set initial loop conditions
        self.received = False
        kill = self.pos[0][2] >= 0
        # Add initial propagation effects
        self.propagation_effects(atmosphere)

        # Loop while condition holds
        while not (self.received or kill):
            # Step the ray forward
            vel, direction, pos, delta_s = self.ray_step(delta_t, atmosphere)
            # Check if it is received at the given receiver
            self.received = self.check_reception(receiver, delta_s)
            # Add propagation effects
            self.propagation_effects(atmosphere)

            # Kill the ray when time limit is exceeded
            kill = self.t[-1] - self.t[0] > t_lim

    def pos_array(self) -> np.array:
        """
        Convert position history to an easily readable array of shape (self.pos.size, 3).
        :return: numpy array with full position history unpacked
        """
        # Create empty initial array
        arr = np.zeros((3, self.pos.size))
        # Loop over the list of coordinates
        for pi, pos in enumerate(self.pos):
            # Fill the position in the array
            arr[:, pi] = pos.vec

        return arr


class SoundRay(Ray):
    def __init__(self, pos_0: hf.Cartesian, vel_0: hf.Cartesian, s_0: float, source_pos: hf.Cartesian,
                 beam_width: float, amplitude_spectrum: pd.DataFrame, models: tuple,
                 t_0: float = 0., label: str = None) -> None:
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
        super().__init__(pos_0, vel_0, s_0, source_pos, t_0)

        self.bw = beam_width
        self.label = label
        self.models = models

        # Initialise the sound spectrum
        self.spectrum = pd.DataFrame(amplitude_spectrum)
        # Set the column name of the given spectrum to 'a' for amplitude
        self.spectrum.columns = ['a']
        # Initialise phase and the attenuation effects
        self.spectrum['p'] = 0.
        self.spectrum['gaussian'] = 1.
        self.spectrum['spherical'] = 1.
        self.spectrum['atmospheric'] = 1.

    def copy(self) -> Self:
        """
        Create a not-yet-propagated copy of this SoundRay.
        """
        return SoundRay(self.pos[0], self.vel[0], self.s[0], self.source_pos, self.bw, self.spectrum['a'],
                        self.models, self.t[0], self.label)

    def propagation_effects(self, atmosphere: hf.Atmosphere) -> None:
        """
        Add atmospheric absorption and spherical spreading to the sound spectrum.
        :param atmosphere: the hf.Atmosphere to propagate the SoundRay through
        """
        # Get current temperature, pressure and speed of sound from the atmosphere
        t_current, p_current, _, c_current, _ = atmosphere.get_conditions(-self.pos[-1][2])

        if 'spherical' in self.models and self.t.size >= 2:
            c_previous = atmosphere.get_speed_of_sound(-self.pos[-2][2])
            # Spherical spreading factor
            self.spectrum['spherical'] /= (self.s[-1] / self.s[-2]) * np.sqrt(c_previous / c_current)

        if 'atmospheric' in self.models:
            # Reference values
            t_0, p_0 = 293.15, 101325
            # Extract frequencies from spectrum
            f = self.spectrum.index
            # Determine saturation pressure
            p_sat = 622.2 * np.exp(17.67 * (t_current - 273.15) / (t_current - 29.65))
            # Determine absolute humidity from relative
            humidity_abs = atmosphere.humidity * p_sat / p_current
            # Determine gas resonance frequencies
            f_rn = (p_current / p_0) * ((t_0 / t_current) ** .5) * (
                    9. + 280. * humidity_abs * np.exp(-4.17 * ((t_0 / t_current) ** (1 / 3) - 1)))
            f_ro = (p_current / p_0) * (24. + 4.04e4 * humidity_abs * (.02 + humidity_abs) / (.391 + humidity_abs))

            # Determine the individual terms of the absorption coefficient
            term_1 = 1.84e-11 / ((p_current / p_0) * (t_0 / t_current) ** .5)
            term_2 = .1068 * np.exp(-3352 / t_current) * f_rn / (f ** 2 + f_rn ** 2)
            term_3 = .01278 * np.exp(-2239.1 / t_current) * f_ro / (f ** 2 + f_ro ** 2)
            # Determine the absorption coefficient
            alpha = (term_1 + (term_2 + term_3) * (t_0 / t_current) ** 2.5) * f**2

            delta_s = self.s[-1] - self.s[-2] if self.t.size >= 2 else self.s[-1]

            # More absorption :)
            self.spectrum['atmospheric'] *= np.exp(-alpha * delta_s / 2)

    def gaussian_factor(self, receiver: Receiver) -> None:
        """
        Calculate the Gaussian beam reception transfer function.
        :param receiver: an instance of Receiver
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

    def receive(self, receiver: Receiver) -> (float, pd.DataFrame, hf.Cartesian):
        """
        Function that adds the SoundRay to the receiver.
        :param receiver: instance of Receiver
        """
        # Determine the Gaussian beam attenuation spectrum
        self.gaussian_factor(receiver)
        # Take the amplitude and phase spectra
        spectrum = self.spectrum[['a', 'p']].copy()
        # Add the Gaussian beam attenuation
        spectrum['a'] *= self.spectrum['gaussian'] ** .5
        # Add attenuation from selected models
        for model in self.models:
            spectrum['a'] *= self.spectrum[model]

        # Return what is needed to create a ReceivedSound instance
        return self.t[-1], spectrum, self.source_pos


class PropagationModel:
    def __init__(self, aur_conditions_dict: dict, aur_propagation_dict: dict, atmosphere: hf.Atmosphere) -> None:
        """
        ================================================================================================================
        Class that manages the whole propagation model
        ================================================================================================================
        :param aur_conditions_dict: conditions_dict from the Case class
        :param aur_propagation_dict: propagation_dict from the Case class
        :param atmosphere: the hf.Atmosphere to propagate the SoundRays through
        """
        self.conditions_dict = aur_conditions_dict
        self.params = aur_propagation_dict
        self.atmosphere = atmosphere

    def run(self, receiver: Receiver, in_queue: list[SoundRay]) -> None:
        """
        Run the propagation model for one receiver.
        :param receiver: instance of rm.Receiver
        :param in_queue: queue.Queue instance containing non-propagated SoundRays
        :return: A queue.Queue instance containing all propagated SoundRays.
        """
        # Set the time limit to limit compute time
        t_limit = 3 * receiver.dist(self.conditions_dict['hub_pos']) / hf.c

        # Start a ProgressThread to follow the propagation
        p_thread = hf.ProgressThread(len(in_queue), f'Propagating rays')
        p_thread.start()

        for ray in in_queue:
            ray.propagate(self.conditions_dict['delta_t'], receiver, self.atmosphere, t_limit)
            p_thread.update()

        # Stop the ProgressThread
        p_thread.stop()
        del p_thread

    @staticmethod
    def pickle_ray_queue(ray_queue: list[SoundRay], ray_cache_path: str) -> None:
        """
        Cache a queue of Rays to compressed pickles in the given directory.
        :param ray_queue: queue.Queue containing Rays (or SoundRays)
        :param ray_cache_path: path to store the pickles to
        """
        # Check the existence of the cache directory
        if not os.path.isdir(ray_cache_path):
            os.mkdir(ray_cache_path)

        # Start a ProgressThread
        p_thread = hf.ProgressThread(len(ray_queue), 'Pickle-ing sound rays')
        p_thread.start()

        # Loop over the queue without emptying it
        ray: SoundRay
        for ray in ray_queue:
            # Open the pickle file
            ray_file = open(os.path.join(ray_cache_path, f'SoundRay_{p_thread.step}.pickle.gz'), 'wb')
            # Dump the pickle
            pickle.dump(ray, ray_file)
            # Close the pickle file
            ray_file.close()
            # Update the ProgressThread
            p_thread.update()

        # Stop the ProgressThread
        p_thread.stop()
        del p_thread

    @staticmethod
    def unpickle_ray_queue(ray_cache_path: str) -> list[SoundRay]:
        """
        Read out a cache directory of compressed Ray pickles.
        :param ray_cache_path: path to directory containing pickles
        """
        # Check the existence of the cache directory
        if not os.path.isdir(ray_cache_path):
            raise NotADirectoryError(f'No {ray_cache_path} directory found. Cannot un-pickle ray queue.')

        # Create empty queue.Queue
        ray_queue = list[SoundRay]()
        # Get paths of all files in the directory with ".pickle" in them
        ray_paths = [ray_path for ray_path in os.listdir(ray_cache_path) if '.pickle' in ray_path]

        # Check the existence of pickles
        if len(ray_paths) <= 0:
            raise FileNotFoundError(f'No pickles found in {ray_cache_path}. Cannot un-pickle ray queue.')

        # Start a ProgressThread
        p_thread = hf.ProgressThread(len(ray_paths), 'Un-pickle-ing sound rays')
        p_thread.start()
        # Loop over the pickles
        for ray_path in ray_paths:
            # Open the jar
            ray_file = open(os.path.join(ray_cache_path, ray_path), 'rb')
            # Take the pickle
            ray_queue.append(pickle.load(ray_file))
            # Close the jar
            ray_file.close()
            # Update the ProgressThread
            p_thread.update()

        # Stop the ProgressThread
        p_thread.stop()
        del p_thread

        return ray_queue

    @staticmethod
    def interactive_ray_plot(ray_queue: list[SoundRay], receiver: Receiver, gif_out: str = None) -> None:
        """
        Makes an interactive plot of the SoundRays in a queue.Queue.
        :param ray_queue: queue.Queue containing SoundRays
        :param receiver: instance of rm.Receiver with which the SoundRays where propagated
        :param gif_out: A path to output a gif animation to
        """
        # Create an empty ray dictionary
        rays = dict[float: list]()
        # Loop over the ray_queue without removing the SoundRays
        for ray in ray_queue:
            # Add any received rays to the dictionary
            if ray.received:
                # Just stupid sh!te to avoid floating point errors...
                t = round(ray.t[0], 10)
                # Fill into the dictionary
                if t in rays.keys():
                    rays[t].append(ray)
                else:
                    rays[t] = list[SoundRay]([ray, ])

        def update_plot(t_plt: float):
            """
            Internal function to update the interactive plot with the Slider.
            :param t_plt: time input (s)
            """
            # Clear the plot
            ax.clear()
            # Re-set the axis limits and aspect ratio
            ax.set_aspect('equal')
            ax.set_xlim(-75, 75)
            ax.set_ylim(-75, 75)
            ax.set_zlim(0, 150)
            # Re-set the labels
            ax.set_xlabel('-x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('-z (m)')
            # Put the receiver point in there
            ax.scatter(-receiver[0], receiver[1], -receiver[2])

            # Get the rays at input time
            ray_lst = list[SoundRay](rays[t_plt])
            # Loop over all SoundRays at input time
            for ry in ray_lst:
                # Obtain the pos_array for nicer plotting
                pos_array = ry.pos_array()

                # Get the sound spectrum
                _, spectrum, source_pos = ry.receive(receiver)
                # Integrate to get the energy
                energy = np.trapz(spectrum['a'], spectrum.index)
                # Check that energy is not zero before continuing
                if energy > 0:
                    # dB that energy
                    energy = 10 * np.log10(energy / hf.p_ref ** 2)
                    # Bin the energy
                    energy_bin = 10 * int(energy // 10) + 5
                    # Apply color to that energy
                    cmap_lvl = float(np.argwhere(levels == np.clip(energy_bin, 5, 95))) / (levels.size - 1)
                    color = cmap(cmap_lvl)

                    # Plot the ray
                    ax.plot(-pos_array[0], pos_array[1], -pos_array[2], color=color)

                    x, y, z = source_pos.vec
                    x_0, y_0, z_0 = ry.pos[0].vec

                    ax.scatter(-x, y, -z, color='k', marker='8')
                    ax.plot((-x, -x_0), (y, y_0), (-z, -z_0), color='0.8', )

        def colorbar():
            """
            Create the colorbar for the energy levels
            """
            # Set the ticks and create an axis on the figure for the colorbar
            ticks = np.arange(0, 100 + 10, 10)
            ax_cbar = fig.add_axes([.85, 0.1, 0.05, 0.8])

            norm = mpl.colors.BoundaryNorm(ticks, cmap.N)
            plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                         cax=ax_cbar,
                         extend='both',
                         orientation='vertical',
                         label='Received OSPL of Sound Ray (dB) (Binned per 10 dB)'
                         )

        # Create the main plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Pre-set the colorbar levels and colormap
        levels = np.arange(5, 95 + 10, 10)
        cmap = mpl.colormaps['viridis'].resampled(10)

        # Adjust the main plot to make room for the colorbar
        fig.subplots_adjust(left=0., bottom=0.1, right=0.85, top=1.)
        colorbar()

        # Preset the valstep parameter for the Slider, and to determine the frames of the animation
        valstep = list(sorted(rays.keys()))

        # Create an animated GIF file if so desired
        if gif_out is not None:
            ani = FuncAnimation(fig=fig, func=update_plot, frames=valstep, interval=valstep[1] - valstep[0])
            ani.save(gif_out, writer='pillow')

            # Create the figure again to avoid issues with the animation stuff
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            colorbar()

        # Adjust the main plot to make room for the colorbar
        fig.subplots_adjust(left=0., bottom=0.2, right=0.85, top=1.)

        # Make a horizontal slider to control the time.
        ax_time = fig.add_axes([0.11, 0.1, 0.65, 0.05])
        slider = Slider(
            ax=ax_time,
            label='Time (s)',
            valmin=valstep[0],
            valmax=valstep[-1],
            valstep=valstep,
            valinit=valstep[0],
        )
        # Set the initial plot at the first available time step
        update_plot(valstep[0])
        # Set the slider update function
        slider.on_changed(update_plot)

        # Show the plot
        plt.show()
