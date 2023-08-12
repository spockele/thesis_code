import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

import helper_functions as hf
import propagation_model as pm
import reception_model as rm


"""
========================================================================================================================
===                                                                                                                  ===
=== The source model for this auralisation tool                                                                      ===
===                                                                                                                  ===
========================================================================================================================
"""
__all__ = ['H2Observer', 'H2Sphere', 'Source', 'SourceModel']


class H2Observer(hf.Cartesian):
    def __init__(self, file_path: str, scope: str, delta_t: float) -> None:
        """
        ================================================================================================================
        Class to wrap the HAWC2 result at a single observer point. Subclass of Cartesian for ease of use.
        ================================================================================================================
        :param file_path: path to the result file for this observer point
        :param scope: selects the noise model result to load ('All', 'TI', 'TE', 'ST', 'TP')
        :param delta_t: simulation time step (s) to interpolate the HAWC2 results to
        """
        # Read the given HAWC2 noise output file
        pos, time_series, psd = hf.read_hawc2_aero_noise(file_path, scope=scope)
        # Set the point position
        super().__init__(*pos)
        # Store the time series and the spectrograms
        self.time_series = time_series
        self.psd = psd

        # Set the initial time of the time series and spectrograms to 0
        # as the absolute time is irrelevant for this project
        self.time_series.index = self.time_series.index - self.time_series.index[0]
        self.psd[0].columns = self.time_series.index
        self.psd[1].columns = self.time_series.index
        self.psd[2].columns = self.time_series.index
        self.psd[3].columns = self.time_series.index

        # Interpolation to hrtf frequency resolution for better ground effect (I Hope...)
        for psd_idx in range(4):
            new_psd = pd.DataFrame(0., columns=self.psd[psd_idx].columns, index=rm.hrtf.f[rm.hrtf.f > 0])
            for t in self.psd[psd_idx].columns:
                new_psd.loc[:, t] = np.interp(rm.hrtf.f[rm.hrtf.f > 0],
                                              self.psd[psd_idx].index, self.psd[psd_idx].loc[:, t])

            self.psd[psd_idx] = new_psd

        # Interpolation to simulation time step
        sim_time = np.round(np.arange(self.time_series.index[0], self.time_series.index[-1] + delta_t, delta_t), 10)
        #  of the time series
        self.time_series.index = np.round(self.time_series.index, 10)
        self.time_series = self.time_series.reindex(index=sim_time)
        self.time_series.interpolate(axis='index', inplace=True)
        #  of the turbine and blades psd
        for psd_idx in range(4):
            self.psd[psd_idx].columns = np.round(self.psd[psd_idx].columns, 10)
            self.psd[psd_idx] = self.psd[psd_idx].reindex(columns=sim_time)
            self.psd[psd_idx].interpolate(axis='columns', inplace=True)

    def __repr__(self) -> str:
        return f'<HAWC2 Observer: {str(self)}>'


class H2Sphere(list[H2Observer]):
    def __init__(self, h2_result_path: str, aur_source_dict: dict, aur_conditions_dict: dict) -> None:
        """
        ================================================================================================================
        Class that saves the sphere that comes from the HAWC2 simulation.
        ================================================================================================================
        Subclass of list, where the items are H2Observer instances.
        :param h2_result_path: path to the directory with all the HAWC2 noise .out files
        :param aur_source_dict: source_dict from the Case class
        :param aur_conditions_dict: conditions_dict from the Case class
        """
        # Check the .out files directory does exist
        if not os.path.isdir(h2_result_path):
            raise NotADirectoryError('Given directory for HAWC2 results does not exist.')
        # Initialise list class
        super().__init__()
        # Set the results path and scope
        self.h2_result_path = h2_result_path
        self.scope = aur_source_dict['scope']
        # Set the radius and origin point of the sphere
        self.radius = aur_conditions_dict['rotor_radius'] * aur_source_dict['radius_factor']
        self.origin = aur_conditions_dict['hub_pos']

        # Obtain all the filenames of the HAWC2 noise output
        out_files = [os.path.join(self.h2_result_path, file_name)
                     for file_name in os.listdir(self.h2_result_path) if file_name.endswith('.out')]

        # Check HAWC2 result files actually exist
        if len(out_files) <= 0:
            raise FileNotFoundError('No HAWC2 noise results found in results folder.')

        # Load the sphere from the HAWC2 output files
        p_thread = hf.ProgressThread(len(out_files), 'Loading HAWC2 results')
        p_thread.start()
        for file_path in out_files:
            self.append(H2Observer(file_path, self.scope, aur_conditions_dict['delta_t']))
            p_thread.update()
        p_thread.stop()
        del p_thread

        self.time_series = self[0].time_series

    def interpolate_sound(self, pos: hf.Cartesian, blade_idx: int, t: float) -> pd.DataFrame:
        """
        Triangular interpolation of the sound spectrum.
        :param pos: Cartesian point to interpolate to
        :param blade_idx: index of the blade (0: turbine, 1-3: blade 1-3) of which to interpolate the psd
        :param t: time step (s) at which to interpolate
        :return: the interpolated spectrum of the selected blade and timestep
        """
        # Determine the distances to the H2Observer points
        dist = np.array([pos.dist(observer) for observer in self])
        # Sorting index for dist (and thus the H2Observers)
        sort_idx = np.argsort(dist)
        # Apply the sorting and get 3 closest points
        closest = np.array(self)[sort_idx][:3]
        closest_dist = dist[sort_idx][:3]

        # Determine the interpolation denominator
        den = sum(1 / closest_dist)

        # Weighted triangular interpolation of the sound spectra
        psd = sum([observer.psd[blade_idx][t] / closest_dist[io] for io, observer in enumerate(closest)]) / den

        return psd


class Source(hf.Cartesian):
    def __init__(self, x: float, y: float, z: float, sphere: list[hf.Cartesian],
                 sphere_dist: float, t: float, blade: str) -> None:
        """
        ================================================================================================================
        Class that stores a sound source. Subclass of Cartesian for ease of use.
        ================================================================================================================
        :param x: coordinate on the x-axis (m)
        :param y: coordinate on the y-axis (m)
        :param z: coordinate on the z-axis (m)
        :param sphere: a unit sphere around (0, 0, 0) to start rays from
        :param sphere_dist: the inter-point distance of the sphere
        :param t: time step (s) when the sound is on the H2Sphere
        :param blade: blade label of this sound Source
        """
        super().__init__(x, y, z)
        self._cartesian = hf.Cartesian(x, y, z)

        self.sphere = [point + self for point in sphere]
        self.sphere_dist = sphere_dist
        self.blade = blade
        self.t = t

    def __repr__(self) -> str:
        return f'<Source: {str(self)}, t = {self.t} s>'

    def generate_rays(self, h2_sphere: H2Sphere, atmosphere: hf.Atmosphere, ray_queue: list[pm.SoundRay],
                      receiver: rm.Receiver, models: tuple, ground_type: str) -> None:
        """
        Generate SoundRays that would come from this source and put them into the ray_queue.
        :param h2_sphere: the sphere of HAWC2 results to obtain sound from
        :param atmosphere: the hf.Atmosphere to propagate used for propagation
        :param ray_queue: queue.Queue to put generated SoundRays in
        :param receiver: instance of rm.Receiver to limit which SoundRays to keep
        :param models: the propagation effects models to be used in propagation
        :param ground_type:
        :return: the ray_queue with more items
        """
        for point in self.sphere:
            # Determine beam width
            beam_width = 2 * np.arcsin(self.sphere_dist / (2 * point.dist(self)))

            dir_ray = ((point - self) / (point - self).len()).to_spherical(hf.Cartesian(0, 0, 0))
            dir_rec = ((receiver - self) / (receiver - self).len()).to_spherical(hf.Cartesian(0, 0, 0))

            # TODO: Make angle an input parameter
            if abs(hf.limit_angle(dir_ray[1] - dir_rec[1])) <= 25 * np.pi / 180 and abs(
                    hf.limit_angle(dir_ray[2] - dir_rec[2])) <= 25 * np.pi / 180:

                # determinant of line-sphere intersection
                nabla = np.sum(((point - self) * (self - h2_sphere.origin)).vec)**2 - (
                        h2_sphere.origin.dist(self)**2 - h2_sphere.radius**2)
                # distance from self to edge of sphere in direction of sphere
                dist = -np.sum(((point - self) * (self - h2_sphere.origin)).vec) + np.sqrt(nabla)
                # point on sphere at end of initial ray
                pos_0 = self + (point - self) * dist

                # determine initial ray direction and initial travel distance
                dir_0 = pos_0 - self
                s_0 = dir_0.len()
                # Determine local speed of sound
                speed_of_sound = atmosphere.get_speed_of_sound(-pos_0[2])
                # Set the initial velocity vector
                vel_0 = speed_of_sound * dir_0 / dir_0.len()
                # Get the relevant amplitude spectrum
                spectrum = np.sqrt(h2_sphere.interpolate_sound(pos_0, int(self.blade[-1]), self.t))

                ray_queue.append(pm.SoundRay(pos_0, vel_0, s_0, self._cartesian, beam_width, spectrum, models,
                                             ground_type=ground_type, t_0=self.t, label=self.blade))


class SourceModel:
    def __init__(self, aur_conditions_dict: dict, aur_source_dict: dict, h2_result_path: str,
                 atmosphere: hf.Atmosphere, dummy: bool = False, simple: bool = False) -> None:
        """
        ================================================================================================================
        Class that manages the whole source model.
        ================================================================================================================
        :param aur_conditions_dict: conditions_dict from the Case class
        :param aur_source_dict: source_dict from the Case class
        :param h2_result_path: path where the HAWC2 results are stored
        :param atmosphere: atmosphere defined in hf.Atmosphere()
        :param simple: boolean to select the simple source model
        """
        # Store the input parameters
        self.conditions_dict = aur_conditions_dict
        self.params = aur_source_dict
        self.h2_result_path = h2_result_path
        self.atmosphere = atmosphere
        self.simple = simple

        print(f' -- Source Model')
        self.h2_sphere = None if dummy else H2Sphere(self.h2_result_path, self.params, self.conditions_dict)
        self.time_series = None if dummy else self.h2_sphere.time_series

        self.source_queue = list[Source]()

        if dummy:
            print('Set up as dummy.')
        else:
            # Set the source origin radius
            radius = self.params['blade_percent'] * self.conditions_dict['rotor_radius'] / 100
            # Loop over time steps
            for t in self.time_series.index:
                # Set the hub point as origin for the cylindrical blade coordinates
                origin = hf.Cartesian(*self.time_series.loc[t, ['hub_x', 'hub_y', 'hub_z']])

                if self.simple:
                    # Assign the hub coordinate
                    self.time_series.loc[t, 'blade_0'] = origin

                    raise NotImplementedError('Simple source model not implemented yet.')

                else:
                    # Assign the coordinates of blades 1 through 3, from their rotation from the HAWC2 output file
                    self.time_series.loc[t, 'blade_1'] = hf.Cylindrical(radius, self.time_series.loc[t, 'psi_1'], 0,
                                                                        origin).to_cartesian()
                    self.time_series.loc[t, 'blade_2'] = hf.Cylindrical(radius, self.time_series.loc[t, 'psi_2'], 0,
                                                                        origin).to_cartesian()
                    self.time_series.loc[t, 'blade_3'] = hf.Cylindrical(radius, self.time_series.loc[t, 'psi_3'], 0,
                                                                        origin).to_cartesian()

            # Generate the sound sources
            points, fail, dist = hf.uniform_spherical_grid(self.params['n_rays'])

            estimate = self.time_series.index.size
            estimate *= 1 if self.simple else 3
            p_thread = hf.ProgressThread(estimate, 'Generating sources')
            p_thread.start()

            for key in self.time_series.columns:
                if 'blade' in key:
                    for t in self.time_series.index:
                        x, y, z = self.time_series.loc[t, key].vec
                        source = Source(x, y, z, points, dist, t, key)

                        self.source_queue.append(source)
                        p_thread.update()

            p_thread.stop()
            del p_thread

    def run(self, receiver: rm.Receiver, models: tuple) -> list[pm.SoundRay]:  # queue.Queue:
        """
        Run the Source model, aka generate all rays from all Sources
        :param receiver:
        :param models:
        :return: a queue containing the generated SoundRays
        """
        # Create an empty queue.Queue for the SoundRays
        ray_queue = list[pm.SoundRay]()
        # Start a ProgressThread
        p_thread = hf.ProgressThread(len(self.source_queue), 'Generating rays')
        p_thread.start()
        # Loop over the source_queue without popping the Sources
        for source in self.source_queue:
            # Generate the rays for the current Source
            source.generate_rays(self.h2_sphere, self.atmosphere, ray_queue, receiver, models, self.conditions_dict['ground_type'])
            # Update the progress thread
            p_thread.update()

        # Stop the ProgressThread
        p_thread.stop()
        del p_thread

        return ray_queue

    def interactive_source_plot(self, gif_out: str = None):
        """
        A beautiful interactive plot of the Source locations.
        :param gif_out: A path to output a gif animation to
        """
        # Create empty dictionary to store Sources per time step
        sources = dict[float: list]()
        # Loop over the source_queue without popping the Sources
        for source in self.source_queue:
            # Avoid stoopid floating point errors with time steps
            t = round(source.t, 10)
            # Fill the dictionary
            if t in sources.keys():
                sources[t].append(source)
            else:
                sources[t] = list[Source]([source, ])

        # create the main plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x_h, y_h, z_h = self.conditions_dict['hub_pos'].vec
        offset = 50

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_h2s = self.h2_sphere.radius * np.outer(np.cos(u), np.sin(v)) + self.h2_sphere.origin[0]
        y_h2s = self.h2_sphere.radius * np.outer(np.sin(u), np.sin(v)) + self.h2_sphere.origin[1]
        z_h2s = self.h2_sphere.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.h2_sphere.origin[2]

        points = []
        for xi, _ in enumerate(x_h2s):
            points.append([])
            for xj, _ in enumerate(x_h2s[xi]):
                points[xi].append(hf.Cartesian(x_h2s[xi, xj], y_h2s[xi, xj], z_h2s[xi, xj]))

        c_h2s = np.zeros(x_h2s.shape)

        cmap = mpl.colormaps['viridis']
        norm = mpl.colors.Normalize(-5, 35)
        mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

        def update_plot(t_plt: float):
            """
            Internal function to update the interactive plot with the Slider.
            :param t_plt: time input (s)
            """
            print(t_plt)
            # Clear the plot
            ax.clear()
            # Re-set the axis limits and aspect ratio
            ax.set_aspect('equal')
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.set_zlim(0, 100)
            # Re-set the labels
            ax.set_xlabel('-x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('-z (m)')

            for xi, _ in enumerate(x_h2s):
                for xj, _ in enumerate(x_h2s[xi]):
                    pt: hf.Cartesian = points[xi][xj]
                    psd = self.h2_sphere.interpolate_sound(pt, 1, t_plt)
                    c_h2s[xi, xj] = 10 * np.log10(psd.iloc[11] / hf.p_ref ** 2)
            # print(np.min(c_h2s), np.max(c_h2s))

            ax.plot_surface(-x_h2s, y_h2s, -z_h2s, facecolors=mappable.to_rgba(c_h2s))

            # Get the Sources at input time
            source_lst = list[Source](sources[t_plt])
            # Plot the source points
            for src in source_lst:
                if src.blade == 'blade_1':
                    color = 'r'
                else:
                    color = 'k'

                x, y, z = src.vec
                ax.scatter(-x, y + offset, -z, s=5, color=color)
                ax.plot((-x, -x_h), (y + offset, y_h + offset), (-z, -z_h), color=color)

        valstep = list(sorted(sources.keys()))

        # Create an animated GIF file if so desired
        if gif_out is not None:
            ani = FuncAnimation(fig=fig, func=update_plot, frames=valstep, interval=(valstep[1] - valstep[0]) * 1000)
            ani.save(gif_out, writer='pillow')

            # Create the figure again to avoid issues with the animation stuff
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0., bottom=0.2, right=1., top=1.)

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
        # Plot the plot
        plt.show()
