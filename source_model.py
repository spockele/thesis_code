import os
import queue
import numpy as np

import helper_functions as hf
import propagation_model as pm


"""
========================================================================================================================
===                                                                                                                  ===
=== The source model for this auralisation tool                                                                      ===
===                                                                                                                  ===
========================================================================================================================
"""


class H2Observer(hf.Cartesian):
    def __init__(self, file_path: str, scope: str, delta_t: float):
        """
        ================================================================================================================
        Class to wrap the HAWC2 result at a single observer point. Subclass of Cartesian for ease of use
        ================================================================================================================
        :param file_path: path to the result file for this observer point.
        :param scope: Selects the noise model result to load ('All', 'TI', 'TE', 'ST', 'TP').
        :param delta_t: simulation time step (s) to interpolate the HAWC2 results to.
        """
        # Read the given HAWC2 noise output file
        pos, time_series, psd = hf.read_hawc2_aero_noise(file_path, scope=scope)
        # Set the point position
        super().__init__(*pos)
        # Store the time series and the spectrograms
        self.time_series = time_series
        self.psd = psd

        # Set the initial time of the time series and spectrograms to 0
        # as the absolute time is not relevant for this project
        self.time_series.index = self.time_series.index - self.time_series.index[0]
        self.psd[0].columns = self.time_series.index
        self.psd[1].columns = self.time_series.index
        self.psd[2].columns = self.time_series.index
        self.psd[3].columns = self.time_series.index

        # Interpolation to simulation time step
        sim_time = np.round(np.arange(self.time_series.index[0], self.time_series.index[-1] + delta_t, delta_t), 10)
        #  of the time series
        self.time_series.index = np.round(self.time_series.index, 10)
        self.time_series = self.time_series.reindex(index=sim_time)
        self.time_series.interpolate(axis='index', inplace=True)
        #  of the turbine psd
        self.psd[0].columns = np.round(self.psd[0].columns, 10)
        self.psd[0] = self.psd[0].reindex(columns=sim_time)
        self.psd[0].interpolate(axis='columns', inplace=True)
        #  of the blade_1 psd
        self.psd[1].columns = np.round(self.psd[1].columns, 10)
        self.psd[1] = self.psd[1].reindex(columns=sim_time)
        self.psd[1].interpolate(axis='columns', inplace=True)
        #  of the blade_2 psd
        self.psd[2].columns = np.round(self.psd[2].columns, 10)
        self.psd[2] = self.psd[2].reindex(columns=sim_time)
        self.psd[2].interpolate(axis='columns', inplace=True)
        #  of the blade_3 psd
        self.psd[3].columns = np.round(self.psd[3].columns, 10)
        self.psd[3] = self.psd[3].reindex(columns=sim_time)
        self.psd[3].interpolate(axis='columns', inplace=True)

    def __repr__(self):
        return f'<HAWC2 Observer: {str(self)}>'


class H2Sphere(list[H2Observer]):
    def __init__(self, h2_result_path: str, aur_source_dict: dict, aur_conditions_dict: dict):
        """
        ================================================================================================================
        Class that saves the sphere that comes from the HAWC2 simulation.
        ================================================================================================================
        Subclass of list, where the items are H2Observer instances.
        :param h2_result_path: Path to the directory with all the HAWC2 noise .out files.
        :param aur_source_dict: source_dict from the Case class.
        :param aur_conditions_dict: conditions_dict from the Case class.
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
        self.radius = aur_conditions_dict['rotor_radius']
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

        self.time_series = self[0].time_series

    def interpolate_sound(self, pos: hf.Cartesian, blade_idx: int, t: float):
        """
        Triangular interpolation of the sound spectrum.
        :param pos: Point to interpolate to.
        :param blade_idx: Index of the blade (0: turbine, 1-3: blade 1-3) of which to interpolate the psd.
        :param t: Time step (s) at which to interpolate.
        :return: The interpolated spectrum of the selected blade and timestep.
        """
        observer: H2Observer
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
                 sphere_dist: float, t: float, blade: str):
        """
        ================================================================================================================

        ================================================================================================================
        TODO: Source.__init__ > write docstring and comments
        :param x: Coordinate on the x-axis (m).
        :param y: Coordinate on the y-axis (m).
        :param z: Coordinate on the z-axis (m).
        :param sphere:
        :param sphere_dist:
        :param t:
        :param blade:
        """
        super().__init__(x, y, z)

        self.sphere = [point + self for point in sphere]
        self.sphere_dist = sphere_dist
        self.blade = blade
        self.t = t

    def __repr__(self):
        return f'<Source: {str(self)}, t = {self.t} s>'

    def generate_rays(self, h2_sphere: H2Sphere, atmosphere: hf.Atmosphere, ray_queue: queue.Queue,
                      p_thread: hf.ProgressThread):
        """
        TODO: Source.generate_rays > write docstring
        :param h2_sphere:
        :param atmosphere:
        :param ray_queue:
        :param p_thread:
        :return:
        """
        for point in self.sphere:
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
            # Determine beam width
            beam_width = 2 * np.arcsin(self.sphere_dist / point.dist(self) / 2)
            # Determine local speed of sound
            speed_of_sound = atmosphere.get_speed_of_sound(-pos_0[2])
            # Set the initial velocity vector
            vel_0 = speed_of_sound * dir_0 / dir_0.len()
            # Get the relevant amplitude spectrum
            spectrum = h2_sphere.interpolate_sound(pos_0, int(self.blade[-1]), self.t)

            ray_queue.put(pm.SoundRay(pos_0, vel_0, s_0, beam_width, spectrum, t_0=self.t, label=self.blade))
            p_thread.update()

        return ray_queue


class SourceModel:
    def __init__(self, aur_conditions_dict: dict, aur_source_dict: dict,
                 h2_result_path: str, atmosphere: hf.Atmosphere, simple: bool = False):
        """
        ================================================================================================================
        Class that manages the whole source model
        ================================================================================================================
        :param aur_conditions_dict: conditions_dict from the Case class.
        :param aur_source_dict: source_dict from the Case class.
        :param h2_result_path: path where the HAWC2 results are stored
        :param atmosphere: atmosphere defined in hf.Atmosphere()
        :param simple: Boolean to select the simple source model
        """
        # Store the input parameters
        self.conditions_dict = aur_conditions_dict
        self.params = aur_source_dict
        self.h2_result_path = h2_result_path
        self.atmosphere = atmosphere
        self.simple = simple

        print(f' -- Source Model')
        self.h2_sphere = H2Sphere(self.h2_result_path, self.params, self.conditions_dict)
        self.time_series = self.h2_sphere.time_series

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

        self.sources = []

    def generate_rays(self):
        """
        TODO: SourceModel.generate_rays > write docstring and comments
        :return: a queue containing the generated SoundRays
        """
        ray_queue = queue.Queue()
        points, fail, dist = hf.uniform_spherical_grid(self.params['n_rays'])

        estimate = self.time_series.index.size * len(points)
        estimate *= 1 if self.simple else 3
        p_thread = hf.ProgressThread(estimate, 'Generating sound rays')
        p_thread.start()

        for key in self.time_series.columns:
            if 'blade' in key:
                for t in self.time_series.index:
                    x, y, z = self.time_series.loc[t, key].vec
                    source = Source(x, y, z, points, dist, t, key)
                    ray_queue = source.generate_rays(self.h2_sphere, self.atmosphere, ray_queue, p_thread)

                    self.sources.append(source)

        p_thread.stop()
        return ray_queue

    def run(self) -> queue.Queue:
        """
        Run the source model
        """
        return self.generate_rays()
