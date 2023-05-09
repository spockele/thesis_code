import os
import queue
import pandas as pd
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


class Source(hf.Cartesian):
    def __init__(self, pos: np.array, time_series, turbine_psd, blade_1_psd, blade_2_psd, blade_3_psd):
        """
        ================================================================================================================
        class containing all information at a single source point
        ================================================================================================================
        :param pos: Position vector of the source point x,y,z (m)
        :param time_series: Time series information as loaded by helper_functions.read_hawc2_aero_noise()
        :param turbine_psd: Power spectral density spectrogram of the whole turbine (Pa^2 / Hz)
        :param blade_1_psd: Power spectral density spectrogram of the first blade (Pa^2 / Hz)
        :param blade_2_psd: Power spectral density spectrogram of the second blade (Pa^2 / Hz)
        :param blade_3_psd: Power spectral density spectrogram of the third blade (Pa^2 / Hz)
        """
        super().__init__(*pos)
        self.time_series = time_series
        self.psd = {'blade_0': turbine_psd, 'blade_1': blade_1_psd, 'blade_2': blade_2_psd, 'blade_3': blade_3_psd}

    def __repr__(self):
        return f'<Sound source: {str(self)}>'

    def to_cartesian(self):
        """
        Output the coordinates of this point as a Cartesian object
        """
        return hf.Cartesian(*self.vec)

    def generate_rays(self, aur_conditions_dict: dict, aur_source_dict: dict, atmosphere: hf.Atmosphere,
                      dist: float, rays: queue.Queue, delta_t: float):
        """
        Generate sound rays over the whole time series for this source point
        :param aur_conditions_dict: conditions_dict from the Case class
        :param aur_source_dict: source_dict from the Case class
        :param atmosphere: atmosphere defined in hf.Atmosphere()
        :param dist: estimate of inter-ray distance to determine beam width (m)
        :return: a queue with all the SoundRay instances
        """
        # Set the source origin radius
        radius = aur_source_dict['blade_percent'] * aur_conditions_dict['rotor_radius'] / 100
        # Interpolation to simulation time step
        sim_time = np.round(np.arange(self.time_series.index[0], self.time_series.index[-1] + delta_t, delta_t), 10)

        self.time_series.index = np.round(self.time_series.index, 10)
        self.time_series = self.time_series.reindex(index=sim_time)
        self.time_series.interpolate(axis='index', inplace=True)

        self.psd['blade_0'].columns = np.round(self.time_series.index, 10)
        self.psd['blade_0'] = self.psd['blade_0'].reindex(columns=sim_time)
        self.psd['blade_0'].interpolate(axis='columns', inplace=True)
        self.psd['blade_1'].columns = np.round(self.time_series.index, 10)
        self.psd['blade_1'] = self.psd['blade_1'].reindex(columns=sim_time)
        self.psd['blade_1'].interpolate(axis='columns', inplace=True)
        self.psd['blade_2'].columns = np.round(self.time_series.index, 10)
        self.psd['blade_2'] = self.psd['blade_2'].reindex(columns=sim_time)
        self.psd['blade_2'].interpolate(axis='columns', inplace=True)
        self.psd['blade_3'].columns = np.round(self.time_series.index, 10)
        self.psd['blade_3'] = self.psd['blade_3'].reindex(columns=sim_time)
        self.psd['blade_3'].interpolate(axis='columns', inplace=True)

        # Loop over time steps
        for t in self.time_series.index:
            # Set the hub point as origin for the cylindrical blade coordinates
            origin = hf.Cartesian(*self.time_series.loc[t, ['hub_x', 'hub_y', 'hub_z']])
            # Assign the hub coordinate
            self.time_series.loc[t, 'blade_0'] = origin
            # Assign the coordinates of blades 1 through 3, from their rotational angle value from the HAWC2 output file
            self.time_series.loc[t, 'blade_1'] = hf.Cylindrical(radius, self.time_series.loc[t, 'psi_1'], 0,
                                                                origin).to_cartesian()
            self.time_series.loc[t, 'blade_2'] = hf.Cylindrical(radius, self.time_series.loc[t, 'psi_2'], 0,
                                                                origin).to_cartesian()
            self.time_series.loc[t, 'blade_3'] = hf.Cylindrical(radius, self.time_series.loc[t, 'psi_3'], 0,
                                                                origin).to_cartesian()

        # Loop over time steps
        for t in self.time_series.index:
            # Loop over blades
            for blade in self.psd.keys():
                # Set initial position of sound ray
                pos_0 = self.to_cartesian()
                # Determine initial ray direction and initial travel distance
                dir_0 = self.to_cartesian() - self.time_series.loc[t, blade]
                s_0 = dir_0.len()
                # Determine beam width
                beam_width = 2 * np.arcsin(dist / s_0 / 2)
                # Determine local speed of sound
                speed_of_sound = atmosphere.get_speed_of_sound(-pos_0[2])
                # Set the initial velocity vector
                vel_0 = speed_of_sound * dir_0 / dir_0.len()
                # Get the relevant amplitude spectrum
                amplitude_spectrum = self.psd[blade].loc[:, t]

                # Create the SoundRay and put it in the pd.DataFrame
                rays.put(pm.SoundRay(pos_0, vel_0, s_0, beam_width, atmosphere, amplitude_spectrum, t_0=t, label=blade))

        # Return the whole pd.DataFrame
        return rays


class H2Observer(hf.Cartesian):
    def __init__(self, fname: str, scope: str):
        """
        ================================================================================================================
        Class to wrap the HAWC2 result at a single observer point. Subclass of Cartesian for ease of use
        ================================================================================================================
        :param fname: path to the result file for this observer point
        :param scope: Selects the noise model result to load ('All', 'TI', 'TE', 'ST', 'TP')
        """
        # Read the given HAWC2 noise output file
        pos, time_series, psd = hf.read_hawc2_aero_noise(fname, scope=scope)
        # Set the point position
        super().__init__(*pos)
        # Store the time series and the spectrograms
        self.time_series = time_series
        self.turbine_psd = psd[0]
        self.blade_1_psd = psd[1]
        self.blade_2_psd = psd[2]
        self.blade_3_psd = psd[3]

        # Set the initial time of the time series and spectrogram to 0
        # as the time shift is not relevant for this project
        self.time_series.index = self.time_series.index - self.time_series.index[0]
        self.turbine_psd.columns = self.time_series.index
        self.blade_1_psd.columns = self.time_series.index
        self.blade_2_psd.columns = self.time_series.index
        self.blade_3_psd.columns = self.time_series.index

    def __repr__(self):
        return f'<HAWC2 Observer: {str(self)}>'


class H2Sphere(list):
    def __init__(self, h2_result_path: str, scope: str):
        """
        ================================================================================================================
        Class that saves the sphere that comes from the HAWC2 simulation.
        ================================================================================================================
        Subclass of list, where the items are H2Observer instances
        :param h2_result_path: path to the folder with all the HAWC2 noise .out files
        :param scope: Selects the noise model result to load ('All', 'TI', 'TE', 'ST', 'TP')
        """
        super().__init__()
        self.h2_result_path = h2_result_path
        self.scope = scope

        # Obtain all the filenames of the HAWC2 noise output
        out_files = [os.path.join(self.h2_result_path, fname)
                     for fname in os.listdir(self.h2_result_path) if fname.endswith('.out')]

        # Load the sphere from the HAWC2 output files
        p_thread = hf.ProgressThread(len(out_files), 'Loading HAWC2 results')
        p_thread.start()
        for fname in out_files:
            self.append(H2Observer(fname, self.scope))
            p_thread.update()
        p_thread.stop()

    def interpolate_sound(self, pos: hf.Cartesian, ):
        """
        Triangular interpolation of closest 3 points to given position
        :param pos: point to interpolate to
        :return: a Source instance with all information at the given point
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
        time_series = sum([observer.time_series / closest_dist[io] for io, observer in enumerate(closest)]) / den
        turbine_psd = sum([observer.turbine_psd / closest_dist[io] for io, observer in enumerate(closest)]) / den
        blade_1_psd = sum([observer.blade_1_psd / closest_dist[io] for io, observer in enumerate(closest)]) / den
        blade_2_psd = sum([observer.blade_2_psd / closest_dist[io] for io, observer in enumerate(closest)]) / den
        blade_3_psd = sum([observer.blade_3_psd / closest_dist[io] for io, observer in enumerate(closest)]) / den

        # Return a Source instance with this interpolated data
        return Source(pos, time_series, turbine_psd, blade_1_psd, blade_2_psd, blade_3_psd)


class SourceSphere(list):
    def __init__(self, n_rays: int, h2_sphere: H2Sphere, radius: float, offset: hf.Cartesian):
        """
        ================================================================================================================
        Class to store the sphere from which SoundRays will be emitted
        ================================================================================================================
        Subclass of list, where the items are Source instances
        :param n_rays: Number of sound rays to emit
        :param h2_sphere: The H2Sphere object to derive the sources from
        :param radius: radius of the source sphere (m)
        :param offset: spatial offset where to place the source sphere x,y,z (m)
        """
        super().__init__()
        self.h2_sphere = h2_sphere

        # Generate the unit sphere (to be scaled by 'radius' and moved by 'offset')
        points, fail, self.dist = hf.uniform_spherical_grid(n_rays)
        if fail:
            raise ValueError(f'Parameter n_rays = {n_rays} resulted in incomplete sphere. Try a different value.')
        self.dist *= radius

        # Interpolate the H2Sphere to this sphere
        p_thread = hf.ProgressThread(len(points), 'Interpolating results')
        p_thread.start()
        for point in points:
            self.append(h2_sphere.interpolate_sound(radius * point + offset))
            p_thread.update()
        p_thread.stop()


class SourceModel:
    def __init__(self, aur_conditions_dict: dict, aur_source_dict: dict,
                 h2_result_path: str, atmosphere: hf.Atmosphere):
        """
        ================================================================================================================
        Class that manages the whole source model
        ================================================================================================================
        :param aur_conditions_dict: conditions_dict from the Case class
        :param aur_source_dict: source_dict from the Case class
        :param h2_result_path: path where the HAWC2 results are stored
        :param atmosphere: atmosphere defined in hf.Atmosphere()
        """
        # Store the input parameters
        self.conditions_dict = aur_conditions_dict
        self.params = aur_source_dict
        self.h2_result_path = h2_result_path
        self.atmosphere = atmosphere

        self.h2_sphere = H2Sphere(self.h2_result_path, self.params['scope'])
        self.source_sphere = SourceSphere(self.params['n_rays'], self.h2_sphere,
                                          self.params['radius_factor'] * self.conditions_dict['rotor_radius'],
                                          self.conditions_dict['hub_pos'])

    def run(self, delta_t: float):
        """
        Run the source model
        """
        ray_queue = queue.Queue()
        source: Source
        p_thread = hf.ProgressThread(len(self.source_sphere), 'Generating sound rays')
        p_thread.start()
        for source in self.source_sphere:
            ray_queue = source.generate_rays(self.conditions_dict, self.params, self.atmosphere, self.source_sphere.dist,
                                             ray_queue, delta_t)
            p_thread.update()

        p_thread.stop()

        return ray_queue
