import os
import pandas as pd
import numpy as np
import helper_functions as hf

import propagation_model as pm


"""
========================================================================================================================
===                                                                                                                  ===
===                                                                                                                  ===
===                                                                                                                  ===
========================================================================================================================
"""


class Source(hf.Cartesian):
    def __init__(self, pos: np.array, time_series, turbine_psd, blade_1_psd, blade_2_psd, blade_3_psd):
        """
        ================================================================================================================

        ================================================================================================================
        :param pos:
        :param time_series:
        :param turbine_psd:
        :param blade_1_psd:
        :param blade_2_psd:
        :param blade_3_psd:
        """
        super().__init__(*pos)
        self.time_series = time_series
        self.psd = {'blade_0': turbine_psd, 'blade_1': blade_1_psd, 'blade_2': blade_2_psd, 'blade_3': blade_3_psd}

    def __repr__(self):
        return f'<Sound source at {str(self)}>'

    def to_cartesian(self):
        return hf.Cartesian(*self.vec)

    def generate_rays(self, aur_conditions_dict: dict, aur_source_dict: dict, atmosphere: hf.Atmosphere,
                      beam_width: float):
        """

        :param aur_conditions_dict:
        :param aur_source_dict:
        :param atmosphere:
        :param beam_width:
        :return:
        """
        radius = aur_source_dict['blade_percent'] * aur_conditions_dict['rotor_radius'] / 100
        for t in self.time_series.index:
            origin = hf.Cartesian(*self.time_series.loc[t, ['hub_x', 'hub_y', 'hub_z']])
            self.time_series.loc[t, 'blade_0'] = origin
            self.time_series.loc[t, 'blade_1'] = hf.Cylindrical(radius, self.time_series.loc[t, 'psi_1'], 0,
                                                                origin).to_cartesian()
            self.time_series.loc[t, 'blade_2'] = hf.Cylindrical(radius, self.time_series.loc[t, 'psi_2'], 0,
                                                                origin).to_cartesian()
            self.time_series.loc[t, 'blade_3'] = hf.Cylindrical(radius, self.time_series.loc[t, 'psi_3'], 0,
                                                                origin).to_cartesian()

        rays = pd.DataFrame(index=self.time_series.index, columns=['blade_0', 'blade_1', 'blade_2', 'blade_3'])
        for t in rays.index:
            for blade in rays.columns:
                pos_0 = self.to_cartesian()

                dir_0 = self.to_cartesian() - self.time_series.loc[t, blade]
                speed_of_sound = atmosphere.get_speed_of_sound(-pos_0[2])
                vel_0 = speed_of_sound * dir_0 / dir_0.len()
                amplitude_spectrum = self.psd[blade].loc[:, t]

                rays.loc[t, blade] = pm.SoundRay(pos_0, vel_0, dir_0.len(), beam_width, atmosphere,
                                                 amplitude_spectrum, t_0=t, label=blade)

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
        pos, time_series, psd = hf.read_hawc2_aero_noise(fname, scope=scope)
        super().__init__(*pos)
        self.time_series = time_series
        self.turbine_psd = psd[0]
        self.blade_1_psd = psd[1]
        self.blade_2_psd = psd[2]
        self.blade_3_psd = psd[3]

        self.time_series.index = self.time_series.index - self.time_series.index[0]
        self.turbine_psd.columns = self.time_series.index
        self.blade_1_psd.columns = self.time_series.index
        self.blade_2_psd.columns = self.time_series.index
        self.blade_3_psd.columns = self.time_series.index

    def __repr__(self):
        return f'<HAWC2 Observer at {str(self)}>'


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

        # Load the sphere from the HAWC2 output files
        out_files = [os.path.join(self.h2_result_path, fname)
                     for fname in os.listdir(self.h2_result_path) if fname.endswith('.out')]

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
        dist = np.array([pos.dist(observer) for observer in self])
        sort_idx = np.argsort(dist)
        closest = np.array(self)[sort_idx][:3]
        closest_dist = dist[sort_idx][:3]

        den = sum(1 / closest_dist)

        time_series = sum([observer.time_series / closest_dist[io] for io, observer in enumerate(closest)]) / den
        turbine_psd = sum([observer.turbine_psd / closest_dist[io] for io, observer in enumerate(closest)]) / den
        blade_1_psd = sum([observer.blade_1_psd / closest_dist[io] for io, observer in enumerate(closest)]) / den
        blade_2_psd = sum([observer.blade_2_psd / closest_dist[io] for io, observer in enumerate(closest)]) / den
        blade_3_psd = sum([observer.blade_3_psd / closest_dist[io] for io, observer in enumerate(closest)]) / den

        return Source(pos, time_series, turbine_psd, blade_1_psd, blade_2_psd, blade_3_psd)


class SourceSphere(list):
    def __init__(self, n_rays: int, h2_sphere: H2Sphere, radius: float, offset: hf.Cartesian):
        """
        ================================================================================================================

        ================================================================================================================
        :param n_rays:
        :param h2_sphere:
        """
        super().__init__()
        self.h2_sphere = h2_sphere

        points, fail, self.dist = hf.uniform_spherical_grid(n_rays)
        if fail:
            raise ValueError(f'Parameter n_rays = {n_rays} resulted in incomplete sphere. Try a different value.')

        p_thread = hf.ProgressThread(len(points), 'Interpolating results')
        p_thread.start()
        for point in points:
            self.append(h2_sphere.interpolate_sound(radius * point + offset))
            p_thread.update()

        p_thread.stop()


class SourceModel:
    def __init__(self, aur_conditions_dict: dict, aur_source_dict: dict, h2_result_path: str, atmosphere: hf.Atmosphere,
                 offset: hf.Cartesian):
        """
        ================================================================================================================

        ================================================================================================================
        :param aur_conditions_dict:
        :param aur_source_dict:
        :param h2_result_path:
        """
        self.conditions_dict = aur_conditions_dict
        self.params = aur_source_dict
        self.h2_result_path = h2_result_path
        self.atmosphere = atmosphere

        self.h2_sphere = H2Sphere(self.h2_result_path, self.params['scope'])
        self.source_sphere = SourceSphere(self.params['n_rays'], self.h2_sphere,
                                          self.conditions_dict['rotor_radius'], offset)

    def run(self):
        """

        """
        ray_list = []
        source: Source
        p_thread = hf.ProgressThread(len(self.source_sphere), 'Generating sound rays')
        p_thread.start()
        for source in self.source_sphere:
            # print(source.to_cartesian())
            rays = source.generate_rays(self.conditions_dict, self.params, self.atmosphere, .1)
            ray_list.append(rays)
            p_thread.update()

        p_thread.stop()

        return ray_list


if __name__ == '__main__':
    pass
