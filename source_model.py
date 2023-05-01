import os
import pandas as pd
import numpy as np
import helper_functions as hf


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
        self.turbine_psd = turbine_psd
        self.blade_1_psd = blade_1_psd
        self.blade_2_psd = blade_2_psd
        self.blade_3_psd = blade_3_psd

    def __repr__(self):
        return f'<Sound source at {str(self)}>'


class H2Observer(hf.Cartesian):
    def __init__(self, fname: str, scope: str):
        """
        ================================================================================================================

        ================================================================================================================
        :param fname:
        :param scope:
        """
        pos, time_series, psd = hf.read_hawc2_aero_noise(fname, scope=scope)
        super().__init__(*pos)
        self.time_series = time_series
        self.turbine_psd = psd[0]
        self.blade_1_psd = psd[1]
        self.blade_2_psd = psd[2]
        self.blade_3_psd = psd[3]

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
    def __init__(self, n_rays: int, h2_sphere: H2Sphere):
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

        p_thread = hf.ProgressThread(n_rays, 'Interpolating results')
        p_thread.start()
        for point in points:
            self.append(h2_sphere.interpolate_sound(point))
            p_thread.update()

        p_thread.stop()


class SourceModel:
    def __init__(self, aur_conditions_dict: dict, aur_source_dict: dict, h2_result_path: str, scope: str = 'All'):
        """
        ================================================================================================================

        ================================================================================================================
        :param aur_conditions_dict:
        :param aur_source_dict:
        :param h2_result_path:
        :param scope:
        """
        self.conditions = aur_conditions_dict
        self.params = aur_source_dict
        self.h2_result_path = h2_result_path

        self.h2_sphere = H2Sphere(self.h2_result_path, scope)
        self.source_sphere = SourceSphere(self.params['n_rays'], self.h2_sphere)


if __name__ == '__main__':
    pass
