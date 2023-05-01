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


class H2Observer(Source):
    def __init__(self, fname: str, scope: str):
        """
        ================================================================================================================

        ================================================================================================================
        :param fname:
        :param scope:
        """
        pos, time_series, psd = hf.read_hawc2_aero_noise(fname, scope=scope)
        super().__init__(pos, time_series, psd[0], psd[1], psd[2], psd[3])

    def __repr__(self):
        return f'<HAWC2 Observer at {str(self)}>'


class H2Sphere(list):
    def __init__(self, h2result_path: str, scope: str):
        """
        ================================================================================================================
        Class that saves the sphere that comes from the HAWC2 simulation.
        ================================================================================================================
        Subclass of list, where the items are H2Observer instances

        :param h2result_path: path to the folder with all the HAWC2 noise .out files
        :param scope: Selects the noise model result to load ('All', 'TI', 'TE', 'ST', 'TP')
        """
        super().__init__()
        self.h2result_path = h2result_path
        self.scope = scope

    def load_sphere(self):
        """
        Load the HAWC2 data sphere to self
        """
        out_files = [os.path.join(self.h2result_path, fname)
                     for fname in os.listdir(self.h2result_path) if fname.endswith('.out')]

        p_thread = hf.ProgressThread(len(out_files), 'Loading HAWC2 results sphere')
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


class SourceModel:
    def __init__(self, aur_conditions_dict: dict, aur_source_dict: dict, h2result_path: str, scope: str = 'All'):
        """
        ================================================================================================================

        ================================================================================================================
        :param aur_conditions_dict:
        :param aur_source_dict:
        :param h2result_path:
        :param scope:
        """
        self.conditions = aur_conditions_dict
        self.params = aur_source_dict
        self.h2result_path = h2result_path

        self.h2sphere = H2Sphere(self.h2result_path, scope)


if __name__ == '__main__':
    pass
