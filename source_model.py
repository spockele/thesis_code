import os
import pandas as pd
import wetb.hawc2 as h2
import numpy as np

import helper_functions as hf


class Source:
    def __init__(self, pos: hf.Cartesian, time_series, turbine_psd, blade_1_psd, blade_2_psd, blade_3_psd):
        self.pos = pos
        self.time_series = time_series
        self.turbine_psd = turbine_psd
        self.blade_1_psd = blade_1_psd
        self.blade_2_psd = blade_2_psd
        self.blade_3_psd = blade_3_psd

    def __repr__(self):
        return f'<Sound source at {str(self.pos)}>'


class H2Observer:
    def __init__(self, psd: dict, pos: hf.Cartesian, time_series: pd.DataFrame, scope: str):
        self.turbine_psd = psd[0][scope]
        self.blade_1_psd = psd[1][scope]
        self.blade_2_psd = psd[2][scope]
        self.blade_3_psd = psd[3][scope]

        self.pos = pos
        self.time_series = time_series

    def __repr__(self):
        return f'<HAWC2 Observer at {str(self.pos)}>'


class H2Sphere(list):
    def __init__(self, h2result_path: str, scope: str):
        super().__init__()
        self.h2result_path = h2result_path
        self.scope = scope

    def load_sphere(self):
        out_files = [os.path.join(self.h2result_path, fname)
                     for fname in os.listdir(self.h2result_path) if fname.endswith('.out')]

        for fname in out_files:
            pos, time_series, psd = hf.read_hawc2_aero_noise(fname)
            self.append(H2Observer(psd, hf.Cartesian(*pos), time_series, self.scope))

    def interpolate_sound(self, pos: hf.Cartesian, ):
        dist = []
        for observer in self:
            obs: hf.Cartesian = observer.pos
            dist.append((pos - obs).len())

        mx = max(dist)
        closest = []
        closest_dist = []
        for i in range(4):
            idx_min = np.argmin(dist)
            closest.append(self[idx_min])
            closest_dist.append(dist[idx_min])
            dist[idx_min] = mx + 1.

        turbine_psd = 0.
        blade_1_psd = 0.
        blade_2_psd = 0.
        blade_3_psd = 0.
        time_series = 0.
        den = 0.

        for oi, observer in enumerate(closest):
            turbine_psd += observer.turbine_psd / closest_dist[oi]
            blade_1_psd += observer.blade_1_psd / closest_dist[oi]
            blade_2_psd += observer.blade_2_psd / closest_dist[oi]
            blade_3_psd += observer.blade_3_psd / closest_dist[oi]
            time_series += observer.time_series / closest_dist[oi]

            den += 1 / closest_dist[oi]

        turbine_psd = turbine_psd / den
        blade_1_psd = blade_1_psd / den
        blade_2_psd = blade_2_psd / den
        blade_3_psd = blade_3_psd / den
        time_series = time_series / den

        return Source(pos, time_series, turbine_psd, blade_1_psd, blade_2_psd, blade_3_psd)


class SourceModel:
    def __init__(self, aur_conditions_dict: dict, aur_source_dict: dict, h2result_path: str, scope: str = 'All'):
        self.conditions = aur_conditions_dict
        self.params = aur_source_dict
        self.h2result_path = h2result_path

        self.h2sphere = H2Sphere(self.h2result_path, scope)


if __name__ == '__main__':
    pass
