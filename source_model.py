import os
import pandas as pd
import wetb.hawc2 as h2
import numpy as np

import helper_functions as hf


class H2Observer:
    def __init__(self, psd: dict, pos: hf.Cartesian, time_series: pd.DataFrame, scope: str):
        self.turbine_psd = psd[0][scope]
        self.blade_1_psd = psd[1][scope]
        self.blade_2_psd = psd[2][scope]
        self.blade_3_psd = psd[3][scope]

        self.pos = pos
        self.time_series = time_series

    def __repr__(self):
        return f'Observer at {str(self.pos)}'


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
            print(pos)
            self.append(H2Observer(psd, hf.Cartesian(*pos), time_series, self.scope))

    def interpolate_sound(self, pos: hf.Cartesian, ):
        dist = []
        for observer in self:
            obs: hf.Cartesian = observer.pos
            dist.append((pos - obs).len())

        closest = []
        for i in range(4):
            idx_min = np.argmin(dist)
            closest.append(self[idx_min])
            print(idx_min, dist[idx_min], self[idx_min])
            dist[idx_min] = 1e15
            print(idx_min, dist[idx_min])

        print(closest)


class SourceModel:
    def __init__(self, aur_conditions_dict: dict, aur_source_dict: dict, h2result_path: str, scope: str = 'All'):
        self.conditions = aur_conditions_dict
        self.params = aur_source_dict
        self.h2result_path = h2result_path

        self.h2sphere = H2Sphere(self.h2result_path, scope)


if __name__ == '__main__':
    pass
