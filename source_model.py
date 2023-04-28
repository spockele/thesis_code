import os

import wetb.hawc2 as h2
import numpy as np

import helper_functions as hf


class H2Sphere:
    def __init__(self, h2result_path: str):
        self.h2result_path = h2result_path
        self.h2result_sphere = None

    def load_sphere(self):
        self.h2result_sphere = [os.path.join(self.h2result_path, fname)
                                for fname in os.listdir(self.h2result_path) if fname.endswith('.out')]

        for fname in self.h2result_sphere:
            pos, time, psd = hf.read_hawc2_aero_noise(fname)
            print(psd)


class SourceModel:
    def __init__(self, aur_file_dict: dict, h2result_path: str, ):
        self.params = aur_file_dict
        self.h2result_path = h2result_path

        self.h2sphere = H2Sphere(self.h2result_path)


if __name__ == '__main__':
    pass
