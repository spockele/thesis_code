import wetb.hawc2 as h2
import numpy as np

import helper_functions as hf


class H2SoundSource(h2.HTCFile):
    def __init__(self, filename: str = None, modelpath: str = None, jinja_tags: dict = None):

        if jinja_tags is None:
            jinja_tags = {}

        super().__init__(filename, modelpath, jinja_tags)

    def insert_aero_noise_sphere(self):
        *coordinates, fail, pd = hf.uniform_spherical_grid(255)
        coo = np.array(coordinates)

        r = 41 / 2
        pos = [hf.Cartesian(r * np.cos(coo[1][idx]) * np.sin(coo[1][idx]),
                            r * np.sin(coo[1][idx]) * np.sin(coo[1][idx]),
                            r * np.cos(coo[1][idx]))
               for idx in range(coo.shape[1])]

        for p in pos:
            print(*np.round(p.vec, 5))


if __name__ == '__main__':
    sound = H2SoundSource('H2model/htc/ntk_05.5ms.htc')
    sound.insert_aero_noise_sphere()
