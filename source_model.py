import wetb.hawc2 as h2
import numpy as np

import helper_functions as hf


class H2SoundSource:
    def __init__(self, turbine_radius: float, turbine_hub_height: float, path: str = None, name: str = None,
                 filename: str = None, modelpath: str = None, jinja_tags: dict = None):

        self.hub_height = turbine_hub_height
        self.radius = turbine_radius
        self.filename = filename

        if path is None:
            self.path = '/'.join(filename.split('/')[:-1])
        else:
            self.path = path

        if name is None:
            self.name = filename.split('/')[-1].strip('.htc')
        else:
            self.name = name

        if jinja_tags is None:
            jinja_tags = {}

        self.htc = h2.HTCFile(filename, modelpath, jinja_tags)

    def insert_aero_noise_sphere(self, to_file: bool = False):
        *coordinates, fail, pd = hf.uniform_spherical_grid(255)
        coo = np.array(coordinates)

        offset = hf.Cartesian(0, 0, -self.hub_height)

        pos = [hf.Cartesian(self.radius * np.cos(coo[1][idx]) * np.sin(coo[1][idx]),
                            self.radius * np.sin(coo[1][idx]) * np.sin(coo[1][idx]),
                            self.radius * np.cos(coo[1][idx])) + offset
               for idx in range(coo.shape[1])]

        for pi, p in enumerate(pos):
            self.htc.aero.aero_noise.add_line(name="xyz_observer", values=p.vec, comments=f'Observer_{pi}')

        if to_file:
            self.htc.save(f'{self.path}/{self.name}.htc')

    def run_simulation(self, ):
        pass


if __name__ == '__main__':
    sound = H2SoundSource(20.5, 35.5, filename='NTK/H2model/htc/ntk_500_41.htc')
    # sound.insert_aero_noise_sphere()
    # print(sound.htc.aero.aero_noise)
