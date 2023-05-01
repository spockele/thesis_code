import numpy as np
import threading
import time
import sys


"""
========================================================================================================================
===                                                                                                                  ===
=== Uncategorised functions needed for other modules in this package                                                 ===
===                                                                                                                  ===
========================================================================================================================
"""


c = 343  # Speed of Sound [m/s]
p_ref = 2e-5  # Sound reference pressure [Pa]
g = 9.80665  # Gravity [m/s2]
r_air = 287.  # Specific gas constant for air [J/kgK]
t_0 = 273.15  # Zero celsius in kelvin [K]
gamma_air = 1.4  # Specific heat ratio of air [-]


def limit_angle(angle):
    """
    Limit a radian angle between -pi and pi
    :param angle: input angle(s) IN RADIANS
    :return: limited angle(s) IN RADIANS
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def a_weighting(f):
    """
    A-weighting function Delta L_A
    :param f: frequency array
    :return: Array with corresponding values of Delta L_A
    """
    return -145.528 + 98.262 * np.log10(f) - 19.509 * np.log10(f) ** 2 + 0.975 * np.log10(f) ** 3


class ProgressThread(threading.Thread):
    """
    ====================================================================================================================
    Subclass of threading.Thread to print the progress of a program in steps
    ====================================================================================================================
    Originally developed for EWI3615TU - Computer Science Project 2019/2020; Group 14 - Twitter's influenza; authored by
        Jérémie Gaffarel, Josephine Siebert Pockelé, Guillermo Presa, Enes Ugurlu, Sebastiaan van Wijk
    """
    def __init__(self, total: int, task: str):
        super().__init__(name='ProgressThread')
        self.step = 1
        self.total = total
        self.task = task
        self.work = True
        self.t0 = time.time()

    def run(self) -> None:
        """
        Override of threading.Thread.run(self) for the printing
        """
        i = 0
        while self.work and threading.main_thread().is_alive():
            sys.stdout.write(f'\r{self.task}: {self.step}/{self.total}        ')
            sys.stdout.write(f'\r{self.task}: {self.step}/{self.total} {i*"."}')
            sys.stdout.flush()
            i %= 5
            i += 1
            time.sleep(0.5)

        if not threading.main_thread().is_alive() and self.work:
            print(f'Stopped {self} after Interupt of MainThread')

    def stop(self) -> None:
        """
        Function to stop the thread when it is not needed anymore
        """
        elapsed = round(time.time()- self.t0, 2)
        sys.stdout.write(f'\r{self.task}: {self.total}/{self.total} Done! (Elapsed time: {elapsed} s)\n')
        sys.stdout.flush()
        self.work = False

    def update(self):
        """
        Update the step counter by one
        """
        self.step += 1
