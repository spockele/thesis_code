import os
import shutil as sh

import case_mgmt as cm
import helper_functions as hf


"""
========================================================================================================================
===                                                                                                                  ===
=== Main run code of the Auralisation Tool                                                                           ===
===                                                                                                                  ===
========================================================================================================================
"""


class Project:
    def __init__(self, project_path: str,):
        """
        ================================================================================================================
        Class that manages an auralisation project
        ================================================================================================================
        :param project_path: path of the directory containing the auralisation project
        """
        # Check if project folder exists.
        if not os.path.isdir(project_path):
            raise NotADirectoryError('Invalid project directory path given.')

        # Create paths for project and for the HAWC2 model
        self.project_path = project_path
        self.h2model_path = os.path.join(project_path, 'H2model')

        # Check if the project contains a HAWC2 model
        if not os.path.isdir(self.h2model_path):
            raise NotADirectoryError('The given project directory does not contain a HAWC2 model in folder "H2model".')

        # Make atmosphere folder if that does not exist yet
        if not os.path.isdir(os.path.join(self.project_path, 'atm')):
            os.mkdir(os.path.join(self.project_path, 'atm'))

        # Make wavfile folder if that does not exist yet
        if not os.path.isdir(os.path.join(self.project_path, 'wavfiles')):
            os.mkdir(os.path.join(self.project_path, 'wavfiles'))

        # Make spectrograms folder if that does not exist yet
        if not os.path.isdir(os.path.join(self.project_path, 'spectrograms')):
            os.mkdir(os.path.join(self.project_path, 'spectrograms'))

        # Make wavfile folder if that does not exist yet
        if not os.path.isdir(os.path.join(self.project_path, 'pickles')):
            os.mkdir(os.path.join(self.project_path, 'pickles'))

        # Obtain cases from the project folder
        cases = [aur_file for aur_file in os.listdir(self.project_path) if aur_file.endswith('.aur')]

        if len(cases) <= 0:
            raise FileNotFoundError('No input files found in project folder.')

        p_thread = hf.ProgressThread(len(cases), 'Loading case files')
        p_thread.start()

        self.cases = []
        for aur_file in cases:
            self.cases.append(cm.Case(self.project_path, aur_file))
            p_thread.update()
        p_thread.stop()

    def run(self):
        """
        Runs all cases in the project
        """
        for ci, case in enumerate(self.cases):
            print(f'================= Simulating case {case.case_name} ({ci + 1}/{len(self.cases)}) =================')
            if case.run_hawc:
                case.run_hawc2()
            case.run()
            print()


if __name__ == '__main__':
    proj_name = input('Enter your project name: ')
    proj_path = os.path.abspath(proj_name)
    proj = Project(proj_path)
    proj.run()
