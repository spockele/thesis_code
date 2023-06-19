# Auralisation of Modelled Wind Turbine Noise for Psychoacoustic Listening Experiments

---
### Code repository for the M.Sc. thesis by Josephine Siebert Pockelé.

---
to obtain the degree of *Master of Science in Aerospace Engineering* at *Delft University of Technology*,\
and *Master of Science in Engineering (European Wind Energy)* at *Technical University of Denmark*.  
~~An electronic version of the report is available at <https://repository.tudelft.nl/>.~~

### Supervisors
[TU Delft, Aerospace Engineering](https://www.tudelft.nl/lr):
- [Dr. R. Merino-Martinez](https://research.tudelft.nl/en/persons/r-merino-martinez) (Assistant Professor, [Department of Aircraft Noise and Climate Effects](https://www.tudelft.nl/lr/organisatie/afdelingen/control-and-operations/aircraft-noise-and-climate-effects-ance))
- [Dr. D. Ragni](https://research.tudelft.nl/en/persons/d-ragni) (Associate Professor, [Department of Wind Energy](https://www.tudelft.nl/?id=4543))

[DTU, Wind and Energy Systems](https://wind.dtu.dk), [Wind Turbine Design Division](https://wind.dtu.dk/research/research-divisions/wind-turbine-design):
- [F. Bertagnolio](https://orbit.dtu.dk/en/persons/franck-bertagnolio) (Senior researcher, [Airfoil and Rotor Design](https://wind.dtu.dk/research/research-divisions/wind-turbine-design/airfoil-and-rotor-design))
- [A.W. Fischer](https://orbit.dtu.dk/en/persons/andreas-wolfgang-fischer) (Senior researcher, [Airfoil and Rotor Design](https://wind.dtu.dk/research/research-divisions/wind-turbine-design/airfoil-and-rotor-design))

---
### Requirements
- [Ubuntu](https://ubuntu.com/) [22.04.2 LTS](https://releases.ubuntu.com/jammy/) or [Windows](https://www.microsoft.com/windows) [10 22H2](https://www.microsoft.com/software-download/windows10ISO)
- [Python](https://www.python.org/) [3.11.3](https://www.python.org/downloads/release/python-3113/)
- Required modules can be installed through:
```
python -m pip install -r requirements
```

---
## Tool input
### Project directory structure
- *-case_name-.aur*
  - Input file for the auralisation (Can be multiple files. Tool will autodetect.)
- *H2model/*
  - The HAWC2 model directory containing everything needed for running the HAWC2 simulation
  - It is strongly recommended to test the HAWC2 model before running this tool,\
  as error handling may not be as nice as native HAWC2
- *atm/*
  - Directory containingg information about the atmosphere used to run cases. \
  Will be automatically generated.
- *spectrograms/*
  - Directory containing csv files with generated spectrograms. \
  Will be automatically generated.
- *pickles/*
  - Directory containing Python [compressed pickle](https://pypi.org/project/compress-pickle/) files. \
  Will be automatically generated.

### Input file structure
Below is the general structure of the input (*.aur*) files that should be used. \
Input code blocks and the variables inside these blocks can be placed in any order.
```
name -str- ; A name for the case defined in this file
;
begin conditions ;
    ; ----------------------------------------------------------------------------------
    ; This section defines the operating conditions of the turbine
    ; Used in both auralisation and HAWC2 simulations
    ; ----------------------------------------------------------------------------------
    hub_pos -float-,-float-,-float- ; Position of the wind turbine hub x,y,z (m)
    rotor_radius -float- ;            Wind turbine rotor radius (m)
    rotor_rpm -float- ;               Operating rotational speed of turbine (RPM)
    ;
    wsp -float- ;           Wind speed (m/s) at z_wsp height
    z_wsp -float- ;         Height (m)) at which wsp is defined
    z0_wsp -float- ;        Roughness height (m) for the wind speed profile
    ;
    groundtemp -float- ;    Ground level air temperature (celcius)
    groundpres -float- ;    Ground level air pressure (Pa)
    humidity -float- ;      Air relative humidity (%)
    ;
    delta_t -float- ; Defines the auralisation time step (s)
end conditions ;
;
begin HAWC2;
    ; ----------------------------------------------------------------------------------
    ; Define operating parameters for HAWC2 model, as would normally 
    ; be defined in .htc files. The .htc file(s) of the model should only contain the 
    ; turbine model.
    ; See HAWC2 manual for more info. Use .htc file syntax.
    ; ----------------------------------------------------------------------------------
    ;
    ; ----------------------------------------------------------------------------------
    ; First is the wind block, where all parameters should be defined per the manual
    ; ----------------------------------------------------------------------------------
    begin wind ;
        ; SEE HAWC2 MANUAL
    end wind;
    ;
    ; ----------------------------------------------------------------------------------
    ; Define which aero noise models to use. See HAWC2 manual
    ; All other parameters are set by the tool.
    ; ----------------------------------------------------------------------------------
    begin aero_noise ;
       noise_start_end_time -float- -float- ; Define this to be approx. (4/3) * (single rotation)
       noise_deltat -float- ;
       turbulent_inflow_noise -int- ;
       inflow_turbulence_intensity -float- ;
       surface_roughness -float- ;
       trailing_edge_noise -int- ;
       bldata_filename -str- ;
       stall_noise -int- ;
       tip_noise -int- ;
    end aero_noise ;
    ;
    ; ----------------------------------------------------------------------------------
    ; Other parameters
    ; ----------------------------------------------------------------------------------
    htc_name -str- ;      Name of the .htc file in the H2model folder (without ".htc")
    hawc2_path -float- ;  File path where the HAWC2 executable is located
    n_obs -int- ;         Number of observer points in HAWC2 (must be <256)
    ;
end HAWC2;
;
begin source ;
    n_rays -int- ;          Defines the number of sound rays (per time step) used for propagation
    blade_percent -float- ; Defines r = blade_percent * R at which the source is assumed to be located.
    scope -str- ;           Selects the noise model result to load ('All', 'TI', 'TE', 'ST', 'TP')
    radius_factor -float- ; Scaling factor for the radius of the sound source sphere
end source ;
;
begin propagation ;
    n_threads -int- ;        Defines the number of compute threads used for propagating sound rays
    models -str-,-str-,... ; Defines which propagation effect models to apply ('spherical', 'atmosphere', )
    pickle -int- ;           Save SoundRays to pickle files for later use (0 for no, or 1 for yes)
    unpickle -int- ;         Use saved SoundRays for reception and reconstruction model
end propagation ;
;
begin reception ;
    save_spectrogram -int- ;        Save the resulting spectrograms (0 for no, or 1 for yes)
    load_spectrogram -int- ;        Load previously generated spectrograms (0 for no, or 1 for yes)
    mode -str- ;                    Binaural rendering mode ('mono' or 'stereo')
    ;
    ; ----------------------------------------------------------------------------------
    ; Define the receiver point(s) in this block (can be multiple)
    ; ----------------------------------------------------------------------------------
    begin receiver ;
        index -int- ;                 Indexing number >=0
        pos -float-,-float-,-float- ; Location x,y,z (m) in HAWC2 global coordinates
        rotation -float- ;            Head rotation of the receiver (clockwise positive, from the y-axis) (deg)
    end receiver ;
    ;
end reception ;
;
begin reconstruction ;
    f_s_desired -int- ; Desired sample frequency of the output audio file
    overlap -int- ;     Amount of overlap between istft time segments
    wav_norm -float- ;  Pressure to normalise the WAV files to (Pa) (Recomended 1 Pa)
    t_audio -float- ;   Time duration if the output audio file
    model -str- ;       Select the signal reconstruction model ('random', 'gla', )
end reconstruction ;
``` 

---
### Code Structure
- *main.py*  
  - Runs the auralisation tool.

- *source_model.py*
  - Module containing the source model.

- *propagation_model.py*
  - Module containing the *ray-tracing / Gaussian beam*  sound propagation model.

- *reception_model.py*
  - Module containing the reception model.

- *reconstruction_model.py*
  - Module containing the reconstruction model.

- *helper_functions/*  
  - Package containing all functions required by the main code, but not directly related to the main code:

> - *coordinate_systems.py*  
>   - Definitions of the used coordinate systems.
> - *data_structures.py*  
>   - Definitions of special datastructures to be used.
> - *funcs.py*
>   - Fun little module with homeless functions and the list of constants.
> - *geometry.py*
>   - A module for geometrical definitions and operations.
> - *hrtf.py*  
>   - Definition of the MIT measured HRTF function.
> - *in_out.py*  
>   - Specialised I/O functions for data files used in this code.
> - *isa.py*  
>   - Definition of the ISO standard atmosphere (*ISO 2533-1975*).
> - *data/*  
>   - Folder with all data files used by the helper functions.

- *plots/*
  - Plots generated from the helper functions for the report.

```
Get-Content NTK/H2model/log/aeroload_noise.log –Wait
```
