# Auralisation of Modelled Wind Turbine Noise for Psychoacoustic Listening Experiments

---
### Code repository for the M.Sc. thesis by Josephine Siebert Pockelé.

---
To obtain the degree of Master of Science at the *Delft University of Technology* and the *Technical University of Denmark*.  
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
- Ubuntu 22.04.2 LTS (Testing on Windows 10 22H2 WIP)
- [Python](https://www.python.org/) [3.10.10](https://www.python.org/downloads/release/python-31010/)
- Required modules can be installed through:
```
pip install -r python_310_reqs
```

---
## Tool input
### Project folder structure
- *case.aur*
  - Input file for the auralisation (Can be multiple files. Tool will autodetect.)
- *H2model/*
  - The HAWC2 model folder containing everything needed for running the HAWC2 simulation
  - It is strongly recommended to test the HAWC2 model before running this tool, as error handling may not be as nice as native HAWC2

### Input file structure
Below is the general structure of the input files that should be used
```
begin conditions ;
    ; ----------------------------------------------------------------------------------
    ; This section defines the operating conditions of the turbine
    ; Used in both auralisation and HAWC2 simulations
    ; ----------------------------------------------------------------------------------
    hub_height -float- ;    Wind turbine hub height (m)
    rotor_radius -float- ;  Wind turbine rotor radius (m)
    ; rotor_rpm -float- ;   Operating rotational speed of turbine (RPM)
    ;
    wsp -float- ;           Wind speed (m/s) at z_wsp height
    z_wsp -float- ;         Height (m)) at which wsp is defined
    z0_wsp -float- ;        Roughness height (m) for the wind speed profile
    ;
    groundtemp -float- ;    Ground level air temperature (celcius)
    groundpres -float- ;    Ground level air pressure (Pa)
    ;
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
       noise_start_end_time -float- -float- ; Define this to be one rotor rotation
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
    htc_name -str- ; Name of the .htc file in the H2model folder
    hawc2_path -float- ; File path where the HAWC2 executable is located
    n_obs -int- ; must be <256
    ;
end HAWC2;
;
begin source ;
    n_rays -int- ; Defines the number of sound rays used for propagation
end source ;
;
begin propagation ;
    n_threads -int- ; Defines the number of threads used for propagating sound rays
end propagation ;
;
begin reception ;
    ; --- WIP ---
end reception ;
;
begin reconstruction ;
    ; --- WIP ---
end reconstruction ;
``` 

---
### Code Structure
- *main.py*  
  - ~~Runs the auralisation tool~~ A mess, at this point :(

- *propagation_model.py*
  - Module containing the *ray-tracing / Gaussian beam*  sound propagation model.

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

- *hawc2_out/*  
  - Contains the HAWC2 output files used for the report.

- *plots/*
  - Plots generated from the helper functions for the report.

```
Get-Content NTK/H2model/log/aeroload_noise.log –Wait
```
