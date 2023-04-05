# Auralisation of Realistic Synthetic Wind Turbine Noise for Psychoacoustic Listening Experiments

---
### Code repository for the MSc. thesis by Josephine Siebert Pockelé.

---
To obtain the degree of Master of Science at the *Delft University of Technology*  
and the *Technical University of Denmark*.  
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
- [Python](https://www.python.org/) [3.10.10](https://www.python.org/downloads/release/python-31010/)
- Required external modules in *./python_310_reqs*
```
pip install -r python_310_reqs
```

### Structure
- *main.py*  
  - ~~Runs the auralisation tool~~ A mess at this point
- *helper_functions/*  
  - Contains all functions required by the main code:
> - *coordinate_systems.py*  
>   - Definitions of the used coordinate systems
> - *data_structures.py*  
>   - Definitions of special datastructures to be used
> - *hrtf.py*  
>   - Definition of the MIT measured HRTF function
> - *in_out.py*  
>   - Specialised I/O functions for data files used in this code
> - *isa.py*  
>   - Definition of the ISO standard atmosphere (*ISO 2533-1975*)
> - *data/*  
>   - Folder with all data files used by the helper functions
- *hawc2_out/*  
  - Contains the HAWC2 output files used for the report
- *plots/*
  - Plots generated from the helper functions for the report
