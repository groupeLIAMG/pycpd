# pycpd

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](./01_LICENSE.txt)
[![DOI](https://zenodo.org/badge/51318634.svg)](https://zenodo.org/badge/latestdoi/51318634)

Interactive data processing and analysis tool to compute the Curie point depth from aeromagnetic data using the method of Bouligand *et al*. (2009)

The user interface is really a work in progress for now.  More to come soon.

## Requirements

pycpd is programmed in python 3 and was tested on a mac running python 3.8 installed via macports.

The following python modules are needed to run the software
- numpy and scipy
- pandas
- PyQt5
- cartopy (https://scitools.org.uk/cartopy/docs/latest/)
- netCDF4 (https://github.com/Unidata/netcdf4-python)
- pyproj (https://github.com/jswhit/pyproj)
- pyfftw (https://pypi.python.org/pypi/pyFFTW)
- spectrum (http://www.thomas-cokelaer.info/software/spectrum/html/contents.html)

### Cython file

Run the following command in the source directory in order to use the maximum entropy method to estimate the spectra (c code wrapped with cython)

```
python setup.py build_ext --inplace
```
## Data

Examples can be found in the data directory.

### Aeromagnetic Data

- Aeromagnetic data should be gridded on a cartesian grid with spatial units in meters
- Recognized formats are:
    * netCDF (COARDS compliant)
    * USGS sgd grid
- In order to display the map, the user is asked to enter coordinate projection information.  This is done by giving a proj4 string (http://proj4.org), e.g. for coordinates projected in the Lambert conic conformal for Eastern Canada, the string is

```
+proj=lcc +lat_1=49 +lat_2=77 +lat_0=63 +lon_0=-92 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs
```

### Borehole Data

Borehole data should be in a csv file with the same header as given in the Global Heat Flow Database of the International Heat Flow Commission (http://www.heatflow.und.edu/index2.html).  An example file can be found at http://www.heatflow.und.edu/Global2010.csv. The first lines of this file are:
```
Data Number,Codes,Site Name,Latitude,Longitude,Elevation,minD,maxD,No. Temps,Gradient,No. Cond.,Conductivity,No.Heat Prod.,Heat Prod.,Heat Flow,No. sites,Year of Pub.,Reference,Comments,,
1,,SMU-KG2,44.4637,-111.7322,1987,28,66,,81,2,1.88,,,,,1983,Brott_etal1983,Williams_etal1995,,
2,,SMU-SP3,44.3278,-112.2128,1795,10,99,,55,5,2.06,,,,,1983,Brott_etal1983,Brott_etal1983,,
3,,SMU-SP2,44.3678,-112.1432,1859,25,70,,46,5,1.67,,,,,1983,Brott_etal1983,Brott_etal1983,,
```

The script `mk_db.py` can be used to extract the heat flow data for the area corresponding to your aeromagnetic data grid, and store it in a python shelf that pycpd understands.

## References
```
@Article{bouligand09,
  Title                    = {Mapping {C}urie temperature depth in the western {U}nited {S}tates with a fractal model for crustal magnetization},
  Author                   = {Bouligand, Claire and Glen, Jonathan M. G. and Blakely, Richard J.},
  Journal                  = {Journal of Geophysical Research: Solid Earth},
  Year                     = {2009},
  Month                    = nov,
  Number                   = {B11},
  Pages                    = {B11104--},
  Volume                   = {114},
  DOI                      = {10.1029/2009JB006494},
  ISSN                     = {0148-0227},
  Keywords                 = {aeromagnetic compilation, Curie temperature isotherm, western United States, Great Basin, 1517 Geomagnetism and Paleomagnetism: Magnetic anomalies: modeling and interpretation, 3255 Mathematical Geophysics: Spectral analysis, 4440 Nonlinear Geophysics: Fractals and multifractals, 5418 Planetary Sciences: Solid Surface Planets: Heat flow, 0903 Exploration Geophysics: Computational methods: potential fields, geothermie},
  Owner                    = {giroux},
  Publisher                = {AGU},
  Timestamp                = {2012.05.02},
  URL                      = {http://dx.doi.org/10.1029/2009JB006494}
}
```
