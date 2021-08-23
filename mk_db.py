#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:53:39 2017

@author: giroux
"""
from cpd import Forage, Grid2d
import numpy as np
import pandas as pd

import shelve


proj4string = '+proj=lcc +lat_1=49 +lat_2=77 +lat_0=63 +lon_0=-92 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs -f %9.9f'

g = Grid2d(proj4string)
g.read_nc('data/Qc_lcc_clipped.nc')

df = pd.read_csv('data/Global2010edit.csv')

# %%

forages = []


for n in np.arange(len(df)):
    try:
        f = Forage(float(df['Latitude'][n]), float(df['Longitude'][n]), proj4string)
    except ValueError:
        continue
    if g.inside(f):
        f.site_name = str(df['Site Name'][n])
        if f in forages:
            ind = forages.index(f)
            f = forages[ind]

        try:
            if not np.isnan(df['Heat Flow'][n]):
                f.Q0.append(df['Heat Flow'][n])
            if not np.isnan(df['Conductivity'][n]):
                f.k.append(df['Conductivity'][n])
            if not np.isnan(df['Heat Prod.'][n]):
                f.A.append(df['Heat Prod.'][n])
        except TypeError as e:
            print(e)
            continue

        if f not in forages:
            forages.append(f)

forages2 = []
for f in forages:

    if len(f.Q0) > 0:
        f.Q0 = np.unique(f.Q0)
    else:
        f.Q0 = np.array([])  # make sure we have numpy arrays...
    if len(f.k) > 0:
        f.k = np.unique(f.k)
    else:
        f.k = np.array([])
    if len(f.A) > 0:
        f.A = np.unique(f.A)
    else:
        f.A = np.array([])

    if f.Q0.size != 0 or f.k.size != 0 or f.A.size != 0:
        forages2.append(f)

db = shelve.open('data/forages', 'n')
db['forages'] = forages2
db.close()
