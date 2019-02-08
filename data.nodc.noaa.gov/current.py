#!/usr/bin/python3

import csv
import datetime
from mpl_toolkits.basemap import Basemap, cm
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
from scipy.io import netcdf
from time import sleep

latmin=35
latmax=50
lonmin=-70
lonmax=-55
map = Basemap (projection = 'merc', lon_0 = lonmin, lat_0 = 90, lat_ts = latmin, \
               llcrnrlat=latmin, urcrnrlat=latmax, \
               llcrnrlon=lonmin, urcrnrlon=lonmax, \
               rsphere=6371200., ellps = 'GRS67', resolution='l', area_thresh=1000)    
map.drawcoastlines (linewidth = 0.25)
map.drawcountries (linewidth = 0.25)
map.fillcontinents (color = 'coral', lake_color = 'aqua')
f = netcdf.netcdf_file ('ofs_atl.t00z.n000.20170321.grb.grib2.nc', 'r', mmap = False)
ucdf = f.variables ['u-component_of_current_hybrid_layer']
vcdf = f.variables ['v-component_of_current_hybrid_layer']    
loncdf = f.variables ['Longitude_of_U_Wind_Component_of_Velocity_surface']
latcdf = f.variables ['Latitude_of_U_Wind_Component_of_Velocity_surface']    
nxTooMany = ucdf[0][0].shape[0]
nyTooMany = ucdf[0][0].shape[1]
lonTooMany=[]
latTooMany=[]
uTooMany=[]
vTooMany=[]
for xTooMany in range (nxTooMany):
    for yTooMany in range (nyTooMany):
        if not math.isnan(ucdf[0][0][xTooMany][yTooMany]) and not math.isnan(vcdf[0][0][xTooMany][yTooMany]):
            if lonmin<=loncdf[xTooMany][yTooMany] and loncdf[xTooMany][yTooMany]<=lonmax and \
               latmin<=latcdf[xTooMany][yTooMany] and latcdf[xTooMany][yTooMany]<=latmax:
                lonTooMany.append(loncdf[xTooMany][yTooMany])
                latTooMany.append(latcdf[xTooMany][yTooMany])
                uTooMany.append(ucdf[0][0][xTooMany][yTooMany])
                vTooMany.append(vcdf[0][0][xTooMany][yTooMany])
nx = 20
ny = 20
lon = np.zeros((nx, ny))
lat = np.zeros((nx, ny))
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
for x in range (nx):
    for y in range (ny):
        lon [x][y] = lonmin + (lonmax - lonmin) * float (x) / float (nx - 1)            
        lat [x][y] = latmin + (latmax - latmin) * float (y) / float (ny - 1)
        # Find closest lon/lat point
        isFirst = True
        closestDistanceSquared = 0
        for iTooMany in range (len (lonTooMany)):
            lonT = lonTooMany[iTooMany]
            latT = latTooMany[iTooMany]
            # Preliminary filtering for efficiency
            if lon [x][y] - 0.1 < lonT and lonT < lon [x][y] + 0.1 and lat [x][y] - 0.1 < latT and latT  < lat [x][y] + 0.1:
                # Slower processing of close points
                distanceSquared = (lon [x][y] - lonT) * (lon [x][y] - lonT) + (lat [x][y] - latT) * (lat [x][y] - latT)
                if isFirst or distanceSquared < closestDistanceSquared:
                    # Save this as the best so far
                    isFirst = False
                    closestDistanceSquared = distanceSquared
                    uClosest = uTooMany[iTooMany]
                    vClosest = vTooMany[iTooMany]
        if isFirst:
            # All nearby original u and v values must have been NaN
            u [x][y] = 0
            v [x][y] = 0
        else:
            # Save the closest
            u [x][y] = uClosest
            v [x][y] = vClosest
lonMapped, latMapped = map (lon, lat)
map.quiver (lonMapped, latMapped, u, v)
# Labels
plt.xlabel ('longitude')
plt.ylabel ('latitude')
plt.show ()
