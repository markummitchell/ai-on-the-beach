#!/usr/bin/python3

import csv
from datetime import datetime
from mpl_toolkits.basemap import Basemap, cm
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.io import netcdf
from time import sleep

# Some settings
is3d = True

FILENAME = 'etopo1_bedrock.nc'
LINECOLOR = 'lime' # https://matplotlib.org/examples/color/named_colors.html
PAUSESECONDS = 0.00001 # Nonzero value seems to be required

# Based on https://matplotlib.org/basemap/users/examples.html. The parameter
# mmap=False prevents issues related to closing plots/files when arrays are
# still open
f = netcdf.netcdf_file (FILENAME, 'r', mmap = False)
elecdf = f.variables ['Band1']
loncdf = f.variables ['lon']
latcdf = f.variables ['lat']
crscdf = f.variables ['crs'] # Do not know what this array contains, other than 1 character strings
# limits
elemin = np.min (elecdf.data)
elemax = np.max (elecdf.data)
lonmin = np.min (loncdf [:])
latmin = np.min (latcdf [:])
lonmax = np.max (loncdf [:])
latmax = np.max (latcdf [:])
print ("input limits: " + \
       str (lonmin) + "<lon<" + str (lonmax) + " " + \
       str (latmin) + "<lat<" + str (latmax) + " " + \
       str (elemin) + "<ele<" + str (elemax))

# 2D -----------------------------------------------------------------------------
lon = loncdf.data
plt.plot (lon)
plt.ylabel ('longitude')
#plt.show()

lat = latcdf.data
plt.plot (lat)
plt.ylabel ('latitude')
#plt.show()

# 3D -----------------------------------------------------------------------------
# Constructor args, like ellipsoid, are at https://media.readthedocs.org/pdf/basemaptutorial/latest/basemaptutorial.pdf.
# ellps values are at https://github.com/erdc/pyproj/blob/master/src/pj_ellps.c:
#    default (seems to be WGS84)
#    GRS67
#    WGS84
# GEBCO data is based on SRTM30 according to GEBCO_2014.html,
# but Basemap seems to be based on WGS84 according to https://media.readthedocs.org/pdf/basemaptutorial/latest/basemaptutorial.pdf

if is3d:
    map = Basemap (projection = 'stere', lon_0 = lonmin, lat_0 = 90, lat_ts = latmin, \
                   llcrnrlat=latmin, urcrnrlat=latmax, \
                   llcrnrlon=lonmin, urcrnrlon=lonmax, \
                   rsphere=6371200., ellps = 'GRS67', resolution='l', area_thresh=1000)
else:
    map = Basemap (projection = 'merc', lon_0 = lonmin, lat_0 = 90, lat_ts = latmin, \
                   llcrnrlat=latmin, urcrnrlat=latmax, \
                   llcrnrlon=lonmin, urcrnrlon=lonmax, \
                   rsphere=6371200., ellps = 'GRS67', resolution='l', area_thresh=1000)    
    
# The following lines are sometimes skipped because the land coordinates seem to be in  a different
# reference frame than the bathysphere data, leading to misaligned land-sea boundaries
showLines = True
if showLines:
    map.drawcoastlines (linewidth = 0.25)
    map.drawcountries (linewidth = 0.25)
    map.fillcontinents (color = 'coral', lake_color = 'aqua')

# These are always drawn
map.drawmapboundary (fill_color = 'aqua')
map.drawmeridians (np.arange (0, 360, 1))
map.drawparallels (np.arange (-90, 90, 1))

# limits are -5443 to 1298 (elemin to elemax)
contours = np.array([-5500., -5000., -4500., -4000., -3500., -3000., -2500., -2000., -1500., -1000., -500., 0., 500., 1000., 1500.])

# Convert netcdf arrays to regular array. Actually, we will generate
# the lon and lat 2d arrays using map.makegrid. Note that there is a subtle
# nonlinearity in the latitude versus row number - as seen in lat.csv file below
ele = np.zeros ((elecdf.shape[0], elecdf.shape[1]))
lon = np.zeros ((elecdf.shape[0], elecdf.shape[1]))
lat = np.zeros ((elecdf.shape[0], elecdf.shape[1]))
for i in range (ele.shape[0]):
    for j in range (ele.shape[1]):
        ele [i] [j] = elecdf [j] [i]
        lon [i] [j] = loncdf [i]
        lat [i] [j] = latcdf [j]
x, y = map (lon, lat)
cs = map.contourf (x, y, ele, contours, cmap = cm.s3pcpn)

# Color scale
cbar = map.colorbar (cs,location = 'right', pad = '5%')
cbar.set_label ('Elevation (meters)')

# Show circle at northernmost point in nova scotia which is northwest of Meat Point, according
# to https://stackoverflow.com/questions/49134634/how-to-draw-circle-in-basemap-or-add-artiste
#circle = Circle (xy = map (-60.593352, 47.041354), radius = (map.ymax - map.ymin) / 60, fill = False)
#plt.gca().add_patch (circle)

# Labels
plt.xlabel ('longitude')
plt.ylabel ('latitude')
plt.title (FILENAME)

# Waypoints
lonLast = None
latLast = None
with open ('../Beneath The Waves - Blue Shark Atlantic - Data Jan 21, 2019.csv', 'r') as f:
    reader = csv.DictReader (f)
    headers = reader.fieldnames
    for line in reader:
        latStr = line ['Latitude']
        lonStr = line ['Longitude']
        if latStr != '' and lonStr != '':
            lon = float (lonStr)
            lat = float (latStr)
            if lonLast != None:
                lons = [lonLast, lon]
                lats = [latLast, lat]
                map.plot (lons, lats, linewidth = 1.2, color = LINECOLOR, latlon = True)
            lonLast = lon
            latLast = lat
            # Besides the delay, this also seems trigger the display, and without it nothing appears
            plt.pause (PAUSESECONDS)
