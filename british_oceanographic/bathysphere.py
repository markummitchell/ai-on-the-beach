#!/usr/bin/python3

import csv
from datetime import datetime
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf

FILENAME = 'GRIDONE_2D_-70.0_35.0_-55.0_50.0.nc'
LINECOLOR = 'lime' # https://matplotlib.org/examples/color/named_colors.html

# Based on https://matplotlib.org/basemap/users/examples.html

f = netcdf.netcdf_file (FILENAME, 'r')
elecdf = f.variables ['elevation']
loncdf = f.variables ['lon']
latcdf = f.variables ['lat']
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

map = Basemap (projection = 'stere', lon_0 = lonmin, lat_0 = 90, lat_ts = latmin, \
               llcrnrlat=latmin, urcrnrlat=latmax, \
               llcrnrlon=lonmin, urcrnrlon=lonmax, \
               rsphere=6371200., ellps = 'GRS67', resolution='l', area_thresh=1000)
# The following lines are skipped because the land coordinates seem to be in  a different
# reference frame than the bathysphere data, leading to misaligned land-sea boundaries
#map.drawcoastlines (linewidth = 0.25)
#map.drawcountries (linewidth = 0.25)
#map.fillcontinents (color = 'coral', lake_color = 'aqua')
map.drawmapboundary (fill_color = 'aqua')
map.drawmeridians (np.arange (0, 360, 30))
map.drawparallels (np.arange (-90, 90, 30))

# convert netcdf arrays to regular array. Actually, we will generate
# the lon and lat 2d arrays using map.makegrid - so there is an implicit
# assumption that the lon and lat arrays coming from netcdf are evenly
# spaced from the lon/lat lower limits to the lon/lat upper limits, respectively
ele = np.zeros ((elecdf.shape[0], elecdf.shape[1]))
lon = np.zeros ((elecdf.shape[0], elecdf.shape[1]))
lat = np.zeros ((elecdf.shape[0], elecdf.shape[1]))
for i in range (ele.shape[0]):
    for j in range (ele.shape[1]):
        ele [i] [j] = elecdf [i] [j]
        lon [i] [j] = loncdf [i]
        lat [i] [j] = latcdf [j]
        
# limits are -5443 to 1298 (elemin to elemax)
contours = np.array([-5500., -5000., -4500., -4000., -3500., -3000., -2500., -2000., -1500., -1000., -500., 0., 500., 1000., 1500.])

lon, lat = map.makegrid (elecdf.shape[0], elecdf.shape[1])
lonmin = np.min (lon)
lonmax = np.max (lon)
latmin = np.min (lat)
latmax = np.max (lat)
print ("grid limits: " + str (lonmin) + "<lon<" + str (lonmax) + " " + str (latmin) + "<lat<" + str (latmax))
x, y = map (lon, lat)
cs = map.contourf (x, y, ele, contours, cmap = cm.s3pcpn)

# color scale
cbar = map.colorbar (cs,location = 'right', pad = '5%')
cbar.set_label ('Elevation (meters)')

# waypoints
lats = []
lons = []
with open ('../Beneath The Waves - Blue Shark Atlantic - Data Jan 21, 2019.csv', 'r') as f:
    reader = csv.DictReader (f)
    headers = reader.fieldnames
    for line in reader:
        latStr = line ['Latitude']
        lonStr = line ['Longitude']
        if latStr != '' and lonStr != '':
            lats.append (float (latStr))
            lons.append (float (lonStr))
map.plot (lons, lats, linewidth = 2.5, color = LINECOLOR, latlon = True)

# show
plt.xlabel ('longitude')
plt.ylabel ('latitude')
plt.title (FILENAME)
plt.show()
