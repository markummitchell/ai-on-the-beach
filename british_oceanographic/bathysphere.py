#!/usr/bin/python3

from datetime import datetime
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf

# Based on https://matplotlib.org/basemap/users/examples.html

map = Basemap (projection = 'ortho', lat_0 = 42, lon_0 = -69.4, resolution = 'l')
map.drawcoastlines (linewidth = 0.25)
map.drawcountries (linewidth = 0.25)
map.fillcontinents (color = 'coral', lake_color = 'aqua')
map.drawmapboundary (fill_color = 'aqua')
map.drawmeridians (np.arange (0, 360, 30))
map.drawparallels (np.arange (-90, 90, 30))
f = netcdf.netcdf_file ('in.nc', 'r')
elecdf = f.variables ['elevation']
loncdf = f.variables ['lon']
latcdf = f.variables ['lat']
# convert netcdf arrays to regular array
ele = np.zeros ((elecdf.shape[0], elecdf.shape[1]))
lon = np.zeros ((loncdf.shape[0]))
lat = np.zeros ((latcdf.shape[0]))
for i in range (ele.shape[0]):
    for j in range (ele.shape[1]):
        ele [i] [j] = elecdf [i] [j]
for i in range (lon.shape[0]):
    lon [i] = loncdf [i]
for i in range (lat.shape[0]):
    lat [i] = latcdf [i]
# compute map projection coodinates                
x, y = map (lon, lat)
# limits are -5443 to 1298
contours = [-5500, -5000, -4500, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0, 500, 1000, 1500]
cs = map.contourf (x, y, ele, contours, cmap = cm.s3pcpn)

plt.title ("{} through {}" . format (msgDates [0], msgDates [-1]))
plt.show()
