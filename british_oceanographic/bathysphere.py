#!/usr/bin/python3

from datetime import datetime
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf

# Based on https://matplotlib.org/basemap/users/examples.html

f = netcdf.netcdf_file ('in.nc', 'r')
elecdf = f.variables ['elevation']
loncdf = f.variables ['lon']
latcdf = f.variables ['lat']
# limits
lonmin = min (loncdf [:])
latmin = min (latcdf [:])
lonmax = max (loncdf [:])
latmax = max (latcdf [:])

map = Basemap (projection = 'stere', lat_0 = latmin, lon_0 = lonmin, lat_ts = latmin, \
               llcrnrlat=latmin, urcrnrlat=latmax, \
               llcrnrlon=lonmin, urcrnrlon=lonmax, \
               rsphere=6371200., resolution='l', area_thresh=10000)
map.drawcoastlines (linewidth = 0.25)
map.drawcountries (linewidth = 0.25)
map.fillcontinents (color = 'coral', lake_color = 'aqua')
#map.drawmapboundary (fill_color = 'aqua')
map.drawmeridians (np.arange (0, 360, 30))
map.drawparallels (np.arange (-90, 90, 30))

# convert netcdf arrays to regular array. Actually, we will generate
# the lon and lat 2d arrays using map.makegrid - so there is an implicit
# assumption that the lon and lat arrays coming from netcdf are evenly
# spaced from the lon/lat lower limits to the lon/lat upper limits, respectively
ele = np.zeros ((elecdf.shape[0], elecdf.shape[1]))
for i in range (ele.shape[0]):
    for j in range (ele.shape[1]):
        ele [i] [j] = elecdf [i] [j]
# compute map projection coodinates                
x, y = map.makegrid (ele.shape[0], ele.shape[1])
# limits are -5443 to 1298
contours = [-5500, -5000, -4500, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0, 500, 1000, 1500]
cs = map.contourf (x, y, ele, contours, cmap = cm.s3pcpn)

#plt.title ("{} through {}" . format (msgDates [0], msgDates [-1]))
plt.show()
