import csv
from datetime import datetime
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

lats = []
lons = []
msgDates = []
with open ('Beneath The Waves - Blue Shark Atlantic - Data Jan 21, 2019.csv', 'r') as f:
    reader = csv.DictReader (f)
    headers = reader.fieldnames
    for line in reader:
        latStr = line ['Latitude']
        lonStr = line ['Longitude']
        datStr = line ['Msg Date']
        if latStr != '' and lonStr != '' and datStr != '':
            lats.append (float (latStr))
            lons.append (float (lonStr))
            msgDates.append (datetime.strptime (datStr, '%m/%d/%y %H:%M'))

# Based on https://matplotlib.org/basemap/users/examples.html

map = Basemap (projection = 'ortho', lat_0 = 42, lon_0 = -69.4, resolution = 'l')
map.drawcoastlines (linewidth = 0.25)
map.drawcountries (linewidth = 0.25)
map.fillcontinents (color = 'coral', lake_color = 'aqua')
map.drawmapboundary (fill_color = 'aqua')
map.drawmeridians (np.arange (0, 360, 30))
map.drawparallels (np.arange (-90, 90, 30))
# compute native map projection coordinates of lat/lon grid
map.plot (lons, lats, linewidth = 2.5, color = 'r', latlon = True)
plt.title ("{} through {}" . format (msgDates [0], msgDates [-1]))
plt.show()
