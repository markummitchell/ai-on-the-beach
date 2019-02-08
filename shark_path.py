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

def drawBathysphere (map, elecdf, loncdf, latcdf, contours):
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
    cbar = map.colorbar (cs, location = 'right', pad = '5%')
    cbar.set_label ('Elevation (meters)')

def drawCircle(map):
    # Show circle at northernmost point in nova scotia which is northwest of Meat Point, according
    # to https://stackoverflow.com/questions/49134634/how-to-draw-circle-in-basemap-or-add-artiste
    #circle = Circle (xy = map (-60.593352, 47.041354), radius = (map.ymax - map.ymin) / 60, fill = False)
    #plt.gca().add_patch (circle)
    pass

def drawCurrent (map, lon, lat, u, v):
    map.quiver (lon, lat, u, v)

def drawDeclination (map, lon, lat, dec):
    map.contour (lon, lat, dec)
    
def drawMap(is3d, lonmin, lonmax, latmin, latmax):

    # 3D map
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
    map.drawcoastlines (linewidth = 0.25)
    map.drawcountries (linewidth = 0.25)
    map.fillcontinents (color = 'coral', lake_color = 'aqua')
    map.drawmapboundary (fill_color = 'aqua')
    map.drawmeridians (np.arange (0, 360, 1))
    map.drawparallels (np.arange (-90, 90, 1))

    # Labels
    plt.xlabel ('longitude')
    plt.ylabel ('latitude')

    return map

def drawSharkPath (map, isBetterMap, lonmin, lonmax, latmin, latmax):
    FILEBATHYSPHERE, ELEVARIABLE = loadMapParameters (isBetterMap)    
    FILESHARK = 'Beneath The Waves - Blue Shark Atlantic - Data Jan 21, 2019.csv'
    PAUSESECONDS = 0.00001 # Nonzero value seems to be required
    LINECOLOR = 'lime' # https://matplotlib.org/examples/color/named_colors.html

    # Shark waypoints
    lonLast = None
    latLast = None
    datLast = pd.to_datetime ('8/1/18 12:00')
    imgLast = 0
    moves = 0
    miles = 0
    with open (FILESHARK, 'r') as f:
        reader = csv.DictReader (f)
        headers = reader.fieldnames
        for line in reader:
            latStr = line ['Latitude']
            lonStr = line ['Longitude']
            dateStr = line ['Msg Date']
            if latStr != '' and lonStr != '' and dateStr != '':
                lon = float (lonStr)
                lat = float (latStr)
                dat = pd.to_datetime (dateStr, format='%m/%d/%y %H:%M')
                if lonLast != None:
                    # Draw a line
                    lons = [lonLast, lon]
                    lats = [latLast, lat]
                    map.plot (lons, lats, linewidth = 1.2, color = LINECOLOR, latlon = True)
                    moves += 1
                    miles += milesMoved (lonLast, latLast, lon, lat)
                if datLast.dayofyear != dat.dayofyear:
                    # Move N days forward, where N=1,2,3...
                    for dayofyear in range (datLast.dayofyear, dat.dayofyear):
                        svgFile = ('outputs/shark_path{:03d}.svg' . format (imgLast)) # svg format gives lossless quality
                        datFile = datetime.datetime(dat.year, 1, 1) + datetime.timedelta(dayofyear - 1)
                        movesAndMiles = '{} moves, {} miles' . format (moves, int (miles + 0.5))
                        print (str (datFile) + ": " + svgFile + " (" + movesAndMiles + ")")
                        plt.title (FILEBATHYSPHERE + '\n' + FILESHARK + '\n' + datFile.strftime ('%Y/%m/%d') + \
                                   ' [' + movesAndMiles + '] ' + \
                                   str (int (lonmin - 0.1)) + '<lon<' + str (int (lonmax - 0.1)) + ' ' + \
                                   str (int (latmin + 0.1)) + '<lat<' + str (int (latmax + 0.1)))
                        plt.savefig (svgFile)
                        imgLast += 1
                        moves = 0
                        miles = 0
                lonLast = lon
                latLast = lat
                datLast = dat
                # Besides the delay, this also seems trigger the display, and without it nothing appears
                plt.pause (PAUSESECONDS)
                
def loadBathysphere (isBetterMap):
    FILEBATHYSPHERE, ELEVARIABLE = loadMapParameters (isBetterMap)

    # Based on https://matplotlib.org/basemap/users/examples.html. The parameter
    # mmap=False prevents issues related to closing plots/files when arrays are
    # still open
    f = netcdf.netcdf_file (FILEBATHYSPHERE, 'r', mmap = False)
    elecdf = f.variables [ELEVARIABLE]
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

    # 2D -----------------------------------------------------------------------------
    lon = loncdf.data
    #plt.plot (lon)
    #plt.ylabel ('longitude')
    #plt.show()

    lat = latcdf.data
    #plt.plot (lat)
    #plt.ylabel ('latitude')
    #plt.show()

    return elecdf, loncdf, latcdf

def loadCurrent(map, lonmin, lonmax, latmin, latmax):
    FILENAMECURRENT = 'data.nodc.noaa.gov/ofs_atl.t00z.n000.20170321.grb.grib2.nc'
    f = netcdf.netcdf_file (FILENAMECURRENT, 'r', mmap = False)

    # We have available way too many points - 55,000 just in our desired lon/lat range alone. So
    # we filter out just what we want here so later processing is not unacceptably slow
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
                    
    # Still too many, so lets make an nx x ny grid
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

    return lonMapped, latMapped, u, v

def loadDeclination (map):
    FILEDECLINATION = 'ngdc.noaa.gov-geomag/D_Grid_mf_2020.grd'
    f = netcdf.netcdf_file (FILEDECLINATION, 'r', mmap = False)
    x = f.variables ['x']
    y = f.variables ['y']
    z = f.variables ['z']
    nx = len (x.data)
    ny = len (y.data)
    lon = np.zeros ((nx, ny))
    lat = np.zeros ((nx, ny))
    for i in range (nx):
        for j in range (ny):
            lon [i] [j] = x.data [i]
            lat [i] [j] = y.data [j]
    lonMapped, latMapped = map (lon, lat)
    # Declination is indexed by (lat,lon) so we tranpose it
    return lonMapped, latMapped, z.data.transpose()

def loadMapParameters (isBetterMap):
    if isBetterMap:
        FILEBATHYSPHERE = 'maps.ngdcc.noaa.gov/etopo1_bedrock.nc'
        ELEVARIABLE = 'Band1'        
    else:
        FILEBATHYSPHERE = 'GRIDONE_2D_-70.0_35.0_-55.0_50.0.nc'
        ELEVARIABLE = 'elevation'        
    return FILEBATHYSPHERE, ELEVARIABLE

def main():
    # Some settings
    is3d = False
    isBetterMap = True
    contours = np.array([-5500., -5000., -4500., -4000., -3500., -3000., -2500., -2000., \
                         -1500., -1000., -500., 0., 500., 1000., 1500.]) # limits are -5443 to 1298 (elemin to elemax)

    # Grids should be preprocessed to have these bounds
    lonmin = -70
    lonmax = -55
    latmin = 35
    latmax = 50
    
    map = drawMap (is3d, lonmin, lonmax, latmin, latmax)
    elecdf, loncdf, latcdf = loadBathysphere (isBetterMap)
    lonCurrent, latCurrent, uCurrent, vCurrent = loadCurrent (map, lonmin, lonmax, latmin, latmax)
    lonDeclination, latDeclination, declination = loadDeclination (map)
    drawBathysphere (map, elecdf, loncdf, latcdf, contours)
    drawCurrent (map, lonCurrent, latCurrent, uCurrent, vCurrent)
    drawDeclination (map, lonDeclination, latDeclination, declination)
    drawSharkPath (map, isBetterMap, lonmin, lonmax, latmin, latmax)

    # Scale values are pixels with 958x719 for dpi=150, or 1916x1438 for dpi=300 (too big for Discord)
    print ("Convert using: ffmpeg -r 1 -i outputs/shark_path%03d.svg -vf scale=958x719 -r 10 outputs/shark_path.mp4")

def milesMoved (lonLast, latLast, lon, lat):
    # Return the distance between two (lon,lat) points
    RADIUSEARTH = 3959 # miles
    lonLast = lonLast * np.pi / 180.0
    latLast = latLast * np.pi / 180.0
    lon = lon * np.pi / 180.0
    lat = lat * np.pi / 180.0
    xLast = RADIUSEARTH * math.cos (lonLast) * math.cos (latLast)
    yLast = RADIUSEARTH * math.sin (lonLast) * math.cos (latLast)
    zLast = RADIUSEARTH * math.sin (latLast)
    x = RADIUSEARTH * math.cos (lon) * math.cos (lat)
    y = RADIUSEARTH * math.sin (lon) * math.cos (lat)
    z = RADIUSEARTH * math.sin (lat)
    distance = math.sqrt ((x - xLast) * (x - xLast) + (y - yLast) * (y - yLast) + (z - zLast) * (z - zLast))
    return distance

main()
