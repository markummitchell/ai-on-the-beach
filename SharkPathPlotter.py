#!/usr/bin/python3

import csv
import datetime
from mpl_toolkits.basemap import Basemap, cm
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import operator
import pandas as pd
from scipy.interpolate import griddata
from scipy.io import netcdf
from time import sleep

COL_DATE = 1
# svg format gives lossless quality but huge 33MB files for gallagher
IMGEXTENSION = 'png'

def drawBathysphere (map, elecdf, loncdf, latcdf, contours):
    # Convert netcdf arrays to regular array, with components transposed
    ele = np.zeros ((elecdf.shape[1], elecdf.shape[0]))
    lon = np.zeros ((elecdf.shape[1], elecdf.shape[0]))
    lat = np.zeros ((elecdf.shape[1], elecdf.shape[0]))
    for i in range (elecdf.shape[1]):
        for j in range (elecdf.shape[0]):
            ele [i] [j] = elecdf [j] [i]
            lon [i] [j] = loncdf [i]
            lat [i] [j] = latcdf [j]
    x, y = map (lon, lat)
    # cmap = cm.s3pcpn = blue, black, white brown
    # cmap = cm.Blues_r
    # cmap = plt.cm.get_cmap('Blues'))
    # cmap = plt.cm.get_cmap('RdBu'))
    # cmap = plt.cm.get_cmap('Dark2'))        
    cs = map.contourf (x, y, ele, contours, cmap = plt.cm.get_cmap('Blues'))

    # Color scale
    cbar = map.colorbar (cs, location = 'right', pad = '5%')
    cbar.set_label ('Elevation (meters)')

def drawCircle(map):
    # Show circle at northernmost point in nova scotia which is northwest of Meat Point, according
    # to https://stackoverflow.com/questions/49134634/how-to-draw-circle-in-basemap-or-add-artiste
    circle = Circle (xy = map (-60.593352, 47.041354), radius = (map.ymax - map.ymin) / 60, fill = False)
    plt.gca().add_patch (circle)

def drawContinents(map):
    map.drawcoastlines (linewidth = 0.25)
    map.drawcountries (linewidth = 0.25)
    #map.fillcontinents (color = 'coral', lake_color = 'aqua') # Overwrites land portions of quivers drawn later
    map.drawmapboundary (fill_color = 'aqua')
    map.drawmeridians (np.arange (0, 360, 10))
    map.drawparallels (np.arange (-90, 90, 10))
    
def drawCurrent (map, lon, lat, u, v):
    map.quiver (lon, lat, u, v)

def drawDeclinationContours (map, lon, lat, dec):
    map.contour (lon, lat, dec)
    
def drawDeclinationVectors (map, lon, lat, udec, vdec):
    map.quiver (lon, lat, udec, vdec, scale = 12, color = 'r') # Larger scale for smaller vectors

def drawLine (map, id, lonLast, latLast, lon, lat, moves, miles):
    LINECOLOR = 'lime' # https://matplotlib.org/examples/color/named_colors.html    
    if id in lonLast:
        # Draw a line
        lons = [lonLast [id], lon]
        lats = [latLast [id], lat]
        map.plot (lons, lats, linewidth = 1.2, color = LINECOLOR, latlon = True)
        moves += 1
        miles += milesMoved (lonLast [id], latLast [id], lon, lat)
    return moves, miles

def drawSharkPath (map, isBetterMap, FILESHARK, fKml, waypoints, lonmin, lonmax, latmin, latmax):    
    # Shark waypoints. There can be multiple ids so lonLast and latLast are vectors indexed by id
    lonLast = {}
    latLast = {}
    datLast = waypoints [0][COL_DATE]
    imgLast = 0
    moves = 0
    miles = 0
    kmlPoints = {} # First index is id, second index is date, value is '<lon>,<lat>' string
    for row in waypoints:
        id  = row [0]
        dat = row [COL_DATE]
        # For debugging
        #if dat > pd.to_datetime ('10/21/2015', format='%m/%d/%Y'):
        #    break;
        lon = row [2]
        lat = row [3]
        moves, miles = drawLine (map, id, lonLast, latLast, lon, lat, moves, miles)
        imgLast, moves, miles = moveDaysForward (isBetterMap, FILESHARK, datLast, dat, \
                                                 lonmin, lonmax, latmin, latmax, imgLast, moves, miles)
        kmlFileWrite (fKml, kmlPoints, id, dat, lon, lat)
        
        lonLast [id] = lon
        latLast [id] = lat
        datLast = dat
    kmlFileWriteAll (fKml, kmlPoints)
    kmlFileClose (fKml)
    
def kmlFileClose (fKml):
    fKml.write ('</Document>\n')
    fKml.write ('</kml>\n')

def kmlFileOpen (idsForKml):    
    FILEKML = 'outputs/gallagher.kml'
    fKml = open (FILEKML, 'w')
    fKml.write ('<?xml version="1.0" encoding="UTF-8"?>\n')
    fKml.write ('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
    fKml.write ('<Document>\n')
    fKml.write ('  <LookAt>\n')
    fKml.write ('    <longitude>-68.38</longitude>\n')
    fKml.write ('    <latitude>41.27</latitude>\n')
    fKml.write ('    <altitude>0</altitude>\n')
    fKml.write ('    <range>687414</range>\n')
    fKml.write ('    <tilt>0.0303</tilt>\n')
    fKml.write ('    <heading>0</heading>\n')                
    fKml.write ('  </LookAt>\n')
    return fKml

def kmlFileWrite (fKml, kmlPoints, id, dat, lon, lat):
    LIFETIME = datetime.timedelta(hours=4)
    if not id in kmlPoints:
        kmlPoints [id] = {}
    kmlPoints [id] [dat] = '{},{}' . format (lon, lat)
    
def kmlFileWriteAll (fKml, kmlPoints):
    # Colors to disinguish sharks and also match points with paths
    colors = ['ffff0000', \
              'ff00ff00', \
              'ff0000ff', \
              'ffff00ff', \
              'ff00ffff', \
              'ffffff00', \
              'ffffffff']
    for indexColor in range (len (colors)):
        color = colors [indexColor]
        fKml.write ('  <StyleMap id="mapcolor{}">\n' . format (indexColor))
        fKml.write ('    <Pair>\n')
        fKml.write ('      <key>normal</key>\n')
        fKml.write ('      <styleUrl>#color{}</styleUrl>\n' . format (indexColor))
        fKml.write ('    </Pair>\n')
        fKml.write ('    <Pair>\n')
        fKml.write ('      <key>highlight</key>\n')
        fKml.write ('      <styleUrl>#color{}</styleUrl>\n' . format (indexColor))
        fKml.write ('    </Pair>\n')                
        fKml.write ('  </StyleMap>\n')
        fKml.write ('  <Style id="color{}">\n' . format (indexColor))
        fKml.write ('    <BalloonStyle>\n')
        fKml.write ('      <displayMode>{}</displayMode>\n' . format ('default'))                        
        fKml.write ('      <bgColor>{}</bgColor>\n' . format (color))                
        fKml.write ('      <textColor>{}</textColor>\n' . format (color))        
        fKml.write ('      <color>{}</color>\n' . format (color))
        fKml.write ('    </BalloonStyle>\n')
        fKml.write ('    <IconStyle>\n')
        fKml.write ('      <color>{}</color>\n' . format (color))                
        fKml.write ('    </IconStyle>\n')
        fKml.write ('    <LabelStyle>\n')
        fKml.write ('      <color>{}</color>\n' . format (color))                
        fKml.write ('    </LabelStyle>\n')                
        fKml.write ('    <LineStyle>\n')
        fKml.write ('      <color>{}</color>\n' . format (color))
        fKml.write ('      <width>{}</width>\n' . format (2.0))        
        fKml.write ('    </LineStyle>\n')
        fKml.write ('    <PolyStyle>\n')
        fKml.write ('      <color>{}</color>\n' . format (color))                
        fKml.write ('    </PolyStyle>\n')                        
        fKml.write ('  </Style>\n')
    indexId = 0
    indexColor = 0
    for id in kmlPoints.keys():
        pointsForId = kmlPoints [id]

        datFirst = list (pointsForId.keys())[0]
        datLast = list (pointsForId.keys()) [len (pointsForId) - 1]
        numberDays = (datLast - datFirst).days
        fKml.write ('  <Folder>\n')
        fKml.write ('    <name>{}_{}days</name>\n' . format (id, numberDays))
        fKml.write ('    <Folder>\n')
        fKml.write ('      <name>waypoints</name>\n')
        indexPoint = 0
        for dat in pointsForId.keys():
            # First write out points for this id            
            fKml.write ('      <Placemark>\n')
            fKml.write ('        <name>{}_{}</name>\n' . format (id, indexPoint))
            indexPoint += 1
            fKml.write ('        <styleUrl>#mapcolor{}</styleUrl>\n' . format (indexColor))
            fKml.write ('        <TimeStamp>{}</TimeStamp>\n' . format (dat.strftime ('%Y-%m-%dT%H:%M:%SZ'))) # 2000-01-01T09:00:00Z
            fKml.write ('        <Point>\n')
            fKml.write ('          <coordinates>{},0</coordinates>\n' . format (pointsForId [dat])) # No spaces are allowed!
            fKml.write ('        </Point>\n')
            fKml.write ('      </Placemark>\n')
        fKml.write ('    </Folder>\n')            
        # Now write out path for this id        
        fKml.write ('    <Placemark>\n')
        fKml.write ('      <name>path</name>\n')
        fKml.write ('      <styleUrl>#color{}</styleUrl>\n' . format (indexColor))
        fKml.write ('      <LineString>\n')
        fKml.write ('        <tessellate>1</tessellate>\n')
        fKml.write ('        <altitudeMode>absolute</altitudeMode>\n')
        fKml.write ('        <coordinates>');
        pointsForId = kmlPoints [id]        
        for dat in pointsForId:
            fKml.write ('{} ' . format (pointsForId [dat]));
        fKml.write ('</coordinates>');            
        fKml.write ('      </LineString>\n')                
        fKml.write ('    </Placemark>\n')            
        fKml.write ('  </Folder>\n')
        indexId += 1
        indexColor = (indexColor + 1) % len (colors)

    indexColor = 0
    for id in kmlPoints.keys():

        indexColor = (indexColor + 1) % len (colors)
        
def loadBathysphere (isBetterMap):
    print ("loadBathysphere")
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
    print ("loadCurrent")
    FILENAMECURRENT = 'data.nodc.noaa.gov/ofs_atl.t00z.n000.20170321.grb.grib2.nc'
    f = netcdf.netcdf_file (FILENAMECURRENT, 'r', mmap = False)

    loncdf = f.variables ['Longitude_of_Presure_Point_surface'] # 1684x1200
    latcdf = f.variables ['Latitude_of_Presure_Point_surface'] # 1684x1200
    ucdf = f.variables ['Barotropic_U_velocity_entire_ocean_single_layer'] # 1x1684x1200
    vcdf = f.variables ['Barotropic_V_velocity_entire_ocean_single_layer'] # 1x1684x1200
    nx = len (loncdf.data)
    ny = len (loncdf.data[0])

    # Griddata wants 1D versus 2D. We need to discard all points associated with u=NaN or v=NaN
    pointsRaw = []
    uRaw = []
    vRaw = []
    for i in range (nx):
        for j in range (ny):
            u = ucdf.data[0][i][j]
            v = vcdf.data[0][i][j]
            if not np.isnan(u) and not np.isnan(v):
                pointsRaw.append ([loncdf.data[i][j], latcdf.data[i][j]])
                uRaw.append (u)
                vRaw.append (v)

    # Desired grid is computed using list comprehension
    lonstep = 10 # Degrees
    latstep = 10 # Degrees
    LONLATSTEP = 3
    desired = [[i, j] for j in range (latmin, latmax + 1, LONLATSTEP) for i in range (lonmin, lonmax + 1, LONLATSTEP)]

    # Interpolate
    u = griddata (pointsRaw, uRaw, desired, method = 'linear', fill_value = 0.0)
    v = griddata (pointsRaw, vRaw, desired, method = 'linear', fill_value = 0.0)

    # Numpy nicely allows extracting individual components
    nx = len (desired)
    ny = len (desired  [0])
    lon = np.array (desired) [:, 0]
    lat = np.array (desired) [:, 1]

    # Convert to basemap units
    lonMapped, latMapped = map (lon, lat)

    return lonMapped, latMapped, u, v

def loadDeclination (map, lonmin, lonmax, latmin, latmax):
    print ("loadDeclination")
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

    # Declination is indexed by (lat,lon) so we tranpose it
    dec = z.data.transpose ()
    
    # We throw away points outside the desired range. This makes no difference with contour
    # plots but seems to be required for quiver plots to work at all
    iMap = []
    jMap = []
    COUNT = 4
    lonMargin = (lonmax - lonmin) / (2 * COUNT)
    latMargin = (latmax - latmin) / (2 * COUNT)
    lonDelta = lon [1][0] - lon [0][0]
    latDelta = lat [0][1] - lat [0][0]
    lonCount = 1 + int ((lonmax - lonmin) / lonDelta)
    latCount = 1 + int ((latmax - latmin) / latDelta)
    lonStep = int (lonCount / (COUNT - 1))
    latStep = int (latCount / (COUNT - 1))
    for i in range (nx):
        if lonmin + lonMargin < lon [i][0]:
            # We found the first so loop through the good ones
            while i < nx:
                if lon [i][0] > lonmax - lonMargin:
                    break;
                iMap.append (i)
                i += lonStep
            break
    for j in range (ny):
        if latmin + latMargin < lat [0][j]:
            # We found the first so loop through the good ones
            while j < ny:
                if lat [0][j] > latmax - latMargin:
                    break;
                jMap.append (j)
                j += latStep
            break
    nxFiltered = len (iMap)
    nyFiltered = len (jMap)
    lonFiltered = np.zeros ((nxFiltered, nyFiltered))
    latFiltered = np.zeros ((nxFiltered, nyFiltered))
    decFiltered = np.zeros ((nxFiltered, nyFiltered))
    udecFiltered = np.zeros ((nxFiltered, nyFiltered))
    vdecFiltered = np.zeros ((nxFiltered, nyFiltered))    
    for i in range (nxFiltered):
        for j in range (nyFiltered):
            lonFiltered [i][j] = lon [iMap [i]] [jMap [j]]
            latFiltered [i][j] = lat [iMap [i]] [jMap [j]]
            ang = dec [iMap [i]] [jMap [j]]
            decFiltered [i][j] = ang
            udecFiltered [i][j] = math.sin (ang * np.pi / 180)
            vdecFiltered [i][j] = math.cos (ang * np.pi / 180)

    lonMapped, latMapped = map (lonFiltered, latFiltered)

    return lonMapped, latMapped, decFiltered, udecFiltered, vdecFiltered

def loadMapParameters (isBetterMap):
    if isBetterMap:
        FILEBATHYSPHERE = 'maps.ngdcc.noaa.gov/etopo1_bedrock_-80_-35_10_45.nc'
        ELEVARIABLE = 'Band1'        
    else:
        FILEBATHYSPHERE = 'GRIDONE_2D_-70.0_35.0_-55.0_50.0.nc'
        ELEVARIABLE = 'elevation'        
    return FILEBATHYSPHERE, ELEVARIABLE

def loadSharkPath (FILESHARK):
    print ("loadSharkPath")
    waypoints = []
    idsForKml = {}
    with open (FILESHARK, 'r') as f:
        reader = csv.DictReader (f, delimiter = '\t') # Read tab separated values
        headers = reader.fieldnames
        for line in reader:
            idStr = line ['ptt']
            dateStr = line ['dt']
            latStr = line ['lat']
            lonStr = line ['lon']
            classStr = line ['class']

            # Only keep the good data
            if classStr == '0' or classStr == '1' or classStr == '2' or classStr == '3':
                id = int (idStr)
                idsForKml [id] = True
                lon = float (lonStr)
                lat = float (latStr)
                dat = pd.to_datetime (dateStr, format='%m/%d/%y %H:%M')
                waypoints.append ([id, dat, lon, lat]) # COL_DATE must match position here!
    # For some reason the data is not in chronological order so sort it
    waypoints = sorted (waypoints, key = lambda row:row[COL_DATE])
    return waypoints, idsForKml

def main():
    # Some settings
    is3d = False
    isBetterMap = True
    contours = np.array([-11000., -10000., -9000., -5000., -3000., -1500,
                         0., 1500., 3000.])

    # Grids should be preprocessed to have these bounds
    lonmin = -80
    lonmax = -35
    latmin = 10
    latmax = 45
    FILESHARK = 'gallagher_4feb19/Gallagher_BTW_Tracks_4Feb19.tsv'
    
    map = makeMap (is3d, lonmin, lonmax, latmin, latmax)
    drawContinents (map)        
    elecdf, loncdf, latcdf = loadBathysphere (isBetterMap)
    lonCurrent, latCurrent, uCurrent, vCurrent = loadCurrent (map, lonmin, lonmax, latmin, latmax)
    lonDeclination, latDeclination, declination, udec, vdec = loadDeclination (map, lonmin, lonmax, latmin, latmax)
    waypoints, idsForKml = loadSharkPath (FILESHARK)
    fKml = kmlFileOpen (idsForKml)
    #drawDeclinationContours (map, lonDeclination, latDeclination, declination)
    drawBathysphere (map, elecdf, loncdf, latcdf, contours)
    drawDeclinationVectors (map, lonDeclination, latDeclination, udec, vdec) # Fails if before drawBathysphere
    drawCurrent (map, lonCurrent, latCurrent, uCurrent, vCurrent) # Fails if before drawBathysphere
    drawSharkPath (map, isBetterMap, FILESHARK, fKml, waypoints, lonmin, lonmax, latmin, latmax) # Main plotting loop
    
    # Scale values are pixels with 958x719 for dpi=150, or 1916x1438 for dpi=300 (too big for Discord).
    # setpts value is smaller for faster movement and smaller file
    print ('Convert using: ffmpeg -i outputs/gallagher%03d.{} -vf scale=958x719 -filter:v "setpts=0.4*PTS" outputs/gallagher.mp4' . \
           format (IMGEXTENSION))
    
def makeMap(is3d, lonmin, lonmax, latmin, latmax):

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

    # Labels
    plt.xlabel ('longitude')
    plt.ylabel ('latitude')

    return map

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

def moveDaysForward (isBetterMap, FILESHARK, datLast, dat, lonmin, lonmax, latmin, latmax, imgLast, moves, miles):
    PAUSESECONDS = 0.00001 # Nonzero value seems to be required    
    FILEBATHYSPHERE, ELEVARIABLE = loadMapParameters (isBetterMap)    
    if datLast.dayofyear != dat.dayofyear:

        # Besides the delay, this also seems trigger the display, and without it nothing appears.
        # We do this when the date has changed once, but before the loop below which will
        # repeat (after the first iteration) for days in which nothing happens so a plot would be unhelpful
        plt.pause (PAUSESECONDS)
        
        # Move N days forward, where N=1,2,3...
        for dayofyear in range (datLast.dayofyear, dat.dayofyear):
            imgFile = ('outputs/gallagher{:03d}.{}' . format (imgLast, IMGEXTENSION)) 
            datFile = datetime.datetime(dat.year, 1, 1) + datetime.timedelta(dayofyear - 1)
            movesAndMiles = '{} moves, {} miles' . format (moves, int (miles + 0.5))
            print (str (datFile) + ": " + imgFile + " (" + movesAndMiles + ")")
            plt.title (FILEBATHYSPHERE + '\n' + FILESHARK + '\n' + datFile.strftime ('%Y/%m/%d') + \
                       ' [' + movesAndMiles + '] ' + \
                       str (int (lonmin - 0.1)) + '<lon<' + str (int (lonmax - 0.1)) + ' ' + \
                       str (int (latmin + 0.1)) + '<lat<' + str (int (latmax + 0.1)))
            plt.savefig (imgFile)
            imgLast += 1
            moves = 0
            miles = 0
    return imgLast, moves, miles

main()

