#!/usr/bin/python3

from datetime import datetime, timedelta
from google_drive_downloader import GoogleDriveDownloader as gdd
import io
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import numpy as np
import os
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.io import netcdf
import shutil
import tempfile

earthRadiusKilometers = 6378.16

def appendDepthAndDepthChanges(df, interpBathysphere, interpCurrentU, interpCurrentV):
    # Create arrays with depth and downstream/upstream depth changes for each lon/lat coordinate
    df['depth'] = pd.Series (0., index=df.index)
    df['depthChangeDownstream'] = pd.Series (0., index=df.index)
    df['depthChangeUpstream'] = pd.Series (0., index=df.index)
    lonLast = {}  # Indexed by shark id
    indexTo = 0
    indexDepth = df.columns.get_loc('depth')
    indexDepthChangeDownstream = df.columns.get_loc('depthChangeDownstream')
    indexDepthChangeUpstream = df.columns.get_loc('depthChangeUpstream')
    for idRow, row in df.iterrows():
        lon = row['long']
        lat = row['lat']

        # Perform interpolations
        depth = interpBathysphere([lon, lat])[0]
        u = interpCurrentU([lon, lat])[0]
        v = interpCurrentV([lon, lat])[0]

        # Get downstream and upstream points
        lonDownstream, latDownstream = separatedPointsFromSeparation(lon, lat, u, v)
        lonUpstream, latUpstream = separatedPointsFromSeparation(lon, lat, -1.0 * u, -1.0 * v)

        # More interpolations at downstream and upstream points
        depthDownstream = interpBathysphere([lonDownstream, latDownstream])[0]
        depthUpstream = interpBathysphere([lonUpstream, latUpstream])[0]

        # Save results
        idx = df.index[indexTo]
        df.at [idx, 'depth'] = depth
        df.at [idx, 'depthChangeDownstream'] = depthDownstream - depth
        df.at [idx, 'depthChangeUpstream'] = depth - depthUpstream

        indexTo += 1

    return df

def appendDirectionAndLocationQuantities (df, interpDeclination):
    # Create arrays using points I-1 and I:
    # 1) absolute bearing angle (degrees), 0=magnetic north and +90=eastward
    # 2) time between successive locations (hours)
    # 3) distance between successive locations (kilometers)
    # The last two quantities may be useful to account for how readings just a short time apart
    # (minutes) may be highly correlated, but readings far apart in time (months) will be lacking
    # much important information so maybe the correlations are less reliable
    df['bearing'] = pd.Series (0., index=df.index)
    df['timeStep'] = pd.Series (timedelta(0), index=df.index)
    df['distanceStep'] = pd.Series (0., index=df.index)
    indexTo = 0
    indexBearing = df.columns.get_loc('bearing')
    indexTimeStep = df.columns.get_loc('timeStep')
    indexDistanceStep = df.columns.get_loc('distanceStep')
    rowLast = {} # Indexed by shark id
    for idRow, row in df.iterrows():
        idShark = row ['shark_id']
        lon = row ['long']
        lat = row ['lat']
        time = row ['loc_date']

        # Perform calculations
        bearing = 0.
        timeStep = time - time
        distanceStep = 0.
        if idShark in rowLast:
            lonLast = rowLast [idShark] ['long']
            latLast = rowLast [idShark] ['lat']
            timeLast = rowLast [idShark] ['loc_date']

            # This code assumes duplicate id/timestamp rows have been removed
            bearing = bearingFromSeparatedPoints (interpDeclination, lonLast, latLast, lon, lat)
            timeStep = time - timeLast
            distanceStep = separationFromSeparatedPoints (lonLast, latLast, lon, lat)
        idx = df.index[indexTo]
        df.at [idx, 'bearing'] = bearing
        df.at [idx, 'timeStep'] = timeStep
        df.at [idx, 'distanceStep'] = distanceStep
        indexTo += 1
        rowLast [idShark] = row

    return df

def bearingFromSeparatedPoints (interpDeclination, lon0, lat0, lon1, lat1):
    # Inverse of separatedPointsFromSeparation.
    # For small enough separations, we can ignore the distortion caused by the
    # longitude lines joining at the north pole, and just convert angular separation into distance
    angleDeclination = interpDeclination ([lon0, lat0])
    # Angle from north pole, ignoring magnetic declination. Note that angle measured from
    # eastward direction would be atan2 (lat1 - lat0, lon1 - lon0)
    angleTrueNorth = 180. * math.atan2 (lon1 - lon0, lat1 - lat0) / np.pi
    angleMagneticNorth = angleTrueNorth - angleDeclination
    return angleMagneticNorth

def check (interp, title):
    lonmin = -80
    lonmax = -35
    latmin = 10
    latmax = 45
    lons = np.linspace (lonmin + 1, lonmax - 1, 240)
    lats = np.linspace (latmin + 1, latmax - 1, 240)
    lons, lats = np.meshgrid (lons, lats)
    lonLat =  np.stack ((lons, lats), axis = -1)
    values = interp (lonLat)
    plt.title (title)
    plt.pcolor (lons, lats, values)
    plt.colorbar()
    plt.show()

def loadBathysphere ():
    print ("loadBathysphere")

    # etopo1_bedrock_-80_-35_10_45.nc
    units = 'meters'    
    googleIdBathysphere = '10VqbV2oNUVcvS6lLP3FekVlFM4LUJj5o' # Extracted from share url
    tmpBathysphere = tempfile.NamedTemporaryFile (suffix = '.nc', \
                                                  prefix = 'tempBathysphere', \
                                                  delete = True) # Need file name but not the file or gdd fails
    tmpBathysphere.close()
    # Download the file from url and save it locally
    gdd.download_file_from_google_drive (file_id = googleIdBathysphere,
                                         dest_path = tmpBathysphere.name)
    with netcdf.netcdf_file (tmpBathysphere.name, 'r', mmap = False) as f:
        loncdf = f.variables ['lon']
        latcdf = f.variables ['lat']
        elecdf = f.variables ['Band1']        
        crscdf = f.variables ['crs'] # Do not know what this array contains, other than 1 character strings
    
    # Transpose lat/lon to lon/lat
    ele = np.transpose (elecdf.data)

    # Create an interpolator. This is a regular grid so we use a regular grid interpolator that
    # exploits the regularity to achieve the most efficient search
    return units, RegularGridInterpolator ((loncdf.data, latcdf.data), ele)

def loadCurrent():
    print ("loadCurrent")

    # https://data.nodc.noaa.gov/thredds/ncss/ncep/rtofs/2017/201703/ofs.20170321/surface/ofs_atl.t00z.n000.20170321.grb.grib2/dataset.html
    googleIdCurrent = '1ZL2ABGc5uqtBt9DK0_m7CxJPMBgpDrW3' # Extracted from share url
    
    tmpCurrent = tempfile.NamedTemporaryFile (suffix = '.nc', \
                                              prefix = 'tempCurrent', \
                                              delete = True) # Need file name but not the file or gdd fails
    tmpCurrent.close()
    # Download the file from url and save it locally
    gdd.download_file_from_google_drive (file_id = googleIdCurrent, \
                                         dest_path = tmpCurrent.name)
    with netcdf.netcdf_file (tmpCurrent.name, 'r', mmap = False) as f:
        loncdf = f.variables ['Longitude_of_Presure_Point_surface']
        latcdf = f.variables ['Latitude_of_Presure_Point_surface']        
        ucdf = f.variables ['Barotropic_U_velocity_entire_ocean_single_layer'] # 1x1684x1200
        vcdf = f.variables ['Barotropic_V_velocity_entire_ocean_single_layer'] # 1x1684x1200
        units = 'm.s-1'

    # Create interpolators. This is an irregular grid (not constant longitude and latitude points)
    # so an inefficient irregular grid is applied
    nx = loncdf.data.shape[0]
    ny = latcdf.data.shape[1]
    lonlat = []
    u = []
    v = []
    for i in range (nx):
        for j in range (ny):
            lonlat.append ([loncdf[i][j], latcdf[i][j]])
            u.append (ucdf[0][i][j])
            v.append (vcdf[0][i][j])
    return units, \
        LinearNDInterpolator (lonlat, u), \
        LinearNDInterpolator (lonlat, v)

def loadDeclination ():
    print ("loadDeclination")

    # https://maps.ngdc.noaa.gov/viewers/historical_declination/
    units = 'Degrees'        
    googleIdDeclination = '1KL-brszjyfiX7yAp_-ZEBbereOm-26lz' # Extracted from share url    
    tmpDeclination = tempfile.NamedTemporaryFile (suffix = '.nc', \
                                                  prefix = 'tempDeclination', \
                                                  delete = True) # Need file name but not the file or gdd fails
    tmpDeclination.close()
    # Download the file from url and save it locally
    gdd.download_file_from_google_drive (file_id = googleIdDeclination,
                                         dest_path = tmpDeclination.name)
    with netcdf.netcdf_file (tmpDeclination.name, 'r', mmap = False) as f:
        loncdf = f.variables ['x']
        latcdf = f.variables ['y']
        deccdf = f.variables ['z']

    # Transpose lat/lon to lon/lat
    dec = np.transpose (deccdf.data)
    
    # Create an interpolator. This is a regular grid so we use a regular grid interpolator that
    # exploits the regularity to achieve the most efficient search
    return units, RegularGridInterpolator ((loncdf.data, latcdf.data), dec)

def loadSharkPath():
    print ("loadSharkPath")        
    # Upload the CSV Here
    # from google.colab import files
    # uploaded = files.upload()

    # # Replace the filename here if you have saved the CSV as a different
    # df = pd.read_csv(io.BytesIO(uploaded[
    #     'Beneath The Waves - Blue Shark Atlantic - Data Jan 21, 2019.csv'])) 

    googleFile = 'https://drive.google.com/uc?id=1XtdF630BEDDv-ixbZ6cE4RJlbVwukiUU&export=download'
    print ('Downloading {}... ' . format (googleFile), end='')    
    df = pd.read_csv(googleFile)
    print ('Done.')

    # Next step is to clean the Data and drop the columns we don't need
    COLUMN_MAPPING = {
        'Shark ID': 'shark_id',
        'Prg No.': 'prg_no',
        'Latitude': 'lat',
        'Longitude': 'long',
        'Loc. quality': 'loc_quality',
        'Loc. date': 'loc_date',
        'Loc. type': 'loc_type',
        'Altitude': 'alt',
        'Pass': 'pass',
        'Sat.': 'satellite',
        'Frequency': 'freq',
        'Msg Date': 'msg_date',
        'Comp.': 'comp',
        'Msg': 'msg',
        '> - 120 DB': 'db_120_gt',
        'Best level': 'best_level',
        'Delta freq.': 'delta_freq',
        'Long. 1': 'long_1', 
        'Lat. sol. 1': 'late_sol_1', 
        'Long. 2': 'long_2',
        'Lat. sol. 2': 'lat_sol_2', 
        'Loc. idx': 'loc_idx', 
        'Nopc': 'nopc', 
        'Error radius': 'err_radius', 
        'Semi-major axis': 'semi_major_axis',
        'Semi-minor axis': 'semi_minor_axis', 
        'Ellipse orientation': 'ellipse_orientation', 
        'GDOP': 'gdop'
      }

    # Drop Columns with no location data
    cleaned_df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Drop Columns with bad location data quality
    cleaned_df = cleaned_df.loc[cleaned_df['Loc. quality'].apply(str.isdigit)]
    
    # Select the important columns
    cleaned_df = cleaned_df[list(COLUMN_MAPPING.keys())]
    
    # Rename the columns to be more pythonic
    cleaned_df = cleaned_df.rename(columns=COLUMN_MAPPING)

    # Cast to datetime values to datetime
    cleaned_df['loc_date'] = cleaned_df.loc_date.apply(lambda x: datetime.strptime(x, '%m/%d/%y %H:%M'))

    # Save to csv for more detailed inspection
    cleaned_df.to_csv ('outputs/cleaned_df_duplicates_included.csv')
    
    # Remove successive entries that are so close in time that the longitude
    # and latitude coordinates are unchanged. This is experimental
    cleaned_df = cleaned_df.drop_duplicates (subset = ['shark_id', 'long', 'lat'])

    # Save to csv for more detailed inspection
    cleaned_df.to_csv ('outputs/cleaned_df_duplicates_removed.csv')

    return cleaned_df

def main():
    unitsBathysphere, interpBathysphere = loadBathysphere()
    #check (interpBathysphere, 'Bathysphere ({})' . format (unitsBathysphere))
    
    unitsCurrent, interpCurrentU, interpCurrentV = loadCurrent()
    #check (interpCurrentU, 'CurrentU ({})' . format (unitsCurrent))
    #check (interpCurrentV, 'CurrentV ({})' . format (unitsCurrent))
    
    unitsDeclination, interpDeclination = loadDeclination()
    #check (interpDeclination, 'Declination ({})' . format (unitsDeclination))
    
    df = loadSharkPath()
    
    df = appendDepthAndDepthChanges (df, interpBathysphere, interpCurrentU, interpCurrentV)
    df = appendDirectionAndLocationQuantities (df, interpDeclination)

    outfile = 'outputs/complete_df.csv'
    print ('Writing csv file {}' . format (outfile))
    df.to_csv (outfile)
    
def separatedPointsFromSeparation (lon, lat, u, v):
    # Inverse of separationFromSeparatedPoints.
    # Google Map investigation of Greater Bank bathysphere data suggests the 2 points used
    # upstream and downstream (in terms of the current) should be about 10 miles from the
    # center point
    separationKilometers = 10.0 * (1.6 / 1.0)

    # Make u and v into a unit vector (u,v) which will be multiplied by angleSeparation below
    # to get a (lon,lat) separation vector with a specified great circle angle
    uvmag = math.sqrt (u * u + v * v)
    u = u / uvmag
    v = v / uvmag
    
    # For small enough separationKilometers, we can ignore the distortion caused by the
    # longitude lines joining at the north pole, and just add longitude and latitude
    # deltas calculated as simply proportional to u and v
    angleSeparation = separationKilometers / earthRadiusKilometers # Great circle angle in radians
    lonNew = lon + angleSeparation * u * 180. / np.pi
    latNew = lat + angleSeparation * v * 180. / np.pi
    return lonNew, latNew

def separationFromSeparatedPoints (lon0, lat0, lon1, lat1):
    # Inverse of separatedPointsFromSeparation. Returns great circle angle between two vectors.
    # For small enough separations, we can ignore the distortion caused by the
    # longitude lines joining at the north pole, and just convert angular separation into distance
    angleSep = math.sqrt ((lon1 - lon0) * (lon1 - lon0) + \
                          (lat1 - lat0) * (lat1 - lat0))
    return (angleSep * np.pi / 180.) * earthRadiusKilometers

main()
