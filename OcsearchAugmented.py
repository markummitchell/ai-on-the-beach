#!/usr/bin/env python
# coding: utf-8

# In[52]:


from datetime import datetime, timedelta
from google_drive_downloader import GoogleDriveDownloader as gdd
import io
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import numpy as np
import os
import pandas as pd
import requests
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.io import netcdf
import shutil
import tempfile

earthRadiusKilometers = 6378.16
COL_BEARING = 'bearing'
COL_DATE = 'tagDate'
COL_DEPTH = 'depth'
COL_DEPTH_CHANGE_DOWNSTREAM = 'depthChangeDownstream'
COL_DEPTH_CHANGE_UPSTREAM = 'depthChangeUpstream'
COL_DISTANCESTEP = 'distanceStep'
COL_LATITUDE = 'latitude'
COL_LONGITUDE = 'longitude'
COL_SHARK_ID = 'id' # May also be shark_id
COL_TIMESTEP = 'timeStep'

def appendDepthAndDepthChanges(df, interpBathysphere, interpCurrentU, interpCurrentV):
    # Create arrays with depth and downstream/upstream depth changes for each lon/lat coordinate
    df[COL_DEPTH] = pd.Series (0., index=df.index)
    df[COL_DEPTH_CHANGE_DOWNSTREAM] = pd.Series (0., index=df.index)
    df[COL_DEPTH_CHANGE_UPSTREAM] = pd.Series (0., index=df.index)
    lonLast = {}  # Indexed by shark id
    indexTo = 0
    latMin = -39.9999
    latMax = 49.9999
    for idRow, row in df.iterrows():
        lon = float (row[COL_LONGITUDE])
        lat = float (row[COL_LATITUDE])
        # Keep in bounds
        lat = min (max (lat, latMin), latMax)
        # Perform interpolations
        depth = interpBathysphere([lat, lon])[0]
        u = interpCurrentU([lon, lat])[0]
        v = interpCurrentV([lon, lat])[0]

        if math.isnan (u) or math.isnan (v):
            # Out of the defined current area
            depthDownstream = depth
            depthUpstream = depth
        else:
            # Get downstream and upstream points
            lonDownstream, latDownstream = separatedPointsFromSeparation(lon, lat, u, v)
            lonUpstream, latUpstream = separatedPointsFromSeparation(lon, lat, -1.0 * u, -1.0 * v)
            # Keep in bounds
            lonDownstream = lonDownstream % 180.
            lonUpstream = latUpstream % 180.
            latDownstream = min (max (latDownstream, latMin), latMax)
            latUpstream = min (max (latUpstream, latMin), latMax)            
            # More interpolations at downstream and upstream points
            depthDownstream = interpBathysphere([latDownstream, lonDownstream])[0]
            depthUpstream = interpBathysphere([latUpstream, lonUpstream])[0]
            
        # Save results
        idx = df.index[indexTo]
        df.at [idx, COL_DEPTH] = depth
        df.at [idx, COL_DEPTH_CHANGE_DOWNSTREAM] = depthDownstream - depth
        df.at [idx, COL_DEPTH_CHANGE_UPSTREAM] = depth - depthUpstream
        
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
    df[COL_BEARING] = pd.Series (0., index=df.index)
    df[COL_TIMESTEP] = pd.Series (timedelta(0), index=df.index)
    df[COL_DISTANCESTEP] = pd.Series (0., index=df.index)
    indexTo = 0
    indexBearing = df.columns.get_loc(COL_BEARING)
    indexTimeStep = df.columns.get_loc(COL_TIMESTEP)
    indexDistanceStep = df.columns.get_loc(COL_DISTANCESTEP)
    rowLast = {} # Indexed by shark id
    DATE_FORMAT = '%d %B %Y' # '7 March 2019' would be '%d %B %Y'
    print ('columns.keys={}' . format (df.columns.values))
    for idRow, row in df.iterrows():
        idShark = int (row [COL_SHARK_ID])
        lon = float (row [COL_LONGITUDE])
        lat = float (row [COL_LATITUDE])
        time = datetime.strptime (row [COL_DATE], DATE_FORMAT)
                              
        # Perform calculations
        bearing = 0.
        timeStep = time - time
        distanceStep = 0.
        if idShark in rowLast:
            lonLast = float (rowLast [idShark] [COL_LONGITUDE])
            latLast = float (rowLast [idShark] [COL_LATITUDE])
            timeLast = datetime.strptime (rowLast [idShark] [COL_DATE], DATE_FORMAT)
            
            # This code assumes duplicate id/timestamp rows have been removed
            bearing = bearingFromSeparatedPoints (interpDeclination, lonLast, latLast, lon, lat)
            timeStep = time - timeLast
            distanceStep = separationFromSeparatedPoints (lonLast, latLast, lon, lat)
        idx = df.index[indexTo]
        df.at [idx, COL_BEARING] = bearing
        df.at [idx, COL_TIMESTEP] = timeStep
        df.at [idx, COL_DISTANCESTEP] = distanceStep
        indexTo += 1
        rowLast [idShark] = row
        
    return df

def bearingFromSeparatedPoints (interpDeclination, lon0, lat0, lon1, lat1):
    # Inverse of separatedPointsFromSeparation.
    # For small enough separations, we can ignore the distortion caused by the
    # longitude lines joining at the north pole, and just convert angular separation into distance
    
    # HACK! angleDeclination = interpDeclination ([lon0, lat0])
    angleDeclination = 0.
    
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

def loadBathysphereAtlantic ():
    print ("loadBathysphereAtlantic")
    #
    # This function is much faster than loadBathysphereWorld because it covers a small area
    # etopo1_bedrock_-80_-35_10_45.nc
    units = 'meters'
    googleIdBathysphere = '10VqbV2oNUVcvS6lLP3FekVlFM4LUJj5o' # Extracted from share url
    tmpBathysphere = tempfile.NamedTemporaryFile (suffix = '.nc',
                                                  prefix = 'tempBathysphere',
                                                  delete = True) # Need file name but not the file or gdd fails
    tmpBathysphere.close()
    # Download the file from url and save it locally
    gdd.download_file_from_google_drive (file_id = googleIdBathysphere,
                                         dest_path = tmpBathysphere.name)
    with netcdf.netcdf_file (tmpBathysphere.name, 'r', mmap = False) as f:
        loncdf = f.variables ['lon'] # 1D data going from lonmin to lonmax
        latcdf = f.variables ['lat'] # 1D data going from latmin to latmax
        elecdf = f.variables ['Band1'] # 2D data indexed by (lat,lon)       
        crscdf = f.variables ['crs'] # Do not know what this array contains, other than 1 character strings
    #
    # Create an interpolator. This is a regular grid so we use a regular grid interpolator that
    # exploits the regularity to achieve the most efficient search
    return units, RegularGridInterpolator ((latcdf.data, loncdf.data), elecdf.data)

def loadBathysphereWorld ():
    print ("loadBathysphereWorld")
    #
    # etopo1_bedrock_-M_-30_-N_50.nc
    # where (M,N) = (-180,-120) (-120,-60) (-60,0) (0,60) (60,120) (120,180)
    units = 'meters'    
    # Extracted from share url
    googleIdBathyspheres = [
        '1bxT1MuGjpa-gGA-45NQA3hmTrblr_h7R',
        '1r8blFCsLdEvWOyZ80fRNxq_pIYYbAM5m',
        '1eMj03kwp3biK1HCzJKI3TvkA62xihMJm',
        '1rvq8mrm58RQzPZWA_d2vrzxgqeJ28ImO',
        '1GUWrfQ0FBuBGqRhd389tcJ_0_82smhDG',
        '1Fc3xEF4gs0xDVCUdZrNuRVDJx1huXkMf'
    ]
    lons = None
    lats = None
    eles = None
    for googleIdBathysphere in googleIdBathyspheres:
        tmpBathysphere = tempfile.NamedTemporaryFile (suffix = '.nc',
                                                      prefix = 'tempBathysphere',
                                                      delete = True) # Need file name but not the file or gdd fails
        tmpBathysphere.close()
        # Download the file from url and save it locally
        gdd.download_file_from_google_drive (file_id = googleIdBathysphere,
                                             dest_path = tmpBathysphere.name)
        with netcdf.netcdf_file (tmpBathysphere.name, 'r', mmap = False) as f:
            loncdf = f.variables ['lon'] # 1D data going from lonmin to lonmax
            latcdf = f.variables ['lat'] # 1D data going from latmin to latmax
            elecdf = f.variables ['Band1'] # 2D data indexed by (lat,lon)
            crscdf = f.variables ['crs'] # Do not know what this array contains, other than 1 character strings
        #
        # Convert 1D longitude and latitude arrays to 2D since latitude arrays change between chunks
        # so a single 1D array of latitudes across all chunks would not work
        nlon = loncdf.data.shape[0]
        nlat = latcdf.data.shape[0]
        # Aggregate with longitudes changing but latitudes repeating
        if lons is None:
            lons = loncdf.data
            lats = latcdf.data
            eles = elecdf.data
        else:
            # First column(s) of new columns may overlap last column(s) of previous columns which triggers 
            # an error so we delete the last column(s) of previous columns
            nlons = len (lons)
            while (lons [nlons - 1] >= loncdf.data [0]):
                lons = np.delete (lons, nlons - 1, 0)
                eles = np.delete (eles, nlons - 1, 1)
                nlons -= 1
            # The actual aggregation
            lons = np.concatenate ((lons, loncdf.data), axis=0)
            eles = np.concatenate ((eles, elecdf.data), axis=1)
        #print ('aggregate=({} {} {})' . format (lons.shape, lats.shape, eles.shape))
    #
    # Reduce from 5401x21604 which causes indexing errors
    lonIndexes = np.arange (0, len (lons), 4)
    latIndexes = np.arange (0, len (lats), 4)
    lons = np.take (lons, lonIndexes)
    lats = np.take (lats, latIndexes)
    eles = np.take (eles, latIndexes, axis=0)
    eles = np.take (eles, lonIndexes, axis=1)
    #
    # Create an interpolator
    #np.savetxt('lons.csv', lons)
    #np.savetxt('lats.csv', lats)
    return units, RegularGridInterpolator ((lats, lons), eles)

def loadCurrent():
    print ("loadCurrent")
    # https://data.nodc.noaa.gov/thredds/ncss/ncep/rtofs/2017/201703/ofs.20170321/surface/ofs_atl.t00z.n000.20170321.grb.grib2/dataset.html
    googleIdCurrent = '1ZL2ABGc5uqtBt9DK0_m7CxJPMBgpDrW3' # Extracted from share url
    
    tmpCurrent = tempfile.NamedTemporaryFile (suffix = '.nc',                                               prefix = 'tempCurrent',                                               delete = True) # Need file name but not the file or gdd fails
    tmpCurrent.close()
    # Download the file from url and save it locally
    gdd.download_file_from_google_drive (file_id = googleIdCurrent,                                          dest_path = tmpCurrent.name)
    with netcdf.netcdf_file (tmpCurrent.name, 'r', mmap = False) as f:
        loncdf = f.variables ['Longitude_of_Presure_Point_surface']
        latcdf = f.variables ['Latitude_of_Presure_Point_surface']        
        ucdf = f.variables ['Barotropic_U_velocity_entire_ocean_single_layer'] # 1x1684x1200
        vcdf = f.variables ['Barotropic_V_velocity_entire_ocean_single_layer'] # 1x1684x1200
        units = 'm.s-1'
    #
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
    return units,         LinearNDInterpolator (lonlat, u),         LinearNDInterpolator (lonlat, v)

def loadDeclination ():
    print ("loadDeclination")
    # https://maps.ngdc.noaa.gov/viewers/historical_declination/
    units = 'Degrees'        
    googleIdDeclination = '1KL-brszjyfiX7yAp_-ZEBbereOm-26lz' # Extracted from share url    
    tmpDeclination = tempfile.NamedTemporaryFile (suffix = '.nc',
                                                  prefix = 'tempDeclination',
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

def loadSharkPathGallagher():
    print ("loadSharkPathGallagher")    
    # Processing code courtesy of Smitesh
    # Upload the CSV Here
    # from google.colab import files
    # uploaded = files.upload()
    #
    # # Replace the filename here if you have saved the CSV as a different
    # df = pd.read_csv(io.BytesIO(uploaded[
    #     'Beneath The Waves - Blue Shark Atlantic - Data Jan 21, 2019.csv'])) 
    #
    googleFile = 'https://drive.google.com/uc?id=1XtdF630BEDDv-ixbZ6cE4RJlbVwukiUU&export=download'
    print ('Downloading {}... ' . format (googleFile), end='')    
    df = pd.read_csv(googleFile)
    print ('Done.')
    # Next step is to clean the Data and drop the columns we don't need
    COLUMN_MAPPING = {
        'Shark ID': COL_SHARK_ID,
        'Prg No.': 'prg_no',
        'Latitude': COL_LATITUDE,
        'Longitude': COL_LONGITUDE,
        'Loc. quality': 'loc_quality',
        'Loc. date': COL_DATE,
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
    cleaned_df[COL_DATE] = cleaned_df.loc_date.apply(lambda x: datetime.strptime(x, '%m/%d/%y %H:%M'))
    # Save to csv for more detailed inspection
    cleaned_df.to_csv ('outputs/cleaned_df_duplicates_included.csv')
    # Remove successive entries that are so close in time that the longitude
    # and latitude coordinates are unchanged. This is experimental
    cleaned_df = cleaned_df.drop_duplicates (subset = [COL_SHARK_ID, COL_LONGITUDE, COL_LATITUDE])
    # Save to csv for more detailed inspection
    cleaned_df.to_csv ('outputs/cleaned_df_duplicates_removed.csv')
    return cleaned_df

def loadSharkPathOcsearch():
    # Processing code courtesy of 
    # https://github.com/botwranglers/ocearch/blob/master/solutions/Query%20Ocearch%20API.ipynb
    url = 'http://www.ocearch.org/tracker/ajax/filter-sharks'
    headers = {'Accept' : 'application/json'}
    # Download
    resp = requests.get(url, headers=headers)
    df = pd.DataFrame (resp.json())
    # Extract just the pings so we eventually simplify the whole data structure
    pingFrames=[]
    for row in df.itertuples():
        pingFrame = pd.DataFrame(row.pings)
        pingFrame['id']=row.id
        pingFrames.append(pingFrame)
    len (pingFrames)
    pings = pd.concat(pingFrames)
    # Convert timestamp from string to datetime object
    pings ['datetime'] = pd.to_datetime (pings.tz_datetime)
    # tz_datetime duplicates datetime so remove it
    pings.drop(columns=['tz_datetime'], inplace=True)
    # Columns from download that we want to keep
    COLUMN_MAPPING = ['id',
                      'name',
                      'gender', 
                      'species', 
                      'weight',
                      'length',
                      COL_DATE,
                      'dist_total']
    # Merge the processed ping data. The ping data adds COL_LONGITUDE and COL_LATITUDE
    joined = pings.merge (df [COLUMN_MAPPING], on='id')
    return joined

def main():
    #unitsBathysphere, interpBathysphere = loadBathysphereAtlantic()
    unitsBathysphere, interpBathysphere = loadBathysphereWorld()
    #check (interpBathysphere, 'Bathysphere ({})' . format (unitsBathysphere))
    unitsCurrent, interpCurrentU, interpCurrentV = loadCurrent()
    #check (interpCurrentU, 'CurrentU ({})' . format (unitsCurrent))
    #check (interpCurrentV, 'CurrentV ({})' . format (unitsCurrent))
    unitsDeclination, interpDeclination = loadDeclination()
    #check (interpDeclination, 'Declination ({})' . format (unitsDeclination))
    df = loadSharkPathOcsearch()
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
    angleSep = math.sqrt ((lon1 - lon0) * (lon1 - lon0) +                           (lat1 - lat0) * (lat1 - lat0))
    return (angleSep * np.pi / 180.) * earthRadiusKilometers

main()

