#!/usr/bin/python3

from datetime import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
import io
import numpy as np
import os
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.io import netcdf
import shutil

def loadBathysphere ():
    print ("loadBathysphere")
    googleIdBathysphere = '10VqbV2oNUVcvS6lLP3FekVlFM4LUJj5o' # Extracted from share url
    tmpBathysphere = '/tmp/tempBathysphere.nc'
    # Download the file from url and save it locally
    if os.path.exists(tmpBathysphere):
        os.remove (tmpBathysphere) # Download will fail if file already exists
    gdd.download_file_from_google_drive (file_id = googleIdBathysphere,
                                         dest_path = tmpBathysphere)
    with netcdf.netcdf_file (tmpBathysphere, 'r', mmap = False) as f:
        loncdf = f.variables ['lon']
        latcdf = f.variables ['lat']
        elecdf = f.variables ['Band1']        
        crscdf = f.variables ['crs'] # Do not know what this array contains, other than 1 character strings

    # Transpose lat/lon to lon/lat
    ele = np.transpose (elecdf.data)
    
    # Create an interpolator. This is a regular grid so we use a regular grid interpolator that
    # exploits the regularity to achieve the most efficient search
    
    return RegularGridInterpolator ((loncdf.data, latcdf.data), ele)

def loadCurrent():
    print ("loadCurrent")    
    googleIdCurrent = '1PZrkKXh0LQ5-4tvW04FOw0z-vdNI39ru' # Extracted from share url
    tmpCurrent = '/tmp/tempCurrent.nc'
    # Download the file from url and save it locally
    if os.path.exists(tmpCurrent):
        os.remove (tmpCurrent) # Download will fail if file already exists
    gdd.download_file_from_google_drive (file_id = googleIdCurrent, \
                                         dest_path = tmpCurrent)
    with netcdf.netcdf_file (tmpCurrent, 'r', mmap = False) as f:
        loncdf = f.variables ['Longitude_of_U_Wind_Component_of_Velocity_surface']
        latcdf = f.variables ['Latitude_of_U_Wind_Component_of_Velocity_surface']
        ucdf = f.variables ['u-component_of_current_hybrid_layer'] # 1x1x1684x1200
        vcdf = f.variables ['v-component_of_current_hybrid_layer'] # 1x1x1684x1200

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
            u.append (ucdf[0][0][i][j])
            v.append (vcdf[0][0][i][j])
    return \
        LinearNDInterpolator (lonlat, u), \
        LinearNDInterpolator (lonlat, v)

def loadSharkPath():
    print ("loadSharkPath")        
    # Upload the CSV Here
    # from google.colab import files
    # uploaded = files.upload()

    # # Replace the filename here if you have saved the CSV as a different
    # df = pd.read_csv(io.BytesIO(uploaded[
    #     'Beneath The Waves - Blue Shark Atlantic - Data Jan 21, 2019.csv'])) 

    df = pd.read_csv(
        'https://drive.google.com/uc?id=1XtdF630BEDDv-ixbZ6cE4RJlbVwukiUU&export=download'
    )

    # Now that we have uploaded the Data we can see it as a Dataframe
    df.head()

    # Next step is to clean the Data and drop the columns we don't need
    COLUMN_MAPPING = {
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

    cleaned_df.head()
    return cleaned_df

loadBathysphere()
loadCurrent()
loadSharkPath()
