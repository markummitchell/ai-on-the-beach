#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io 
import pandas as pd
from datetime import datetime


# In[2]:


# Upload the CSV Here
# from google.colab import files
# uploaded = files.upload()

# # Replace the filename here if you have saved the CSV as a different
# df = pd.read_csv(io.BytesIO(uploaded[
#     'Beneath The Waves - Blue Shark Atlantic - Data Jan 21, 2019.csv'])) 

df = pd.read_csv(
    'https://drive.google.com/uc?id=1XtdF630BEDDv-ixbZ6cE4RJlbVwukiUU&export=download'
)


# In[ ]:


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


# In[ ]:




