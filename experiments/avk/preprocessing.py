import os
import os.path
import pickle
import bz2
from glob import glob
import random
import shutil
import pathlib
from socket import INADDR_LOOPBACK
import tensorflow as tf


import pandas as pd
import numpy as np



# These are the ones we did NOT normalize/standardize
cols_raw = ['fips','JHU_ConfirmedCases.data', 'JHU_ConfirmedDeaths.data', 'cyclical_sin', 'cyclical_cos', 'continuous_sin',
       'continuous_cos']

# this should be the current script's folder

cwd = pathlib.Path.cwd()

def get_latlong_fc(cdf):
    """processes the county centroid lat/long into a TF feature column
    Inputs:
    
        pd.Dataframe                      :   Primary dataframe from Noah's processing steps.
                                                requires columns INTPTLAT and INTPTLONG
    
    Returns:
        tf.feature_column.crossed_column  :   This is the result of crossing lat and long
        
    
        process taken from https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
    """

    cdf.rename(columns={'INTPTLAT':'latitude', 'INTPTLONG':'longitude'},inplace=True)
    
    #dropping Alaska and Hawaii
    # cdf.drop(cdf[(cdf['GEOID'] > 2000) & (cdf['GEOID'] < 3000)].index, inplace=True)
    # cdf.drop(cdf[(cdf['GEOID'] > 15000) & (cdf['GEOID'] < 16000)].index, inplace=True)
    
    # raw_lat = cdf['latitude'].copy()
    # raw_long = cdf['longitude'].copy()
    
    lat_buckets = list(np.linspace(cdf.latitude.min(),cdf.latitude.max(),100))
    long_buckets = list(np.linspace(cdf.longitude.min(),cdf.longitude.max(),100))
    #make feature columns
    lat_fc = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'),lat_buckets)
    long_fc= tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longitude'),long_buckets)
    
    # crossed columns tell the model how the features relate
    cross_coordinate_fc = tf.feature_column.crossed_column(keys=[lat_fc, long_fc], hash_bucket_size=1000) # No precise rule, maybe 1000 buckets will be good?
    
    
    return cross_coordinate_fc






class DatasetManager():
    def __init__(self, filepath) -> None:
        self.datapath = filepath
        self.df = None
        self._load_bz2(self.datapath)
        self.feature_columns = None
        self.standard_drops = ['NYT_ConfirmedCases.data','NYT_ConfirmedDeaths.data','NYT_ConfirmedDeaths.missing',
                       'county','LND110210','countyStateName','stateFip','countyFip']
    def _load_bz2(self)-> pd.DataFrame:
        """Returns dataframe contained within the compressed pickle file
        
        """
        with bz2.BZ2File(self.datapath, 'rb')as data:
            self.df = pd.read_pickle(data)
        
    
    def prep_df(self, drop_cols=None, countyFile='2021_Gaz_counties_national.txt'):
        """ Preps the dataframe according to the group's convention
        
        """
        if drop_cols == None:
            drop_cols=self.standard_drops
        else:
            # ensure the standard columns to drop are still there
            included_drops = self.standard_drops # [x for x in self.standard_drops (if x in drop_cols)]
        # remove 
        # self.df.columns = self.df.columns.str.replace(" ","")
        self.df.drop(drop_cols,axis=1, inplace=True)
        self.county_merger(countyFile)
        
    def county_merger(self, file_Path):
        counties = pd.read_csv(file_Path, delimiter='\t')
        counties.columns = counties.columns.str.replace(" ", "")
        counties = counties[['GEOID', 'INTPTLAT', 'INTPTLONG' ]]
        self.df.fips = self.df.fips.astype('int64')
        self.df = self.df.merge(counties, how='left', left_on='fips', right_on='GEOID')
        self.df.drop(['GEOID'],axis=1, inplace=True)
        
        self.feature_column = get_latlong_fc(self.df)
        return 
    
    

        
        
if __name__ == '__main__':
    
    # avk_ex_folder = pathlib.Path(__file__).parent.resolve()
    # data_path = pathlib.Path.joinpath(avk_ex_folder,'..','..','data','golden','feeFiFoFum.pbz2').resolve()
    
    # data_manager = DatasetManager(data_path)
    # data_manager.prep_df()
    # data_manager.county_merger(avk_ex_folder.joinpath('2021_Gaz_counties_national.txt'))
    # top_dir = pathlib.Path(__file__).parent.parent.resolve()
    top_dir = pathlib.Path(__file__).joinpath("..","..","..").resolve()
    print(top_dir)