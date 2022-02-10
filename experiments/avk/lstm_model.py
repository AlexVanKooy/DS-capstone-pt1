import os
import os.path
import pickle
import bz2
from glob import glob
import random
import shutil
import pathlib
from socket import INADDR_LOOPBACK


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split# Windows version


# this should be the current script's folder

cwd = pathlib.Path.cwd()

class DatasetManager():
    def __init__(self, filepath) -> None:
        self.datapath = filepath
        self.df = None
        self._load_bz2(self.datapath)
    def _load_bz2(self)-> pd.DataFrame:
        """Returns dataframe contained within the compressed pickle file
        
        """
        with bz2.BZ2File(self.datapath, 'rb')as data:
            self.df = pd.read_pickle(data)
        
    
    def prep_df(self, drop_cols=None):
        """ Preps the dataframe according to the group's convention
        
        """
        if drop_cols == None:
            drop_cols=['NYT_ConfirmedCases.data','NYT_ConfirmedDeaths.data','NYT_ConfirmedDeaths.missing',
                       'county','LND110210','countyStateName','stateFip','countyFip']
        # remove 
        # self.df.columns = self.df.columns.str.replace(" ","")
        self.df.drop(drop_cols,axis=1, inplace=True)
   
    def county_merger(self, file_Path):
        counties = pd.read_csv(file_Path, delimiter='\t')
        counties.columns = counties.columns.str.replace(" ", "")
        counties = counties[['GEOID', 'INTPTLAT', 'INTPTLONG' ]]
        self.df.fips = self.df.fips.astype('int64')
        self.df = self.df.merge(counties, how='left', left_on='fips', right_on='GEOID')
        self.df.drop(['GEOID'],axis=1, inplace=True)
    
    
        
        
        
        
    



# golden_dataset_file_name = os.path.join('..', '..', 'data', 'golden', 'feeFiFoFum.pbz2')

# # data = bz2.BZ2File(golden_dataset_file_name,'rb')
# with bz2.BZ2File(golden_dataset_file_name,'rb') as data:
#     df = pd.read_pickle(data)
# df

if __name__ == '__main__':
    avk_ex_folder = pathlib.Path(__file__).parent.resolve()
    data_path = pathlib.Path.joinpath(avk_ex_folder,'..','..','data','golden','feeFiFoFum.pbz2').resolve()
    
    data_manager = DatasetManager(data_path)
    data_manager.prep_df()
    data_manager.county_merger(avk_ex_folder.joinpath('2021_Gaz_counties_national.txt'))


