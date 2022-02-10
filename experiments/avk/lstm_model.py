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


# These are the ones we did NOT normalize/standardize
cols_raw = ['fips','JHU_ConfirmedCases.data', 'JHU_ConfirmedDeaths.data', 'cyclical_sin', 'cyclical_cos', 'continuous_sin',
       'continuous_cos']

# this should be the current script's folder

cwd = pathlib.Path.cwd()


    
    
        
        
        
        
    



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


