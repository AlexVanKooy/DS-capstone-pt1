import os
import os.path
import pickle
import bz2
from glob import glob
import random
import shutil
from datetime import datetime
import pathlib
import configparser

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
from kerashypetune import KerasGridSearch

RANDOM_SEED = 42
this_file_directory = pathlib.Path(__file__).parent.resolve()

def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_pickle(filepath)

    # drop numeric and unnecessary columns
    cols = ['NYT_ConfirmedCases.data', 'NYT_ConfirmedDeaths.data', 'NYT_ConfirmedDeaths.missing',
            'county', 'LND110210', 'countyStateName', 'stateFip', 'countyFip']
    df.drop(cols, axis=1, inplace=True)

    # temporarily replace fips code with latitude and longitude
    counties = pd.read_csv('2021_Gaz_counties_national.txt', delimiter='\t')
    counties.rename(columns={
        'INTPTLONG                                                                                                               ': 'longitude',
        'INTPTLAT': 'latitude'}, inplace=True)

    counties = counties[['GEOID', 'latitude', 'longitude']]
    df.fips = df.fips.astype('int64')

    df = df.merge(counties, how='left', left_on='fips', right_on='GEOID')
    df.drop(['GEOID'], axis=1, inplace=True)

    # Replace dates with monotonically increasing integers starting with the minimum date

    df.dates = pd.to_datetime(df.dates, format='%Y-%m-%d')
    min_date = min(df.dates)
    max_date = max(df.dates)

    df['day'] = (df.dates - min_date).dt.days
    df.drop(['dates'], axis=1, inplace=True)

    # Replace the integer representation of date with sin and cosine encoding
    cyclical_interval = 365
    continuous_interval = 3650
    df['cyclical_sin'] = np.sin((df.day * 2 * np.pi) / cyclical_interval)
    df['cyclical_cos'] = np.cos((df.day * 2 * np.pi) / cyclical_interval)
    df['continuous_sin'] = np.sin((df.day * 2 * np.pi) / continuous_interval)
    df['continuous_cos'] = np.cos((df.day * 2 * np.pi) / continuous_interval)
    df.drop('day', axis=1, inplace=True)

    # Get the feature column for latitude and longitude
    lat_buckets = list(np.linspace(df.latitude.min(), df.latitude.max(), 100))
    long_buckets = list(np.linspace(df.longitude.min(), df.longitude.max(), 100))

    # make feature columns
    lat_fc = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'), lat_buckets)
    long_fc = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longitude'), long_buckets)

    # crossed columns tell the model how the features relate
    # No precise rule, maybe 1000 buckets will be good?
    crossed_latlong = tf.feature_column.crossed_column(keys=[lat_fc, long_fc],
                                                       hash_bucket_size=1000)

    embedded_latlong = tf.feature_column.embedding_column(crossed_latlong, 9)

    feature_layer = tf.keras.layers.DenseFeatures(embedded_latlong)

    df[['geo0', 'geo1', 'geo2', 'geo3', 'geo4', 'geo5', 'geo6', 'geo7', 'geo8']] = feature_layer(
        {'latitude': df.latitude, 'longitude': df.longitude})

    # Normalize the data

    cols_to_normalize = [
        'TotalPopulation.data', 'MaleAndFemale_AtLeast65_Population.data',
        'Male_Total_Population.data', 'Female_Total_Population.data',
        'MaleAndFemale_Under18_Population.data', 'BLS_EmployedPopulation.data',
        'BLS_EmployedPopulation.missing', 'BLS_UnemployedPopulation.data',
        'BLS_UnemployedPopulation.missing', 'BLS_UnemploymentRate.data',
        'BLS_UnemploymentRate.missing', 'BLS_LaborForcePopulation.data',
        'BLS_LaborForcePopulation.missing', 'AverageDailyTemperature.data',
        'AverageDailyTemperature.missing', 'AverageDewPoint.data',
        'AverageDewPoint.missing', 'AverageRelativeHumidity.data',
        'AverageRelativeHumidity.missing', 'AverageSurfaceAirPressure.data',
        'AverageSurfaceAirPressure.missing', 'AveragePrecipitationTotal.data',
        'AveragePrecipitationTotal.missing', 'AveragePrecipitation.data',
        'AveragePrecipitation.missing', 'AverageWindDirection.data',
        'AverageWindDirection.missing', 'AverageWindSpeed.data',
        'AverageWindSpeed.missing', 'hospitalIcuBeds', 'hospitalStaffedBeds',
        'hospitalLicensedBeds', 'latestTotalPopulation', 'jhu_daily_death',
        'jhu_daily_cases', 'jhu_daily_new_cases',
        'jhu_daily_death_rolling_7',
        'jhu_daily_cases_rolling_7', 'jhu_daily_new_cases_rolling_7',
        'jhu_daily_death_rolling_30', 'jhu_daily_cases_rolling_30',
        'jhu_daily_new_cases_rolling_30', 'jhu_death_rate', 'jhu_case_rate',
        'jhu_new_case_rate', 'density', 'icu_beds_per_person',
        'staffed_beds_per_person', 'licensed_beds_per_person', 'cold_days',
        'hot_days', 'moderate_days', 'gte_65_percent', 'lt_18_percent',
        'employed_percent', 'unemployed_percent', 'totalMoved',
        'movedWithinState', 'movedWithoutState', 'movedFromAbroad',
        'publicTrans', 'totalTrans', 'householdsTotal', 'houseWith65',
        'house2+with65', 'houseFamily65', 'houseNonfam65', 'houseNo65',
        'house2+No65', 'houseFamilyNo65', 'houseNonfamNo65',
        'householdStructuresTotal', 'householdIncomeMedian', 'gini',
        'hoursWorkedMean', 'unitsInStructure', 'healthInsTotal',
        'healthInsNativeWith', 'healthInsForeignNatWith',
        'healthInsForeignNoncitWith', 'healthInsForeignNatNo',
        'healthInsForeignNoncitNo', 'healthInsNativeNo', 'pm25', 'latitude',
        'longitude']
    cols_raw = ['fips', 'JHU_ConfirmedCases.data', 'JHU_ConfirmedDeaths.data', 'cyclical_sin', 'cyclical_cos',
                'continuous_sin',
                'continuous_cos', 'geo0', 'geo1', 'geo2', 'geo3', 'geo4', 'geo5', 'geo6', 'geo7', 'geo8']
    df_normalized = df[cols_to_normalize]
    df_normalized = (df_normalized - df_normalized.mean()) / df_normalized.std()
    df_raw = df[cols_raw]
    df = pd.concat([df_raw, df_normalized], axis=1)

    return df


def xy_generator(data, fips, days=31):
    for j, fip in enumerate(fips):
        if not j % 100: print(j, end=' ')
        county = data[data.fips == fip]
        for i in range(days, len(county) + 1):
            data_matrix = data.iloc[i - days: i, 1:].to_numpy()
            yield data_matrix


def prepare_data(df, days_of_history=30, days_to_predict=1, n_samples=200, output_path='./data'):
    fips = df.fips.unique()

    Xi = []
    j = 0

    for i, x in enumerate(xy_generator(df, fips)):
        Xi.append(x)
        if i and not i % (n_samples - 1):
            X = np.asarray(Xi)
            np.save(os.path.join(output_path, f'x_{j}.npy'), X)
            j += 1
            Xi = []
    if Xi:
        X = np.asarray(Xi)
        np.save(os.path.join(output_path, f'x_{j}.npy'), X)


def set_seed(random_seed=RANDOM_SEED):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)



def train_test_eval_split(source='./data', train='./data/train', test='./data/test', eval_='./data/eval'):
    x_files = glob(os.path.join(source, 'x_*.npy'))
    random.shuffle(x_files)
    n_files = len(x_files)
    n_train = int(n_files * 0.70)
    n_eval = int(n_files * 0.15)
    n_test = n_files - n_train - n_eval
    train_files = x_files[:n_train]
    eval_files = x_files[n_train:n_train + n_test]
    test_files = x_files[n_train + n_test:]
    assert n_files == len(train_files) + len(eval_files) + len(test_files)
    for (subdir, lst) in [[train, train_files], [eval_, eval_files], [test, test_files]]:
        for file in lst:
            shutil.move(file, subdir)


def get_train_test_eval_ds(train='./data/train/x_*.npy', eval_='./data/eval/x_*.npy',
                           test='./data/test/x_*.npy', n_readers=5, n_parse_threads=5):
    train_files = glob(train)
    eval_files = glob(eval_)
    test_files = glob(test)

    def create_generator(files, cycle_length=5):
        set_seed()
        random.shuffle(files)
        for i in range(0, len(files), cycle_length):
            subset = files[i:i + cycle_length]
            np_arrays = [np.load(s) for s in subset]
            np_array = np.concatenate(np_arrays, axis=0)
            np.random.shuffle(np_array)
            yield np_array

    def split_xy(np_array):
        X = np_array[:, :-1, :]
        y = np_array[:, -1:, :1]
        return X, y

    train_ds = tf.data.Dataset.from_generator(lambda: create_generator(train_files, cycle_length=n_readers),
                                              output_types=tf.float32)
    train_ds = train_ds.map(split_xy, num_parallel_calls=n_parse_threads).prefetch(1)

    val_ds = tf.data.Dataset.from_generator(lambda: create_generator(eval_files, cycle_length=n_readers),
                                            output_types=tf.float32)
    val_ds = val_ds.map(split_xy, num_parallel_calls=n_parse_threads).prefetch(1)

    test_ds = tf.data.Dataset.from_generator(lambda: create_generator(test_files, cycle_length=n_readers),
                                             output_types=tf.float32)
    test_ds = test_ds.map(split_xy, num_parallel_calls=n_parse_threads).prefetch(1)

    return train_ds, val_ds, test_ds
