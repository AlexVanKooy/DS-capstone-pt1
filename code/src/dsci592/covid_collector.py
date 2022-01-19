import requests
import json
from time import sleep
import os
from os import path
from datetime import datetime, timedelta
from glob import glob
import re
import csv
import math
from typing import List
from collections import Counter, defaultdict
import openpyxl
import pkg_resources

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gamma
import numpy as np


def read_data_json(typename, api, body):
    """
    From C3.ai
    read_data_json directly accesses the C3.ai COVID-19 Data Lake APIs using the requests library,
    and returns the response as a JSON, raising an error if the call fails for any reason.
    ------
    typename: The type you want to access, i.e. 'OutbreakLocation', 'LineListRecord', 'BiblioEntry', etc.
    api: The API you want to access, either 'fetch' or 'evalmetrics'.
    body: The spec you want to pass. For examples, see the API documentation.
    """
    response = requests.post(
        "https://api.c3.ai/covid/api/1/" + typename + "/" + api,
        json=body,
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    )
    if not response.ok:
        print(response.text)

    response.raise_for_status()

    return response.json()


def fetch(typename, body, get_all=False, remove_meta=True):
    """
    From C3.ai
    fetch accesses the Data Lake using read_data_json, and converts the response into a Pandas dataframe.
    fetch is used for all non-timeseries data in the Data Lake, and will call read_data as many times
    as required to access all of the relevant data for a given typename and body.
    ------
    typename: The type you want to access, i.e. 'OutbreakLocation', 'LineListRecord', 'BiblioEntry', etc.
    body: The spec you want to pass. For examples, see the API documentation.
    get_all: If True, get all records and ignore any limit argument passed in the body. If False, use the limit argument passed in the body. The default is False.
    remove_meta: If True, remove metadata about each record. If False, include it. The default is True.
    """
    if get_all:
        has_more = True
        offset = 0
        limit = 2000
        df = pd.DataFrame()

        while has_more:
            body['spec'].update(limit=limit, offset=offset)
            response_json = read_data_json(typename, 'fetch', body)
            new_df = pd.json_normalize(response_json['objs'])
            df = df.append(new_df)
            has_more = response_json['hasMore']
            offset += limit

    else:
        response_json = read_data_json(typename, 'fetch', body)
        df = pd.json_normalize(response_json['objs'])

    if remove_meta:
        df = df.drop(columns=[c for c in df.columns if ('meta' in c) | ('version' in c)])

    return df


def evalmetrics(typename, body, remove_meta=True):
    """
    From C3.ai
    evalmetrics accesses the Data Lake using read_data_json, and converts the response into a Pandas dataframe.
    evalmetrics is used for all timeseries data in the Data Lake.
    ------
    typename: The type you want to access, i.e. 'OutbreakLocation', 'LineListRecord', 'BiblioEntry', etc.
    body: The spec you want to pass. For examples, see the API documentation.
    remove_meta: If True, remove metadata about each record. If False, include it. The default is True.
    """
    response_json = read_data_json(typename, 'evalmetrics', body)
    df = pd.json_normalize(response_json['result'])

    # get the useful data out
    df = df.apply(pd.Series.explode)
    if remove_meta:
        df = df.filter(regex='dates|data|missing')

    # only keep one date column
    date_cols = [col for col in df.columns if 'dates' in col]
    keep_cols = date_cols[:1] + [col for col in df.columns if 'dates' not in col]
    df = df.filter(items=keep_cols).rename(columns={date_cols[0]: "dates"})
    df["dates"] = pd.to_datetime(df["dates"])

    return df


def fetch_one(typename: str, body: dict, objs_only=True) -> dict:
    """
    Returns JSON output from single API call

    Args:
        typename: the C3.ai type name
        body: the body of the request
        objs_only: if True, remove the metadata and just returns the objects

    Returns:
        JSON response as dictionary

    """

    response = read_data_json(typename, 'fetch', body)
    if objs_only:
        for r in response['objs']:
            if 'meta' in r.keys():
                del (r['meta'])

        return response['objs']

    return response


def pwd():
    print(os.getcwd())
    my_data = pkg_resources.resource_filename(__name__, "config/counties.json")
    print(my_data)
    # my_data = pkg_resources.resource_string(__name__, "config/counties.json")
    # print(my_data)


def get_us_locations(file_name=pkg_resources.resource_filename(__name__, "config/C3-ai-Location-IDs.xlsx")):
    """ Loads all US counties from C3 ai spreadsheet

    Args:
        file_name: the name of the spreadsheet

    Returns:
        Pandas dataframe with the results

    """

    locations = pd.read_excel(file_name, sheet_name='County IDs', header=2, engine='openpyxl')
    us_locations = locations[locations.Country == 'United States']

    return us_locations


def make_outbreaklocation_body(county_id: str) -> dict:
    """ Forms the request body for a count for the outbreak location API

    Args:
        county_id: the ID for the County

    Returns:
        The request body

    """
    return {
        "spec": {
            "filter": f"id == '{county_id}'"
        }
    }


def retrieve_and_save_population_data(file_name, max_tries=10):
    """ Loads all population data for US counties and stores in the filename provided"""

    us_locations = get_us_locations()
    keep_going = True
    tries = 0
    while keep_going:
        try:
            with open(file_name) as file:
                county_data = json.load(file)
        except:
            county_data = {}
        i = 0
        for county in us_locations['County id']:
            if county not in county_data.keys():
                try:
                    data = fetch_one('outbreaklocation', make_outbreaklocation_body(county))
                    county_data[county] = data[0]
                    i += 1
                    if i % 100 == 0:
                        print(f'Saving: {i}')
                        with open(file_name, 'w') as file:
                            json.dump(county_data, file)
                except Exception as e:
                    county_data[county] = None
                    print(f'Problem with {county}: {e}')
                # sleep(1)
        with open(file_name, 'w') as file:
            json.dump(county_data, file)
        if len(county_data) >= len(us_locations) or tries >= max_tries:
            keep_going = False
        else:
            tries += 1


def get_counties_df(counties_json_file_name):
    with open(counties_json_file_name) as file:
        county_data = json.load(file)

    df = pd.DataFrame.from_dict(county_data)

    data = [df[col] for col in df.columns]

    # pivot
    return pd.DataFrame(data, columns=df.index, index=df.columns)


def get_county_stats_df(file_name=pkg_resources.resource_filename(__name__, "config/county_stats.csv")):
    """ Get county land area (LND110210) by FIPS code
    Source file at https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html

    """
    return pd.read_csv(file_name)[['fips', 'LND110210']]


def get_last_file_date(raw_file_path: str, county: str) -> str:
    """ Retrieves the last date through which data was loaded for the specified county

    Args:
            file_path: Path to the raw files
            county: The county requested

    Returns:
            A string representing the last date processed.  If the county has never been processed, a
            default date of 2020-01-01 is returned

    """

    max_date = '2020-01-01'
    files = glob(path.join(raw_file_path, f'{county}*.psv'))
    if not files:
        return max_date
    for file in files:
        match = re.search(r'(\d\d\d\d-\d\d-\d\d).psv', file)
        if match:
            max_date = max(max_date, match.group(1))
    return max_date


def download_evalmetrics_data(raw_file_path, counties_json_file_name):
    """ Downloads the evalmetrics data from the last download
    through current """

    today = datetime.now().strftime('%Y-%m-%d')

    # Get the list of counties
    with open(counties_json_file_name) as file:
        counties = json.load(file)

    # Iterate through counties saving the time series data
    for counter, (county, details) in enumerate(counties.items()):
        print('.', end='')
        if counter and not counter % 120:
            print()

        # Skip if missing county details
        if not details:
            continue

        # Get the last date we processed
        last_date = get_last_file_date(raw_file_path, county)

        if last_date == today:
            continue

        expressions = ["JHU_ConfirmedCases",
                       "JHU_ConfirmedDeaths",
                       "JHU_ConfirmedRecoveries",
                       "NYT_ConfirmedCases",
                       "NYT_ConfirmedDeaths",
                       "NYT_AllCausesDeathsWeekly_Deaths_AllCauses",
                       "NYT_AllCausesDeathsWeekly_Excess_Deaths",
                       "NYT_AllCausesDeathsWeekly_Expected_Deaths_AllCauses",
                       "NYT_AllCausesDeathsMonthly_Deaths_AllCauses",
                       "NYT_AllCausesDeathsMonthly_Excess_Deaths",
                       "NYT_AllCausesDeathsMonthly_Expected_Deaths_AllCauses",
                       "TotalPopulation",
                       "Male_Total_Population",
                       "Female_Total_Population",
                       "MaleAndFemale_Under18_Population",
                       "MaleAndFemale_AtLeast65_Population",
                       "BLS_LaborForcePopulation",
                       "BLS_EmployedPopulation",
                       "BLS_UnemployedPopulation",
                       "BLS_UnemploymentRate",
                       "AverageDailyTemperature",
                       "AverageDewPoint",
                       "AverageRelativeHumidity",
                       "AverageSurfaceAirPressure",
                       "AveragePrecipitation",
                       "AverageWindSpeed",
                       "AverageWindDirection",
                       "AveragePrecipitationTotal", ]

        # Get the data for the county from the last date processed
        for i in range(math.ceil(len(expressions) // 4)):
            body = {"spec": {
                "ids": [county],
                "expressions": expressions[i * 4:(i + 1) * 4],
                "start": last_date,
                "end": today,
                "interval": "DAY",
            }
            }

            try:
                df = evalmetrics("outbreaklocation", body)
                file_name = path.join(raw_file_path, f'{county}-part-{i}-{last_date}-{today}.psv')
                df.to_csv(file_name, sep='|')
            except Exception as e:
                print(f'Error processing {county}: {e}')

            sleep(1)


def save_raw_df(df: pd.DataFrame, pickled_data_path):
    df.to_pickle(path.join(pickled_data_path, 'raw_evalmetrics_df.pkl'))


def load_and_normalize_df(file_name):
    df = pd.read_csv(file_name, delimiter='|', index_col=1)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    columns = {c: '.'.join(c.split('.')[-2:]) for c in df.columns}
    df.rename(columns=columns, inplace=True)
    return df


def merge_county_parts_to_dataframe(file_names):
    df = load_and_normalize_df(filenames[0])
    for file_name in file_names[1:]:
        df2 = load_and_normalize_df(file_name)
        df = df.join(df2)
    return df


def save_county_merged_parts_df(county, df, county_merged_parts_path):
    df.to_pickle(path.join(county_merged_parts_path, f'{county}.pkl'))


def get_counties_from_files(raw_file_path):
    return list({f.split('-part')[0] for f in os.listdir(raw_file_path)})


def get_dates_for_county(county, raw_file_path):
    # files = glob(f'./data/{county}*')
    files = glob(path.join(raw_file_path, f'{county}*'))
    dates = {re.findall('\d\d\d\d-\d\d-\d\d-\d\d\d\d-\d\d-\d\d', f)[0] for f in files}
    return list(dates)


def get_county_files_for_date(county, dt, raw_file_path):
    # files = glob(f'./data/raw_data/{county}-part-*-{dt}.psv')
    files = glob(path.join(raw_file_path, f'{county}-part-*-{dt}.psv'))
    return [f[f.index(county):] for f in files]


def process_county(county, county_population_stats, raw_file_path, county_merged_parts_path):
    df = None
    dates = get_dates_for_county(county, raw_file_path)
    for dt in dates:
        files = get_county_files_for_date(county, dt, raw_file_path)
        if df is not None:
            df.append(merge_county_parts_to_dataframe(files))
        else:
            df = merge_county_parts_to_dataframe(files)

    for k, v in county_population_stats.iteritems():
        df[k] = v
    save_county_merged_parts_df(county, df, county_merged_parts_path)
    return df


def get_county_population_stats(counties_json_file_name):
    county_population = get_counties_df(counties_json_file_name)
    county_stats = get_county_stats_df()
    fips = []
    for county, population in county_population.iterrows():
        if isinstance(population.fips, dict):
            fips.append(int(population.fips['id']))
        else:
            fips.append(population.fips)
    county_population.fips = fips
    county_population = county_population.merge(county_stats, how='left', left_on='fips', right_on='fips')
    county_population.set_index('id', inplace=True)
    county_population = county_population.drop(columns=['location', 'name', 'version', 'typeIdent'])
    return county_population


def process_counties(raw_file_path, counties_json_file_name):
    counties = get_counties_from_files(raw_file_path)
    county_population_stats = get_county_population_stats(counties_json_file_name)

    for county in counties:
        # print(county)
        process_county(county, county_population_stats.loc[county])


def make_outbreaklocation_body_with_includes(county_id: str, includes: str = None) -> dict:
    """ Forms the request body for a count for the outbreak location API

    Args:
        county_id: the ID for the County

    Returns:
        The request body
        :param county_id:
        :param includes:

    """
    spec = {"spec": {"filter": f"id == '{county_id}'"}}

    if includes:
        spec["spec"]["include"] = includes

    return spec


def fetch_one_population(county_id: str) -> List[dict]:
    population_data = fetch_one('outbreaklocation',
                                make_outbreaklocation_body_with_includes(county_id, 'populationData'))
    population_data = population_data[0]['populationData']
    last_year = max(e['year'] for e in population_data)
    population_data = [e for e in population_data if e['year'] == last_year]
    return population_data


def unique_gender(population_data: list):
    return list(set(e['gender'] for e in population_data))


def unique_race(population_data: list):
    return list(set(e.get('race') for e in population_data))


def unique_ethnicity(population_data: list):
    return list(set(e.get('ethnicity') for e in population_data))


def unique_population_age(population_data: list):
    return list(set(e.get('populationAge') for e in population_data))


def to_dict(data, col_name):
    result = Counter()
    for e in data:
        try:
            result[e[col_name]] += e['value']
        except:
            pass
    return result


def to_all_counties_dict(col_names, raw_file_path):
    counties = get_counties_from_files(raw_file_path)
    results = defaultdict(defaultdict)
    for county in counties:
        try:
            pop = fetch_one_population(county)
            for col_name in col_names:
                pop_dict = to_dict(pop, col_name)
                results[county][col_name] = pop_dict
        except KeyError:
            pass
    return results


def to_df(all_counties_dict, col_name):
    results = dict(dict())
    for k, v in all_counties_dict.items():
        results[k] = v[col_name]
    return pd.DataFrame(results).T


def to_dfs(all_counties_dict, col_names):
    results = {}
    for col_name in col_names:
        results[col_name] = to_df(all_counties_dict, col_name)
    return results


def save_dfs(dfs, processed_data_demographics_path):
    for k, v in dfs.items():
        v.to_pickle(path.join(processed_data_demographics_path, f'{k}.pkl'))


def run():
    pass
