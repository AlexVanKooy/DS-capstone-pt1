# general imports
import numpy as np
import pandas as pd
from urllib.request import urlopen
import json
import os
import glob
import cv2
import pathlib
import plotly.express as px
"""
This script is able to load a pollution dataset and color the appropriate counties according to 
    the rules in the `choropleth` function call(~line 40)
    It will save an image for each day into the folder sepcified, currently `no2_images` 
    
    Sorry this is a bit clunky to deal with but I figured parameterizing it was unnecessary at this point.
"""





def get_us_counties():

    ################################################################################################################
    # invert the commented section if you do not have the counties json object saved in                            #
    #  this working directory                                                                                      #
    ################################################################################################################


    # with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    #     counties = json.load(response)
    
    with open('counties.json','r') as jin:
        counties = json.load(jin)
    return counties

def make_images(df):
    dates_in_df = df['Date Local'].unique()
    max_ppb = df['Arithmetic Mean'].max()
    counties = get_us_counties()

    for idx, date in enumerate(dates_in_df):
        fig = px.choropleth(df[df['Date Local']==date], geojson=counties, locations='fips', color='Arithmetic Mean',
                           color_continuous_scale="Viridis",
                           range_color=(0, 60), #max value for daily avg is ~60ppb
                           scope="usa",
                           labels={'Arithmetic Mean':'Arithmetic Mean (ppb)'}
                          )
        # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.update_layout(
            autosize=False,
            width=1300,
            height=1100,)
        fig.write_image("no2_images/image{}.jpg".format(idx))

        





# load dataframe of interest
def main():

    df = pd.read_csv("./data/air_quality/no2/daily_no2_2020_squashed.csv",dtype={'fips':'string'})
    make_images(df)


if __name__ == "__main__":
    main()
    print('complete')