#!/usr/bin/env python
# coding: utf-8

# Libraries and options

# In[1]:


import numpy as np, pandas as pd
import sys, os, re, zipfile, shutil, pickle, matplotlib.pyplot as plt, seaborn as sns, bz2,time
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
pd.options.display.max_colwidth = 50


# In[2]:


def importPbz2( file ):
    data = bz2.BZ2File(file,'rb')
    return pd.read_pickle(data)


# In[3]:


newDF = importPbz2('covidPollutionCensus.pbz2')
# newDF = importPbz2('covidCensus.pbz2')


# In[4]:


newDF['fips'] = newDF['fips'].astype('str').str.zfill(5)
newDF['State Code'] = newDF['State Code'].astype('str').str.zfill(5)
newDF['County Code'] = newDF['County Code'].astype('str').str.zfill(5)
newDF['dates'] = pd.to_datetime(newDF['dates'])


# In[5]:


_=[print(x) for x in newDF]


# And explore...

# In[6]:


print(newDF.shape)
mask = newDF.isnull().any(axis=0)
print(len(mask))
noNullDF = newDF.loc[:,~mask]
print(noNullDF.shape)


# In[7]:


_=[print(x) for x in newDF]


# In[8]:


cols=['dates',
'AverageDailyTemperature.data',
'AveragePrecipitation.data',
'AverageWindSpeed.data',
'BLS_EmployedPopulation.data',
'BLS_LaborForcePopulation.data',
'BLS_UnemploymentRate.data',
'Female_Total_Population.data',
'JHU_ConfirmedCases.data',
'JHU_ConfirmedDeaths.data',
'LND110210',
'MaleAndFemale_AtLeast65_Population.data',
'TotalPopulation.data',
'hospitalIcuBeds',
'hospitalLicensedBeds',
'hospitalStaffedBeds',
'Arithmetic Mean',
'rollMeanMean',
'rollMeanUgm3',
'density',
'caseRate',
'deathRate']


# In[5]:


newDF[['totalMoved','movedWithinState','movedWithoutState','movedFromAbroad','publicTrans','totalTrans','householdsTotal','houseWith65',
      'house2+with65','houseFamily65','houseNonfam65','houseNo65','house2+No65','houseFamilyNo65','houseNonfamNo65',
      'householdStructuresTotal','householdIncomeMedian','gini','hoursWorkedMean','unitsInStructure','healthInsTotal',
      'healthInsNativeWith','healthInsForeignNatWith','healthInsForeignNoncitWith','healthInsForeignNatNo',
      'healthInsForeignNoncitNo','healthInsNativeNo']]  = \
newDF[['totalMoved','movedWithinState','movedWithoutState','movedFromAbroad','publicTrans','totalTrans','householdsTotal','houseWith65',
      'house2+with65','houseFamily65','houseNonfam65','houseNo65','house2+No65','houseFamilyNo65','houseNonfamNo65',
      'householdStructuresTotal','householdIncomeMedian','gini','hoursWorkedMean','unitsInStructure','healthInsTotal',
      'healthInsNativeWith','healthInsForeignNatWith','healthInsForeignNoncitWith','healthInsForeignNatNo',
      'healthInsForeignNoncitNo','healthInsNativeNo']].astype('float64')


# In[50]:


testDF = newDF.groupby([pd.Grouper(key='dates',freq='W'),'pollutant','fips']).agg('mean')


# In[6]:


testDF = newDF.groupby('dates').agg('mean')


# In[51]:


testDF = testDF.reset_index()#.set_index(['dates','fips','pollutant'])


# In[10]:


testDF['2020-03-11':'2021-03-11']


# In[7]:


testDF = testDF[testDF['dates'].between('2020-03-11','2021-03-11')]


# In[ ]:


testDF.sort_values(['dates','fips','pollutant'])


# In[13]:


def covidPollutantFipsCorr(df, pollutant, cols=[]):
    if cols:
        polDF =  df.loc[df['pollutant']==pollutant,cols+['fips']]
    else:
        polDF = df.loc[df['pollutant']==pollutant,:]
    fipsDF  = polDF.groupby(by='fips').agg('mean')
    return fipsDF.corr()


# NO2

# In[ ]:


# covidPollutantFipsCorr(newDF, 'no2',b+a)
# nIndex = newDF[newDF['pollutant'].isnull()].index
no2DF =  newDF.query("pollutant == 'no2' | pollutant.isnull()", engine='python')
no2DF = no2DF.sort_values(by=['fips','dates'])


# In[ ]:


no2DF=no2DF[no2DF['dates'].between('2020-03-11','2021-03-11')]
no2DF.sort_values(by=['fips','dates'],inplace=True)


# In[ ]:


covidPollutantFipsCorr(testDF,'no2',cols)


# Lead

# In[ ]:


covidPollutantFipsCorr(testDF, 'lead', cols)


# Ozone

# In[ ]:


covidPollutantFipsCorr(testDF, 'ozone', cols)


# PM2.5

# In[ ]:


covidPollutantFipsCorr(testDF, 'pm25', cols)


# In[ ]:


corrs =  testDF.corr()


# In[ ]:


corrs[['density','deathRate','JHU_ConfirmedDeaths.data','AverageDailyTemperature.data']]


# In[52]:


wokka=[i for i in testDF if not (i.endswith('.missing'))|(i.startswith('NYT'))]
testDF = testDF[wokka]


# In[53]:


def rateCalc( numerators, denominator, DF  ):
    for col in numerators:
        DF[col+'Rate'] = DF[col] / DF[denominator]


# In[64]:


# testDF['density']=testDF['TotalPopulation.data']/testDF['LND110210']
# testDF['caseRate']=testDF['JHU_ConfirmedCases.data'] / testDF['TotalPopulation.data']
# testDF['deathRate']=testDF['JHU_ConfirmedDeaths.data'] / testDF['TotalPopulation.data']


# In[65]:


ratesDF = testDF['2020-03-11':'2021-03-11'].copy()

rateCalc(['movedWithinState',
         'movedWithoutState',
         'movedFromAbroad'], 'totalMoved', ratesDF)

rateCalc(['publicTrans'], 'totalTrans', ratesDF)

rateCalc(['houseWith65', #householdsTotal
'house2+with65',
'houseFamily65',
'houseNonfam65',
'houseNo65',
'house2+No65',
'houseFamilyNo65',
'houseNonfamNo65'], 'householdsTotal', ratesDF)

rateCalc(['healthInsNativeWith', #healthInsTotal
'healthInsForeignNatWith',
'healthInsForeignNoncitWith'], 'healthInsTotal', ratesDF)

rateCalc(['hospitalIcuBeds',
          'hospitalLicensedBeds',
          'hospitalStaffedBeds'], 'TotalPopulation.data', ratesDF)


# In[66]:


ratesDF.drop(columns=['latestTotalPopulation',
              'totalMoved',
              'movedWithinState', 
              'movedWithoutState',
              'movedFromAbroad',
              'publicTrans',
              'totalTrans',
              'householdsTotal', 
              'houseWith65', 
              'house2+with65',
              'houseFamily65',
              'houseNonfam65',
              'houseNo65',
              'house2+No65',
              'houseFamilyNo65',
              'houseNonfamNo65',
              'healthInsTotal',
              'healthInsNativeWith', 
              'healthInsForeignNatWith',
              'healthInsForeignNoncitWith',
              'hospitalIcuBeds',
              'hospitalLicensedBeds',
              'hospitalStaffedBeds',
              'TotalPopulation.data'], inplace=True)


# In[67]:


rateCorrs = ratesDF[ratesDF['dates']>'2020-03-11'].corr()


# In[69]:


rateCorrs[['caseRate','deathRate']]


# In[ ]:


ratesDF['dates'].max()


# In[ ]:


no2Corr = covidPollutantFipsCorr(ratesDF,'no2')
no2Corr[['movedWithinStateRate','movedWithoutStateRate','movedFromAbroadRate','caseRate','deathRate']]


# In[ ]:


ratesDF['publicTransRate'].notna().sum()

