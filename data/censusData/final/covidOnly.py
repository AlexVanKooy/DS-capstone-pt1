#!/usr/bin/env python
# coding: utf-8

# Libraries and options

# In[36]:


import numpy as np, pandas as pd
import sys, os, re, zipfile, shutil, pickle, matplotlib.pyplot as plt, seaborn as sns, bz2,time
import scipy.stats as stats
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
pd.options.display.max_colwidth = 50


# In[2]:


def importPbz2( file ):
    data = bz2.BZ2File(file,'rb')
    return pd.read_pickle(data)


# In[3]:


# newDF = importPbz2('covidPollutionCensus.pbz2')
newDF = importPbz2('covidCensus.pbz2')
# newDF = importPbz2('covid.pbz2')


# In[4]:


newDF['fips'] = newDF['fips'].astype('str').str.zfill(5)
# newDF['State Code'] = newDF['State Code'].astype('str').str.zfill(5)
# newDF['County Code'] = newDF['County Code'].astype('str').str.zfill(5)
newDF['dates'] = pd.to_datetime(newDF['dates'])


# And explore...

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


# In[6]:


testDF = newDF.groupby([pd.Grouper(key='dates',freq='W'),'fips']).agg('mean')

# testDF = newDF.groupby('dates').agg('mean')


# In[7]:


testDF = testDF.reset_index()#.set_index(['dates','fips','pollutant'])


# In[8]:


testDF = testDF[testDF['dates'].between('2020-03-11','2021-03-11')]


# In[9]:


testDF = testDF.sort_values(['dates','fips'])


# In[10]:


wokka=[i for i in testDF if not (i.endswith('.missing'))|(i.startswith('NYT'))]
testDF = testDF[wokka]


# In[11]:


def rateCalc( numerators, denominator, DF  ):
    for col in numerators:
        DF[col+'Rate'] = DF[col] / DF[denominator]


# In[12]:


testDF['density']=testDF['TotalPopulation.data']/testDF['LND110210']
testDF['caseRate']=testDF['JHU_ConfirmedCases.data'] / testDF['latestTotalPopulation']
testDF['deathRate']=testDF['JHU_ConfirmedDeaths.data'] / testDF['latestTotalPopulation']


# In[50]:


# testingDF = testDF[testDF['movedWithinState'].notna()].copy()
# co = testingDF.sample(1)['fips'].values
# print(co[0])
# testingDF = testingDF.loc[testDF['fips']==co[0] ].groupby('dates').agg('mean')

ratesDF = testDF.copy()

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
          'hospitalStaffedBeds'], 'latestTotalPopulation', ratesDF)

rateCalc(['MaleAndFemale_AtLeast65_Population.data'], 'latestTotalPopulation', ratesDF)

ratesDF['healthInsRates']  = ratesDF[['healthInsNativeWith','healthInsForeignNatWith','healthInsForeignNoncitWith']].sum(axis=1) / ratesDF['healthInsTotal']
ratesDF['householdsWith65Rate'] = ratesDF[['houseWith65Rate','house2+with65Rate','houseNonfam65Rate','houseFamily65']].sum(axis=1)
ratesDF.drop(columns=['houseWith65Rate', 
              'house2+with65Rate',
              'houseFamily65Rate',
              'houseNonfam65Rate',
              'houseNo65Rate',
              'house2+No65Rate',
              'houseFamilyNo65Rate',
              'houseNonfamNo65Rate'], inplace=True)


# In[42]:


keep = ['dates','fips','AverageDailyTemperature.data','AveragePrecipitationTotal.data','AverageWindSpeed.data','BLS_UnemploymentRate.data',       'gini','hoursWorkedMean','unitsInStructure','density','householdIncomeMedian']
wokka = [x for x in ratesDF if x.endswith('Rate')]
keep.extend(wokka)
ratesDF = ratesDF[keep]


# In[69]:


cols = ['dates','fips','deathRate','householdIncomeMedian','AverageDailyTemperature.data','gini','hospitalStaffedBedsRate','publicTransRate','householdsWith65Rate','density']
rateFipsMean = ratesDF.loc[:,cols].groupby('fips').agg('mean')
rateFipsMean

rateCorrs = rateFipsMean.corr()#.sort_values('deathRate',ascending=False,key=abs)
rateCorrs.round(3)
ratesDF['fips'].nunique()


# In[62]:


# p values
pVals = rateFipsMean.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(len(rateFipsMean.columns)) 
pVals.set_index(rateCorrs.index).round(3)

