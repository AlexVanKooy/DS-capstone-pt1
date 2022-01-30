#!/usr/bin/env python
# coding: utf-8

# Libraries and options

# In[1]:


import numpy as np, pandas as pd, pickle, bz2, time


# Functions

# In[13]:


def exportPbz2( outfile, infile ):
    start = time.time()
    with bz2.BZ2File(outfile+'.pbz2','wb') as file:
        pickle.dump(infile,file)
    print(time.time()-start)
    
def importPbz2( file ):
    data = bz2.BZ2File(file,'rb')
    return pd.read_pickle(data)


# Importing census data

# In[3]:


censusDF = pd.read_csv('../censusData/censusData.csv')

censusDF['fips'] = censusDF['fips'].astype('str').str.zfill(5)
censusDF['state']  = censusDF['state'].astype('str').str.zfill(2)
censusDF['county'] = censusDF['county'].astype('str').str.zfill(3)


# In[4]:


cols=['totalMoved','movedWithinState','movedWithoutState','movedFromAbroad','publicTrans','totalTrans','householdsTotal',
      'houseWith65','house2+with65','houseFamily65','houseNonfam65','houseNo65',
      'house2+No65','houseFamilyNo65','houseNonfamNo65','householdStructuresTotal','householdIncomeMedian',
      'gini','hoursWorkedMean','unitsInStructure','healthInsTotal','healthInsNativeWith',
      'healthInsForeignNatWith','healthInsForeignNoncitWith','healthInsForeignNatNo','healthInsForeignNoncitNo','healthInsNativeNo',
      'countyStateName','stateFip','countyFip','fips'
     ]
censusDF = pd.DataFrame(censusDF.values, columns=cols)


# Importing covid data

# In[5]:


testDF= pd.read_pickle('../covid/covid_data_df.pkl')

# mask = testDF['fips'].isnull()
testDF.dropna(subset=['fips'], inplace=True)

testDF['fips']=testDF['fips'].astype(int).astype(str)

testDF['fips'] = testDF['fips'].str.zfill(5)


# Joining census and covid data

# In[6]:

testDF = testDF.merge(censusDF, left_on=['fips'], right_on=['fips'], how='left')

# Import pm25

# In[8]:


pm25DF = pd.read_csv('../pm25/county_pm25.txt')
pm25DF.drop(['year'],axis=1,inplace=True)


# In[9]:

pm25DF['fips'] = pm25DF['fips'].astype(str).str.zfill(5)
pm25DF= pm25DF[['fips','pm25']].groupby(['fips'],as_index=False).mean()


# In[10]:


pm25DF['fips'] = pm25DF['fips'].astype(str).str.zfill(5)


# Put it all together

# In[11]:


testDF = testDF.merge(pm25DF, left_on=['fips'], right_on=['fips'], how='inner')


# In[12]:


exportPbz2('feeFiFoFum', testDF)

