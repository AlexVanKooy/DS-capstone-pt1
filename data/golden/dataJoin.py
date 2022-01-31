#!/usr/bin/env python
# coding: utf-8

# Libraries and options

# In[1]:


import numpy as np, pandas as pd, pickle, bz2, time

pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
pd.get_option('display.max_columns'),pd.get_option('display.max_rows')


# Functions

# In[2]:


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


# In[6]:


testDF.drop('Date Local',axis=1,inplace=True)


# Joining census and covid data

# In[7]:


testDF = testDF.merge(censusDF, left_on=['fips'], right_on=['fips'], how='left')


# In[8]:




# Import pm25

# In[9]:


pm25DF = pd.read_csv('../pm25/county_pm25.txt')
pm25DF.drop(['year'],axis=1,inplace=True)


# In[10]:


pm25DF['fips'] = pm25DF['fips'].astype(str).str.zfill(5)


# In[11]:


pm25DF= pm25DF[['fips','pm25']].groupby(['fips'],as_index=False).mean()


# Put it all together

# In[12]:


testDF = testDF.merge(pm25DF, left_on=['fips'], right_on=['fips'], how='inner')


# In[13]:


testDF=testDF.infer_objects()


# In[14]:


print(testDF.shape)
a = testDF.dtypes


# In[15]:


b = testDF.describe()
_=[testDF.drop(col, axis=1, inplace=True) for col in b.columns if b.loc['std',col]==0]


# In[16]:


print(testDF.shape)
c = testDF.dtypes


# In[17]:


a.index[~a.index.isin(c.index)]


# In[19]:


exportPbz2('feeFiFoFum', testDF)


# In[ ]:




