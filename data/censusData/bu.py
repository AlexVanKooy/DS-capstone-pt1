#!/usr/bin/env python
# coding: utf-8

# Libraries and options

# In[2]:


import numpy as np, pandas as pd, sys, os, re, zipfile, shutil, pickle, matplotlib.pyplot as plt, seaborn as sns, bz2
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
pd.options.display.max_colwidth = 50


# Importing census data

# In[ ]:


censusDF = pd.read_csv('./censusData.csv')
varDF = pd.read_csv("./acs2019Variables.csv", skiprows=[1,2,3])
pd.options.display.max_colwidth=1000
#change state and county fip codes to strings with standard lengths
censusDF['state'] = censusDF['state'].astype(str).str.zfill(2)
censusDF['county'] = censusDF['county'].astype(str).str.zfill(3)
#join these to create the combined fips code
censusDF['fips'] = censusDF['state']+censusDF['county']
#create a list of descriptive variable names in the same order as in censusDF
a = []
for i in censusDF.columns[:-4]:
    vName = varDF['label'][varDF['name']==i].to_string(index=False)+varDF['concept'][varDF['name']==i].to_string(index=False)
    a.append(vName)
for i in censusDF.columns[-4:]:
    a.append(i)
#rebuild dataframe with these column names
# censusDF = pd.DataFrame(censusDF.values, columns=a)
#et voila


# Importing covid data

# In[ ]:


path = r'../covid/processed_data/county_merged_parts/'
listFiles = []
for r, d, files in os.walk(path):
    for file in files:
        listFiles.append(file)

testDF = pd.DataFrame()
for i,file in enumerate(listFiles):
    inDF = pd.read_pickle(f'{path}{file}')
    inDF.reset_index(level=0,inplace=True)
    testDF = pd.concat([testDF,inDF],ignore_index=True)
# test = f'{path}{listFiles[0]}'
# testDF= pd.read_pickle(test)

# mask = testDF['fips'].isnull()
testDF.dropna(subset=['fips'], inplace=True)

testDF['fips']=testDF['fips'].astype(int).astype(str)

testDF['fips'] = testDF['fips'].str.zfill(5)


# Joining census and covid data

# In[ ]:


newDF = testDF.join(censusDF.set_index('fips'),on='fips',how='left')


# Exported this DF for future work

# In[ ]:


# newDF.to_csv('covidCensus.gz',index=False,compression='gzip')


# Imported back

# In[ ]:


anotherDF = pd.read_csv('covidCensus.gz')
anotherDF['fips'] = anotherDF['fips'].astype(str)
anotherDF['dates']=pd.to_datetime(anotherDF['dates'])


# Import air quality data

# In[ ]:


path = r'../air_quality/'
dirs = next(os.walk(path))[1]
airDF = pd.DataFrame()
cols = []
for dir in dirs:
    files = next(os.walk(f'{path}{dir}/unzipped/'))[2]
    for m,file in enumerate(files):
        inDF = pd.read_csv(f'{path}{dir}/unzipped/{file}')
        cols.append(inDF.columns.values)
        inDF['Date of Last Change'] = pd.to_datetime(inDF['Date of Last Change'])
        inDF['fips'] = inDF['fips'].astype(str).str.zfill(5)
        prefix = re.search('(?<=daily_)[^_]*(?=_)', f'{file}').group(0)
        print(prefix)
        inDF['pollutant'] = prefix
#         inDF = inDF.add_prefix(prefix+'_')
        airDF =pd.concat([airDF,inDF],ignore_index=True)


# In[ ]:


airDF.columns


# In[ ]:


for m,col in enumerate(cols[1:]):
    try:
        print(m+1,all(col==cols[0]),"col0")
    except:
        print(m+1,all(col==cols[3]),"col3")


# In[ ]:


# pm25DF['Date of Last Change']
anotherDF['dates']


# In[ ]:


print(anotherDF['dates'].dtype)
print(airDF['Date of Last Change'].dtype)
print(anotherDF['fips'].dtype)
print(airDF['fips'].dtype)


# Put it all together

# In[ ]:


newDF = anotherDF.merge(airDF, left_on=['dates','fips'], right_on=['Date of Last Change','fips'], how='inner')


# In[ ]:


# newDF.to_csv('allTogetherNow.gz',index=False,compression='gzip')
outfile = open('allTogetherNow.pkl','wb')
pickle.dump(newDF,outfile)
outfile.close()


# Import back in

# In[6]:


def importPbz2( file ):
    data = bz2.BZ2File(file,'rb')
    # newDF = pd.read_pickle('allTogetherNow.pkl')
    return pd.read_pickle(data)

newDF = importPbz2('allTogetherNow.pbz2')


# And explore...

# In[ ]:


print(newDF.shape)
mask = newDF.isnull().any(axis=0)
noNullDF = newDF.loc[:,~mask]
print(noNullDF.shape)


# In[21]:


newDF['density']=newDF['latestTotalPopulation']/newDF['LND110210']


# In[22]:


# a = [x for x in newDF if x.startswith('ozone')]
b = [x for x in newDF if (x.startswith('JHU')|x.startswith('NYT'))&(~x.endswith('missing'))&('Confirmed' in x)]
# b.extend(a)
b
no2DF = newDF.loc[newDF['pollutant']=='no2',b+['Observation Count','Observation Percent','Arithmetic Mean',                                                           '1st Max Value','Mean ugm3','fips','dates','density']]
# ozoneDF.dropna(subset=a,inplace=True)


# In[29]:


# ozoneDF.dropna(subset=['Mean ugm3'],inplace=True)
no2fipsDF = no2DF[(no2DF['dates']>='2020-03-11')&(no2DF['dates']<='2021-03-11')].groupby(by=['fips']).agg({'JHU_ConfirmedDeaths.data': 'max', 'NYT_ConfirmedDeaths.data': 'max',                              'Mean ugm3': 'mean','density': 'max'})


# In[30]:


no2fipsDF.corr()


# In[26]:


def covidPollutantFipsCorr(df, pollutant, cols=[]):
    if cols:
        polDF =  df.loc[df['pollutant']==pollutant,cols+['fips']]
    else:
        polDF = df.loc[df['pollutant']==pollutant,:]
    fipsDF  = polDF.groupby(by='fips').agg('mean')
    return fipsDF.corr()


# In[31]:


dateFilterDF = newDF[(newDF['dates']>='2020-03-11')|(newDF['dates']<='2021-03-11')]


# NO2

# In[42]:


covidPollutantFipsCorr(dateFilterDF, 'no2', b[:5]+['Arithmetic Mean','density'])


# Lead

# In[43]:


covidPollutantFipsCorr(dateFilterDF, 'lead', b[:5]+['Arithmetic Mean','density'])


# Ozone

# In[44]:


covidPollutantFipsCorr(dateFilterDF, 'ozone', b[:5]+['Arithmetic Mean','density'])


# PM2.5

# In[45]:


covidPollutantFipsCorr(dateFilterDF, 'pm25', b[:5]+['Arithmetic Mean','density'])


# In[80]:


a = no2fipsDF['Arithmetic Mean'].quantile([0,0.25,0.5,0.75,1])
b = no2fipsDF['JHU_ConfirmedDeaths.data'].quantile([0,0.25,0.5,0.75,1])


# In[82]:


print(a)
print(b)


# In[ ]:


corrs =  newDF.corr()


# In[4]:


numbDF = newDF.loc[:,newDF.dtypes!='object']
corrs = numbDF.corr()


# In[46]:


# # newDF.to_csv('allTogetherNow.gz',index=False,compression='gzip')
# with bz2.BZ2File('allTogetherNow.pbz2','wb') as f:
#     pickle.dump(newDF,f)

