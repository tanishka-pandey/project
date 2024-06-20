#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# Index
# 

# #                           Description 

# ## Data
# The data contains the details for the Uber rides across various boroughs (subdivisions) of New York City at an hourly level and attributes associated with weather conditions at that time.
# * pickup_dt: Date and time of the pick-up.
# * borough: NYC's borough.
# * pickups: Number of pickups for the period (hourly).
# * spd: Wind speed in miles/hour.
# * vsb: Visibility in miles to the nearest tenth.
# * temp: Temperature in Fahrenheit.
# * dewp: Dew point in Fahrenheit.
# * slp: Sea level pressure.
# * pcp01: 1-hour liquid precipitation.
# * pcp06: 6-hour liquid precipitation.
# * pcp24: 24-hour liquid precipitation.
# * sd: Snow depth in inches.
# * hday: Being a holiday (Y) or not (N).
# 

# In[1]:


import os 
os.getcwd()


# # Importing libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing the data 

# In[3]:


df=pd.read_csv(r"C:\Users\Tanishka pandey\Desktop\3-Uber_Data_New.csv")


# ## Display the top 5 rows

# In[4]:


df.head()


# ## Observation 
# * hday has wrong entries which need to be corrected.
# * temp has some null values which needs to be rectified later.

# ## Display the last 5 rows.
# 

# In[5]:


df.tail()


# # Observation 
# * borough has  some null entries it should be corrected.

# ## Check the shape of dataset

# In[6]:


df.shape


# ## Observation
# * This dataset contains 29101 rows and 13 columns.

# ## Check the datatype

# In[7]:


df.dtypes


# # Observation
# * datatype of hday should be categorical.
# * datatype of borough should be string.

# ## Check the statistical summary

# In[8]:


df.describe().T


# ## Observation
# * Values are missing from various columns. 
# * Extreme high values in case of slp.

# ## check the null values
# 

# In[9]:


df.isnull().sum()


# ## Observation
# It has null values on two columns, which need to be corrected or replaced.

# ## Check the duplicates values

# In[10]:


df.duplicated().any()


# In[11]:


df.duplicated().sum()


# In[12]:


df.T.duplicated() #column wise


# ## Observation
# There is no duplicated values in the given dataset.

# # Check the outliers and their authenticity.

# In[13]:


plt.figure(figsize=(5,15))
sns.boxplot(data=df,orient='h')
plt.show()


# ## Observe
# It is a combine boxplot figure for all columns.

# In[14]:


plt.figure(figsize=(12,12))

# Loop through each column (excluding non-numeric columns like 'pickup_dt' and 'borough')
for i, col in enumerate(df.select_dtypes(include=['number'])):
    plt.subplot(3, 4, i + 1)  # Adjust subplot layout if your dataset has more columns
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()


# ## Observe 
# These seperate boxplots are giving a better insights abount outliers.

# ## Check the anomalies or wrong entries.
# 

# In[15]:


df['borough'] = df['borough'].astype('string')
df['hday'] = df['hday'].astype('category')


# ## Observation 
# Correcting the datatype of some variable

# In[16]:


#Checking for negative values where not expected
negative_values_columns = ['pickups', 'spd', 'vsb', 'temp', 'dewp', 'slp', 'pcp01', 'pcp06', 'pcp24', 'sd']
for col in negative_values_columns:
    if (df[col] < 0).any():
        print(f"Anomalies in {col}:")
        print(df[df[col] < 0])


# ## Observe
# There are some negative values present in dewp column. As they are possible to have them but still they are less common.

# In[17]:


#temperature above boiling point
if (df['temp'] > 212).any():  # boiling point of water in Fahrenheit
    print("Anomalies in 'temp':")
    print(df[df['temp'] > 212])


# ## Observe 
# Temperature above boiling point of water is a wrong entry.

# In[18]:


categorical_columns=['hday','borough']
for col in categorical_columns:
    unique_values = df[col].unique()
    print(f"Unique values in '{col}': {unique_values}")


# ## Observe 
# Unique values calculations is there for two categorical column.

# # The necessary data cleaning 

# ### First step:-
# 
# * Checking for null values.
# * Checking the percentage of null values, if it is greater than 20 then we would drop it. 

# In[19]:


print("Number of duplicate rows:", df.duplicated().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Confirm duplicates are dropped
print("Number of duplicate rows after dropping:", df.duplicated().sum())


# In[20]:


df.T.duplicated() #columnwise checking


# ## Observation 
# There is no duplicated values.

# ### Second step:-
# * Replacing wrong values with null values.

# In[21]:


df[df['hday']== '?']


# In[22]:


df['hday']=df['hday'].replace("?",np.nan)


# ### Thrid step:-
# * checking the null values in all columns.

# In[23]:


print("Null values in each column:")
print(df.isnull().sum())

print("Percentage wise null values :")
print(df.isnull().sum()/len (df)*100)


# ### Fourth step:-
# ### Decting  outliers-
# * Detcting and removing outliers.
# * Replacing numerical columns with  outliers by median values.
# * Replacing numerical columns without outliers by mean values.
# * Replacing categorical columns with mode values.
# * Replacing object datatype values with null values.

# In[24]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range  


# In[25]:


ll,ul =remove_outlier(df['pickups'])


# In[26]:


df[df['pickups']>ul]


# In[27]:


df[df['pickups']<ll]


# In[28]:


median1=df['pickups'].median()
df['pickups'].replace(np.nan,median1,inplace=True)


# In[29]:


for col in 'hday':
    mode_value = df['hday'].mode()[0]  
    df['hday'].fillna(mode_value, inplace=True)


# In[30]:


df['temp'].fillna(df['temp'].mean(), inplace=True)


# In[31]:


df['borough']=df['borough'].replace(np.nan)


# In[32]:


df.isnull().sum()


# ## Outliers Tretment
# ### Step one- Detecting for outliers.

# In[36]:


for i in ['pickups','spd','vsb','slp','pcp01','pcp06','pcp24','sd']:
    plt.figure(figsize=(10,3))
    df.boxplot(column = i)
    plt.show()


# ### Step two- Removing them

# In[39]:


def remove_outliers(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range 


# In[41]:


lrpickups,urpickups=remove_outliers(df['pickups'])
df['pickups']=np.where(df['pickups']>urpickups,urpickups,df['pickups'])
df['pickups']=np.where(df['pickups']<lrpickups,lrpickups,df['pickups'])

lrspd,urspd=remove_outliers(df['spd'])
df['spd']=np.where(df['spd']>urspd,urspd,df['spd'])
df['spd']=np.where(df['spd']<lrspd,lrspd,df['spd'])

lrvsb,urvsb=remove_outliers(df['vsb'])
df['vsb']=np.where(df['vsb']>urvsb,urvsb,df['vsb'])
df['vsb']=np.where(df['vsb']<lrvsb,lrvsb,df['vsb'])

lrslp,urslp=remove_outliers(df['slp'])
df['slp']=np.where(df['slp']>urslp,urslp,df['slp'])
df['slp']=np.where(df['slp']<lrslp,lrslp,df['slp'])

lrpcp01,urpcp01=remove_outliers(df['pcp01'])
df['pcp01']=np.where(df['pcp01']>urpcp01,urpcp01,df['pcp01'])
df['pcp01']=np.where(df['pcp01']<lrpcp01,lrpcp01,df['pcp01'])

lrpcp06,urpcp06=remove_outliers(df['pcp06'])
df['pcp06']=np.where(df['pcp06']>urpcp06,urpcp06,df['pcp06'])
df['pcp06']=np.where(df['pcp06']<lrpcp06,lrpcp06,df['pcp06'])

lrpcp24,urpcp24=remove_outliers(df['pcp24'])
df['pcp24']=np.where(df['pcp24']>urpcp24,urpcp24,df['pcp24'])
df['pcp24']=np.where(df['pcp24']<lrpcp24,lrpcp24,df['pcp24'])

lrsd,ursd=remove_outliers(df['sd'])
df['sd']=np.where(df['sd']>ursd,ursd,df['sd'])
df['sd']=np.where(df['sd']<lrsd,lrsd,df['sd'])


# ### Step three- Rechecking to ddetect if any outliers remains.

# In[42]:


for i in ['pickups','spd','vsb','slp','pcp01','pcp06','pcp24','sd']:
    plt.figure(figsize=(10,3))
    df.boxplot(column = i)
    plt.show()


# # Observe-
# ### All the ouliers are succesfully removed.

# In[ ]:




