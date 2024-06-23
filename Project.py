#!/usr/bin/env python
# coding: utf-8

# ![image-2.png](attachment:image-2.png)

# ![image.png](attachment:image.png)

# #                          Problem Outline

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

# # Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing the data 

# In[2]:


df=pd.read_csv(r"C:\Users\Tanishka pandey\Desktop\3-Uber_Data_New.csv")


# ## Display the top 5 rows

# In[3]:


df.head()


# ## Analyse
# * hday has wrong entries which need to be corrected.
# * temp has some null values which needs to be rectified later.

# ## Display the last 5 rows.
# 

# In[4]:


df.tail()


# ## Analyse
# * borough has  some null entries it should be corrected.

# ## Check the shape of dataset

# In[5]:


df.shape


# ## Analyse
# * This dataset contains 29101 rows and 13 columns.

# ## Check the datatype

# In[6]:


df.dtypes


# ## Analyse
# * datatype of hday should be categorical.
# * datatype of borough should be string.

# ## Check the statistical summary

# In[7]:


df.describe().T


# ## Analyse
# 
# * Values are missing from various columns. 
# * Extreme high values in case of slp.

# ## check the null values
# 

# In[8]:


df.isnull().sum()


# ## Analyse
# It has null values on two columns, which need to be corrected or replaced.

# ## Check the duplicates values

# In[9]:


df.duplicated().any()


# In[10]:


df.duplicated().sum()


# In[11]:


df.T.duplicated() #column wise


# ## Analyse
# There is no duplicated values in the given dataset.

# # Check the outliers and their authenticity.

# In[12]:


plt.figure(figsize=(10,10))
sns.boxplot(data=df,orient='h')
plt.show()


# ## Analyse
# It is a combine boxplot figure for all columns ,showing outliers.

# In[13]:


plt.figure(figsize=(12,12))

# Loop through each column (excluding non-numeric columns like 'pickup_dt' and 'borough')
for i, col in enumerate(df.select_dtypes(include=['number'])):
    plt.subplot(3, 4, i + 1)  # Adjust subplot layout if your dataset has more columns
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()


# ## Analyse
# These seperate boxplots are giving a better insights abount outliers.

# ## Check the anomalies or wrong entries.
# 

# In[14]:


df['borough'] = df['borough'].astype('string')
df['hday'] = df['hday'].astype('category')


# ## Analyse
# Correcting the datatype of some variable and making the data consistent.

# In[15]:


#Checking for negative values where not expected
negative_values_columns = ['pickups', 'spd', 'vsb', 'temp', 'dewp', 'slp', 'pcp01', 'pcp06', 'pcp24', 'sd']
for col in negative_values_columns:
    if (df[col] < 0).any():
        print(f"Anomalies in {col}:")
        print(df[df[col] < 0])


# ## Analyse
# There are some negative values present in dewp column. As they are possible to have them but still they are less common.

# In[16]:


#temperature above boiling point
if (df['temp'] > 212).any():  # boiling point of water in Fahrenheit
    print("Anomalies in 'temp':")
    print(df[df['temp'] > 212])


# ## Analyse
# 
# Temperature above boiling point of water is a wrong entry.

# In[17]:


categorical_columns=['hday','borough']
for col in categorical_columns:
    unique_values = df[col].unique()
    print(f"Unique values in '{col}': {unique_values}")


# ## Analyse
# Unique values calculations is there for two categorical column.

# # The necessary data cleaning 

# ### First step:-
# 
# * Checking for null values.
# * Checking the percentage of null values, if it is greater than 20 then we would drop it. 

# In[18]:


print("Number of duplicate rows:", df.duplicated().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Confirm duplicates are dropped
print("Number of duplicate rows after dropping:", df.duplicated().sum())


# In[19]:


df.T.duplicated() #columnwise checking


# ## Analyse
# There is no duplicated values.

# ### Second step:-
# * Replacing wrong values with null values.

# In[20]:


df[df['hday']== '?']


# In[21]:


df['hday']=df['hday'].replace("?",np.nan)


# ## Analyse
# Succesfully removed all the null entries.

# ### Thrid step:-
# * checking the null values in all columns.

# In[22]:


print("Null values in each column:")
print(df.isnull().sum())

print("Percentage wise null values :")
print(df.isnull().sum()/len (df)*100)


# ## Analyse
# dropping the columns which have more than 20 % of null values.

# ### Fourth step:-
# ### Decting  outliers-
# * Detcting and removing outliers.
# * Replacing numerical columns with  outliers by median values.
# * Replacing numerical columns without outliers by mean values.
# * Replacing categorical columns with mode values.
# * Replacing object datatype values with null values.

# In[23]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range  


# In[24]:


ll,ul =remove_outlier(df['pickups'])


# In[25]:


df[df['pickups']>ul]


# In[26]:


df[df['pickups']<ll]


# In[27]:


median1=df['pickups'].median()
df['pickups'].replace(np.nan,median1,inplace=True)


# In[28]:


for col in 'hday':
    mode_value = df['hday'].mode()[0]  
    df['hday'].fillna(mode_value, inplace=True)


# In[29]:


df['temp'].fillna(df['temp'].mean(), inplace=True)


# In[30]:


df['borough']=df['borough'].replace(np.nan)


# In[31]:


df.isnull().sum()


# ## Outliers Tretment
# ### Step one- Detecting for outliers.

# In[32]:


for i in ['pickups','spd','vsb','slp','pcp01','pcp06','pcp24','sd']:
    plt.figure(figsize=(10,3))
    df.boxplot(column = i)
    plt.show()


# ### Step two- Removing them

# In[33]:


def remove_outliers(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range 


# In[34]:


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

# In[35]:


for i in ['pickups','spd','vsb','slp','pcp01','pcp06','pcp24','sd']:
    plt.figure(figsize=(10,3))
    df.boxplot(column = i)
    plt.show()


# # Analyse
# #### All the ouliers are succesfully removed.

# ### Adding one more column to the dataset

# In[36]:


z=[]
for j in [i[1].split(':') for i in df['pickup_dt'].str.split(' ')]:
    z. append(int(j[0]))
df['Hours']=z


# ## Analyse
# One more column is added succcesfully.

# # 1.	Pickup Analysis

# • Qa.What is the total number of Uber pickups across all boroughs?

# In[37]:


total_pickups = df['pickups'].sum()
print("Total pickups across all boroughs:",total_pickups)


# #### Conclusion -
# There are total 8207799 uber pickups across all boroughs.

# • Qb.Which borough has the highest average number of hourly pickups?

# In[38]:


average_hourly_pickups = df.groupby('borough')['pickups'].mean()
highest_avg_hourly_pickups_borough = average_hourly_pickups.idxmax()
highest_avg_hourly_pickups_value = average_hourly_pickups.max()
print(f"Borough with the highest average number of hourly pickups: {highest_avg_hourly_pickups_borough} with {highest_avg_hourly_pickups_value} pickups per hour")

# Plot the average number of hourly pickups per borough
plt.figure(figsize=(10, 6))
sns.barplot(x=average_hourly_pickups.index, y=average_hourly_pickups.values, palette='viridis')
plt.title('Average Number of Hourly Pickups per Borough')
plt.xlabel('Borough')
plt.ylabel('Average Number of Hourly Pickups')
plt.xticks(rotation=45)
plt.show()


# #### Conclusion- 
# Manhattan has highest average number of hourly pickups.

# •Qc.How do the number of pickups vary across different hours of the day?

# In[39]:


hourly_pickups = df.groupby('Hours')['pickups'].sum()
print("Number of pickups across different hours of the day:")
print(hourly_pickups)

# Plot the number of pickups across different hours of the day
plt.figure(figsize=(12, 6))
sns.lineplot(x=hourly_pickups.index, y=hourly_pickups.values, marker='o', color='b')
plt.title('Number of Uber Pickups by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Pickups')
plt.xticks(range(0, 24))
plt.grid(True)
plt.show()


# #### Conclusion-
# Line plot shows the variation in the pickups vary across different hours of the day.

# •Qd.Which day of the week has the highest number of pickups?

# ### Adding one more column of day_of_week

# In[40]:


z=[]
for j in [i[1].split(':') for i in df['pickup_dt'].str.split(' ')]:
    z. append(int(j[0]))
df['day_of_week']=z


# In[41]:


# Converting pickup_dt to date and time format.
df['pickup_dt'] = pd.to_datetime(df['pickup_dt'], errors='coerce')

# Drop rows where 'pickup_dt' conversion failed
df = df.dropna(subset=['pickup_dt'])

# Extract day of the week and hour
df['day_of_week'] = df['pickup_dt'].dt.day_name()
df['hour'] = df['pickup_dt'].dt.hour

# Check for successful extraction
print("Unique values in 'day_of_week' column:", df['day_of_week'].unique())

# 4. Day of the week with the highest number of pickups
daily_pickups = df.groupby('day_of_week')['pickups'].sum()
print("\nAggregated pickups by day of the week:")
print(daily_pickups)

highest_pickup_day = daily_pickups.idxmax()
highest_pickup_day_value = daily_pickups.max()
print(f"\nDay of the week with the highest number of pickups: {highest_pickup_day} with {highest_pickup_day_value} pickups")

# Plot the number of pickups per day of the week
plt.figure(figsize=(10, 6))
sns.barplot(x=daily_pickups.index, y=daily_pickups.values, palette='coolwarm')
plt.title('Number of Uber Pickups by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Pickups')
plt.xticks(rotation=45)
plt.show()


# #### Conclusion -
# Friday has the highest number of pickups.

# # 2. Weather Impact

# •Qa.What is the correlation between temperature and the number of pickups?

# In[42]:


temp_pickup_corr = df['temp'].corr(df['pickups'])
print(f"Correlation between temperature and the number of pickups: {temp_pickup_corr}")

# Plot temperature vs. pickups
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='pickups', data=df, alpha=0.5)
plt.title('Temperature vs. Number of Pickups')
plt.xlabel('Temperature (F)')
plt.ylabel('Number of Pickups')
plt.grid(True)
plt.show()


# #### Conclusion
# There is a moderate positive correlation between temperature and the number of Uber pickups.

# •Qb.How does visibility impact the number of pickups?

# In[43]:


vsb_pickup_corr = df['vsb'].corr(df['pickups'])
print(f"Correlation between visibility and the number of pickups: {vsb_pickup_corr}")

# Plot visibility vs. pickups
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vsb', y='pickups', data=df, alpha=0.5)
plt.title('Visibility vs. Number of Pickups')
plt.xlabel('Visibility (miles)')
plt.ylabel('Number of Pickups')
plt.grid(True)
plt.show()


# #### Conclusion-
# Visibility has a weak positive correlation with the number of Uber pickups.
# 

# •Qc.Is there a relationship between wind speed and the number of pickups?

# In[44]:


spd_pickup_corr = df['spd'].corr(df['pickups'])
print(f"Correlation between wind speed and the number of pickups: {spd_pickup_corr}")

# Plot wind speed vs. pickups
plt.figure(figsize=(10, 6))
sns.scatterplot(x='spd', y='pickups', data=df, alpha=0.5)
plt.title('Wind Speed vs. Number of Pickups')
plt.xlabel('Wind Speed (mph)')
plt.ylabel('Number of Pickups')
plt.grid(True)
plt.show()


# #### Conclusion-
# Wind speed shows a very weak negative correlation with the number of Uber pickups.
# 

# •Qd.How does precipitation (1-hour, 6-hour, 24-hour) affect the number of pickups?

# In[45]:


pcp01_pickup_corr = df['pcp01'].corr(df['pickups'])
pcp06_pickup_corr = df['pcp06'].corr(df['pickups'])
pcp24_pickup_corr = df['pcp24'].corr(df['pickups'])
print(f"Correlation between 1-hour precipitation and the number of pickups: {pcp01_pickup_corr}")
print(f"Correlation between 6-hour precipitation and the number of pickups: {pcp06_pickup_corr}")
print(f"Correlation between 24-hour precipitation and the number of pickups: {pcp24_pickup_corr}")

# Plot precipitation (1-hour) vs. pickups
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pcp01', y='pickups', data=df, alpha=0.5)
plt.title('1-Hour Precipitation vs. Number of Pickups')
plt.xlabel('1-Hour Precipitation (inches)')
plt.ylabel('Number of Pickups')
plt.grid(True)
plt.show()

# Plot precipitation (6-hour) vs. pickups
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pcp06', y='pickups', data=df, alpha=0.5)
plt.title('6-Hour Precipitation vs. Number of Pickups')
plt.xlabel('6-Hour Precipitation (inches)')
plt.ylabel('Number of Pickups')
plt.grid(True)
plt.show()

# Plot precipitation (24-hour) vs. pickups
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pcp24', y='pickups', data=df, alpha=0.5)
plt.title('24-Hour Precipitation vs. Number of Pickups')
plt.xlabel('24-Hour Precipitation (inches)')
plt.ylabel('Number of Pickups')
plt.grid(True)
plt.show()


# #### Conclusion-
# Precipitation (1-hour, 6-hour, 24-hour) has a very weak negative correlation with the number of Uber pickups.
# 
# 
# 
# 
# 
# 

# # 3. Seasonal Trends

# •Qa.How do the number of pickups vary across different seasons (winter, spring, summer, fall)?

# In[46]:


# Creating a season column
df['month'] = df['pickup_dt'].dt.month
df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                             'Spring' if x in [3, 4, 5] else
                                             'Summer' if x in [6, 7, 8] else
                                             'Fall')


# In[47]:


seasonal_pickups = df.groupby('season')['pickups'].sum()
print(f"\nNumber of pickups across different seasons:\n{seasonal_pickups}")

# Plot number of pickups by season
plt.figure(figsize=(10, 6))
sns.barplot(x=seasonal_pickups.index, y=seasonal_pickups.values, palette='autumn')
plt.title('Number of Uber Pickups by Season')
plt.xlabel('Season')
plt.ylabel('Number of Pickups')
plt.show()


# #### Conclusion-
# Pickups vary across seasons, with summer having the highest number of pickups.

# •Qb.What is the average number of pickups during holidays compared to non-holidays?

# In[48]:


holiday_pickups = df[df['hday'] == 'Y']['pickups'].mean()
non_holiday_pickups = df[df['hday'] == 'N']['pickups'].mean()
print(f"\nAverage number of pickups on holidays: {holiday_pickups:.2f}")
print(f"Average number of pickups on non-holidays: {non_holiday_pickups:.2f}")

# Plot average pickups on holidays vs. non-holidays
plt.figure(figsize=(10, 6))
sns.barplot(x=['Holiday', 'Non-Holiday'], y=[holiday_pickups, non_holiday_pickups], palette='coolwarm')
plt.title('Average Number of Uber Pickups: Holidays vs. Non-Holidays')
plt.xlabel('Day Type')
plt.ylabel('Average Number of Pickups')
plt.show()


# #### Conclusion-
# Average number of pickups is higher during holidays compared to non-holidays.

# •Qc.How does snow depth influence the number of pickups?

# In[49]:


snow_depth_pickups = df.groupby('sd')['pickups'].mean()
print(f"\nNumber of pickups by snow depth:\n{snow_depth_pickups}")

# Plot snow depth vs. average number of pickups
plt.figure(figsize=(10, 6))
sns.lineplot(x=snow_depth_pickups.index, y=snow_depth_pickups.values, marker='o')
plt.title('Average Number of Uber Pickups by Snow Depth')
plt.xlabel('Snow Depth (inches)')
plt.ylabel('Average Number of Pickups')
plt.grid(True)
plt.show()


# #### Conclusion-
# Snow depth generally shows a decrease in pickups with increasing depth.

# # 4. Hourly Trends

# •Qa.What are the peak hours for Uber pickups in each borough?

# In[50]:


# Extract hour from pickup_dt
df['Hours'] = df['pickup_dt'].dt.hour


# In[51]:


borough_hourly_pickups = df.groupby(['borough', 'hour'])['pickups'].sum().unstack()
print("\nPeak hours for Uber pickups in each borough:")
print(borough_hourly_pickups.idxmax(axis=1))

# Plot hourly pickups for each borough
plt.figure(figsize=(15, 10))
for borough in df['borough'].unique():
    hourly_data = df[df['borough'] == borough].groupby('hour')['pickups'].sum()
    plt.plot(hourly_data.index, hourly_data.values, label=borough)

plt.title('Hourly Uber Pickups by Borough')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Pickups')
plt.legend(title='Borough')
plt.grid(True)
plt.show()


# #### Conclusion-
# Peak hours for Uber pickups vary by borough, typically around evening rush hours.
# 

# •Qb.How do the number of pickups change during rush hours (e.g., 7-9 AM, 5-7 PM)?

# In[52]:


rush_hours_am = df[(df['hour'] >= 7) & (df['hour'] < 9)]
rush_hours_pm = df[(df['hour'] >= 17) & (df['hour'] < 19)]
non_rush_hours = df[(df['hour'] < 7) | ((df['hour'] >= 9) & (df['hour'] < 17)) | (df['hour'] >= 19)]

rush_am_pickups = rush_hours_am['pickups'].sum()
rush_pm_pickups = rush_hours_pm['pickups'].sum()
non_rush_pickups = non_rush_hours['pickups'].sum()
print(f"\nNumber of pickups during rush hours (7-9 AM): {rush_am_pickups}")
print(f"Number of pickups during rush hours (5-7 PM): {rush_pm_pickups}")
print(f"Number of pickups during non-rush hours: {non_rush_pickups}")

# Plot number of pickups during rush hours vs. non-rush hours
rush_hour_data = {'Rush Hours (7-9 AM)': rush_am_pickups, 'Rush Hours (5-7 PM)': rush_pm_pickups, 'Non-Rush Hours': non_rush_pickups}
plt.figure(figsize=(10, 6))
sns.barplot(x=list(rush_hour_data.keys()), y=list(rush_hour_data.values()), palette='viridis')
plt.title('Number of Uber Pickups During Rush Hours vs. Non-Rush Hours')
plt.xlabel('Time Period')
plt.ylabel('Number of Pickups')
plt.show()


# #### Conclusion-
# Uber pickups significantly increase during rush hours, especially in the evening (5-7 PM).
# 

# •Qc.What is the average number of pickups during late-night hours (e.g., 12 AM - 4 AM)?

# In[53]:


late_night_hours = df[(df['hour'] >= 0) & (df['hour'] < 4)]
average_late_night_pickups = late_night_hours['pickups'].mean()
print(f"\nAverage number of pickups during late-night hours (12 AM - 4 AM): {average_late_night_pickups:.2f}")

# Plot average number of pickups during late-night hours
late_night_avg = late_night_hours.groupby('hour')['pickups'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=late_night_avg.index, y=late_night_avg.values, palette='plasma')
plt.title('Average Number of Uber Pickups During Late-Night Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Number of Pickups')
plt.xticks([0, 1, 2, 3])
plt.grid(True)
plt.show()


# #### Conclusion-
# The average number of pickups during late-night hours (12 AM - 4 AM) is relatively lower compared to other times of the day.

# # 5.	Borough Comparison

# •Qa.How do pickup trends differ between boroughs during different weather conditions?

# In[54]:


df['weekday'] = df['pickup_dt'].dt.weekday

# Function to classify weekend vs. weekday
def is_weekend(day):
    if day in [5, 6]:  # Saturday or Sunday
        return 'Weekend'
    else:
        return 'Weekday'
df['day_type'] = df['weekday'].apply(is_weekend)


# In[55]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='temp', y='pickups', hue='borough', data=df, palette='Set1', alpha=0.7)
plt.title('Pickup Trends by Borough and Temperature')
plt.xlabel('Temperature (F)')
plt.ylabel('Number of Pickups')
plt.legend(title='Borough', loc='upper left')
plt.grid(True)
plt.show()


# #### Conclusion-
# Pickup trends vary across boroughs, influenced by different weather conditions such as temperature.

# •Qb.Which borough shows the highest increase in pickups during holidays?

# In[56]:


holiday_pickup_increase = df.groupby(['borough', 'hday'])['pickups'].sum().unstack()
holiday_pickup_increase['Holiday Increase'] = holiday_pickup_increase['Y'] - holiday_pickup_increase['N']
max_increase_borough = holiday_pickup_increase['Holiday Increase'].idxmax()
max_increase_value = holiday_pickup_increase['Holiday Increase'].max()
print(f"\nBorough with the highest increase in pickups during holidays: {max_increase_borough} with an increase of {max_increase_value} pickups.")


# #### Conclusion-
# EWR shows the highest increase in pickups during holidays compared to other boroughs.
# 

# •Qc.How does the number of pickups compare between weekdays and weekends for each borough?

# In[57]:


weekday_weekend_pickups = df.groupby(['borough', 'day_type'])['pickups'].mean().unstack()

plt.figure(figsize=(12, 8))
weekday_weekend_pickups.plot(kind='bar', stacked=False, alpha=0.75)
plt.title('Average Number of Pickups: Weekdays vs. Weekends by Borough')
plt.xlabel('Borough')
plt.ylabel('Average Number of Pickups')
plt.xticks(rotation=45)
plt.legend(title='Day Type')
plt.grid(True)
plt.show()


# #### Conclusion-
# Pickups are generally higher on weekends compared to weekdays across all boroughs, with variations depending on the borough.

# # 6. Weather extremes.

# •Qa.How do extreme weather conditions (e.g., very high or very low temperatures, heavy rainfall, snowstorms) affect the number of pickups?

# In[58]:


high_temp_threshold = 90  # Example threshold for high temperature (F)
low_temp_threshold = 20   # Example threshold for low temperature (F)
heavy_rain_threshold = 0.5  # Example threshold for heavy rainfall (inches)
snowstorm_threshold = 0   # Example threshold for snowstorm (snow depth in inches)

# Filter data for extreme weather conditions
extreme_weather = df[(df['temp'] >= high_temp_threshold) |
                     (df['temp'] <= low_temp_threshold) |
                     (df['pcp01'] > heavy_rain_threshold) |
                     (df['sd'] > snowstorm_threshold)]

# Plot impact of extreme weather conditions on pickups
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='pickups', hue='borough', data=extreme_weather, palette='Set1', alpha=0.7)
plt.title('Impact of Extreme Weather Conditions on Uber Pickups')
plt.xlabel('Temperature (F)')
plt.ylabel('Number of Pickups')
plt.legend(title='Borough', loc='upper left')
plt.grid(True)
plt.show()


# #### Conclusion-
# Extreme weather conditions (high/low temperatures, heavy rainfall, snowstorms) affect Uber pickups, with varying effects across boroughs.

# •Qb.What is the impact of visibility less than 1 mile on the number of pickups?

# In[59]:


low_visibility = df[df['vsb'] < 1]

# Plot impact of low visibility on pickups
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vsb', y='pickups', data=low_visibility, alpha=0.5)
plt.title('Impact of Visibility Less Than 1 Mile on Uber Pickups')
plt.xlabel('Visibility (miles)')
plt.ylabel('Number of Pickups')
plt.grid(True)
plt.show()


# #### Conclusion-
# Visibility less than 1 mile negatively impacts Uber pickups, indicating reduced activity during poor visibility conditions.

# #  7.	Data Correlations

# •Qa.Is there a correlation between sea level pressure and the number of pickups?

# In[60]:


sea_level_pressure_corr = df['slp'].corr(df['pickups'])

print(f"Correlation between sea level pressure and pickups: {sea_level_pressure_corr:.2f}")

# Plot sea level pressure vs. pickups
plt.figure(figsize=(10, 6))
sns.scatterplot(x='slp', y='pickups', data=df, alpha=0.5)
plt.title('Sea Level Pressure vs. Uber Pickups')
plt.xlabel('Sea Level Pressure')
plt.ylabel('Number of Pickups')
plt.grid(True)
plt.show()


# #### Conclusion-
# There is a weak positive correlation between sea level pressure and Uber pickups, indicating a minor influence of pressure on pickup numbers.
# 

# •Qb.How do different weather variables (temperature, dew point, wind speed, visibility) collectively impact the number of pickups?

# In[61]:


weather_variables = ['temp', 'dewp', 'spd', 'vsb']

plt.figure(figsize=(14, 10))
for i, var in enumerate(weather_variables, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=var, y='pickups', data=df, alpha=0.5)
    plt.title(f'{var.capitalize()} vs. Uber Pickups')
    plt.xlabel(var.capitalize())
    plt.ylabel('Number of Pickups')
    plt.grid(True)

plt.tight_layout()
plt.show()


# #### Conclusion-
# Temperature and visibility show noticeable impacts on Uber pickups, with higher temperatures and better visibility generally correlating with increased pickups.
# Dew point and wind speed exhibit weaker correlations with pickups compared to temperature and visibility.

# •Qc.What is the relationship between holiday status and weather conditions on the number of pickups?

# In[62]:


holiday_weather_pickups = df.groupby('hday').agg({
    'temp': 'mean',
    'dewp': 'mean',
    'spd': 'mean',
    'vsb': 'mean',
    'pickups': 'sum'  # Sum of pickups to get total pickups by holiday status
}).reset_index()

# Plot the relationship between weather conditions and pickups by holiday status
plt.figure(figsize=(14, 10))

# Plot for Temperature
plt.subplot(2, 2, 1)
sns.barplot(x='hday', y='temp', data=holiday_weather_pickups, palette='Set1')
plt.title('Average Temperature by Holiday Status')
plt.xlabel('Holiday')
plt.ylabel('Average Temperature (F)')
plt.grid(True)

# Plot for Dew Point
plt.subplot(2, 2, 2)
sns.barplot(x='hday', y='dewp', data=holiday_weather_pickups, palette='Set2')
plt.title('Average Dew Point by Holiday Status')
plt.xlabel('Holiday')
plt.ylabel('Average Dew Point (F)')
plt.grid(True)

# Plot for Wind Speed
plt.subplot(2, 2, 3)
sns.barplot(x='hday', y='spd', data=holiday_weather_pickups, palette='Set3')
plt.title('Average Wind Speed by Holiday Status')
plt.xlabel('Holiday')
plt.ylabel('Average Wind Speed (mph)')
plt.grid(True)

plt.tight_layout()
plt.show()


# #### Conclusion-
# On holidays (hday='Y'), there tends to be slightly higher temperatures and visibility compared to non-holidays (hday='N'). Dew point and wind speed show less variation between holidays and non-holidays.

# # 8. Growth Insights

# •Qa.Which weather conditions are most favorable for Uber pickups, and how can this information be used to optimize driver availability?

# In[63]:


weather_variables = ['temp', 'dewp', 'spd', 'vsb']
correlations = df[weather_variables + ['pickups']].corr()['pickups'].drop('pickups')

print("Correlation coefficients:")
print(correlations)

# Plot the relationships between weather variables and pickups
plt.figure(figsize=(14, 10))

for i, var in enumerate(weather_variables, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=var, y='pickups', data=df, alpha=0.5)
    plt.title(f'{var.capitalize()} vs. Uber Pickups')
    plt.xlabel(var.capitalize())
    plt.ylabel('Number of Pickups')
    plt.grid(True)

plt.tight_layout()
plt.show()


# #### Conclusion-
# Temperature (temp) and visibility (vsb) show the strongest positive correlations with Uber pickups.
# Lower dew point (dewp) and moderate wind speed (spd) also show some positive correlation with pickups.
# This information suggests that drivers may find higher demand during conditions with higher temperatures and better visibility.
# Optimizing driver availability could involve scheduling more drivers during periods of high temperature and good visibility to match increased demand.
# 
# 

# •Qb.Based on the data, what recommendations can be made to Uber to increase pickups during low-demand periods?
# 
# 

# In[64]:


df['hour'] = df['pickup_dt'].dt.hour

# Calculate average number of pickups per hour
hourly_pickups = df.groupby('hour')['pickups'].mean().reset_index()

# Plot hourly pickup trends
plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='pickups', data=hourly_pickups, marker='o', color='b')
plt.title('Average Hourly Uber Pickups')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Number of Pickups')
plt.xticks(range(24))
plt.grid(True)
plt.show()

# Identify low-demand periods
low_demand_hours = hourly_pickups[hourly_pickups['pickups'] == hourly_pickups['pickups'].min()]['hour'].values
print(f"Low-demand hours: {low_demand_hours}")


# #### Conclusion-
# 1. Offer targeted promotions or discounts during identified low-demand hours.
# 2. Implement surge pricing adjustments to incentivize driver availability during low-demand times.
# 3. Optimize driver deployment based on predictive analytics to match supply with demand.
# 
