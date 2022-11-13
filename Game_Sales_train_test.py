#!/usr/bin/env python
# coding: utf-8

# In[1]:


#As a data scientist of a leading decision analysis firm you are required to predict the potential
#global user of the game based on the data provided by the customer so that they can plan their global launch.

#Name - Game name
#Platform - Running platform
#Year of release
#Genre - Game Genre
#Publisher
#Critic_score - Aggregate score compiled by Metacritic staff
#Criticcount - The number of critics used in coming up with the Critic Score
#User_score - Score by Metacritic’s subscribers
#Usercount - Number of users who gave the user score
#Developer - Party responsible for creating the game
#Rating - The ESRB ratings
#NA_Sales - Sales in North America (in millions of units)
#EU_Sales - Sales in the European Union (in millions of units)
#JP_Sales - Sales in Japan (in millions of units)
#Global_Sales - Total sales in the world (in millions of units)


# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing the "Train Data"

# In[3]:


df_train=pd.read_csv('C:/Users/MAVERICK/OneDrive/Desktop/Hack Game Sales/Train.csv')
pd.options.display.max_columns=100


# In[4]:


df_train


# In[5]:


df_train.describe()


# In[6]:


df_train.info()


# In[7]:


# Calculating the Percent of Null values present in each columns
null_val=df_train.isnull().sum()
null_val=pd.DataFrame(null_val)
null_val['percent']=round((null_val/df_train.shape[0])*100,2)
null_val.sort_values('percent',ascending=False)


# ### Treating Null values in 'Name' Columns

# In[8]:


# Treating Null value with 'Mode'
df_train['Name']=df_train['Name'].fillna(df_train.Name.dropna().mode().values[0])


# In[9]:


df_train.Name.isnull().sum()


# In[10]:


df_train.Name.value_counts()


# In[11]:


# checking all the Game Names
for i in df_train.Name:
    print(i)


# ### Treating Null values in 'Publisher'

# In[12]:


df_train.Publisher.value_counts()


# In[13]:


df_train.Publisher.mode()


# In[14]:


df_train['Publisher']=df_train['Publisher'].fillna(df_train['Publisher'].dropna().mode().values[0])


# In[15]:


df_train['Publisher'].isnull().sum()


# In[16]:


df_train.Publisher.unique()


# ### Treating Null values in 'Genre'

# In[17]:


fig=plt.figure(figsize=(20,10))
df_train[('Genre')].value_counts().plot(kind='bar')
plt.title('checking for the total value_counts in individual Type of Games')


# In[18]:


# Checking for exact Null value Index Position
df_train[df_train.Genre.isnull()]


# In[19]:


df_train.Genre.mode()


# In[20]:


# Filling the Null values by Mode
df_train['Genre']=df_train.Genre.fillna(df_train.Genre.dropna().mode().values[0])


# ### Treating null values in 'Year_of_Release'

# In[21]:


fig=plt.figure(figsize=(20,10))
df_train[('Year_of_Release')].value_counts().plot(kind='bar')
plt.title('Number of games released in Specific Year')


# In[22]:


df_train.Year_of_Release.value_counts()


# In[23]:


df_yr=df_train.pivot_table(values='Year_of_Release',index='Genre',aggfunc=np.median)

bool_yr=df_train.Year_of_Release.isnull()

## treating the null for Year_of_Release according to Genre ,so that according to game type the Year_of_Release will fill NaN
df_train.loc[bool_yr,'Year_of_Release']=df_train.loc[bool_yr,'Genre'].apply(lambda x: df_yr.loc[x])


# In[24]:


df_train.Year_of_Release.unique()


# In[25]:


df_train.Year_of_Release.isnull().sum()


# ### Treating null_values in 'Critic_Score'

# In[ ]:





# In[26]:


df_train.Critic_Score.dtypes


# In[27]:


df_train.Critic_Score.unique()


# In[28]:


df_cs=df_train.pivot_table(values='Critic_Score',index='Genre',aggfunc=np.median)

bool_cs=df_train.Critic_Score.isnull()

## treating the null for Critic_Score according to Genre ,so that according to game type the Critic_Score will fill NaN.
df_train.loc[bool_cs,'Critic_Score']=df_train.loc[bool_cs,'Genre'].apply(lambda x: df_cs.loc[x])


# In[29]:


df_train.Critic_Score.unique()


# In[30]:


df_train.Critic_Score.isnull().sum()


# ### Treating null_values in 'Critic_Count'

# In[31]:


df_train.Critic_Count.dtypes


# In[32]:


df_train.isnull().sum()


# In[33]:


df_train.Critic_Count.unique()


# In[34]:


df_cc=df_train.pivot_table(values='Critic_Count',index='Genre',aggfunc=np.median)

bool_cc=df_train.Critic_Count.isnull()

## treating the null for Critic_Count according to Genre ,so that according to game type the Critic_Count will fill NaN.
df_train.loc[bool_cc,'Critic_Count']=df_train.loc[bool_cc,'Genre'].apply(lambda x: df_cc.loc[x])


# In[35]:


df_train.Critic_Count.isnull().sum()


# ### Treating null_values in 'User_Score'

# In[36]:


df_train.User_Score.unique()


# as we can see that there is a extra unique value i.e, 'tbd' hence, we can replace it with 'NaN'

# In[37]:


df_train['User_Score']=df_train['User_Score'].replace('tbd',np.nan)


# In[38]:


df_train.User_Score.isnull().sum()


# here as we can see the null value is almost more than 50% ,So here we can do Two approaches that to remove it or to treat the null values.

# In[39]:


## converting the dtype of User_Score column from 'object to 'float64'
df_train['User_Score']=df_train['User_Score'].astype('float64')
df_train.dtypes


# In[40]:


df_us=df_train.pivot_table(values='User_Score',index='Genre',aggfunc=np.median)

bool_us=df_train.User_Score.isnull()

## treating the null for 'User_Score' according to 'Genre' ,so that according to game type the 'User_Score' will fill 'NaN'.
df_train.loc[bool_us,'User_Score']=df_train.loc[bool_us,'Genre'].apply(lambda x: df_us.loc[x])


# In[41]:


df_train.User_Score.isnull().sum()


# ### Treating null_values in 'User_Count'

# In[42]:


df_train.User_Count.value_counts()


# In[43]:


df_uc=df_train.pivot_table(values='User_Count',index='Genre',aggfunc=np.median)
df_uc
bool_uc=df_train.User_Count.isnull()
bool_uc
## treating the null for 'User_Count' according to 'Genre' ,so that according to game type the 'User_Count' will fill 'NaN'.
df_train.loc[bool_uc,'User_Count']=df_train.loc[bool_uc,'Genre'].apply(lambda x: df_uc.loc[x])


# In[44]:


df_train.User_Count.isnull().sum()


# ### Treating null_values in 'Rating'

# In[45]:


fig=plt.figure(figsize=(10,5))
df_train[('Rating')].value_counts().plot(kind='bar')
plt.title('The Game ratings for restricted Age Groups VS total counts of games')


# In[46]:


df_train.Rating.value_counts()


# In[47]:


df_train[['Rating']] = df_train[['Rating']].replace(dict.fromkeys(['K-A'], 'KA'))
df_train[['Rating']] = df_train[['Rating']].replace(dict.fromkeys(['E10+'], 'E10'))
df_train[['Rating']] = df_train[['Rating']].replace(dict.fromkeys(['RP'], np.nan))


# def replace_rating(col,df):
#     for i in df[col].unique():
#         if i=='K-A':
#             df[col].replace(dict.fromkeys([i], 'KA'))
#         if i=='E10+':
#             df[col].replace(dict.fromkeys([i], 'E10'))
#         if i=='RP':
#             df[col].replace(dict.fromkeys([i], np.nan))
#         else:
#             return i
# print(replace_rating('Rating',df_train))
#             
#         
#             

# In[48]:


#In above coding we have combined the rating of Age-type grouping of people playing the game that are 
#'E'=E(Everyone)
#'EC','E10+','T'= (Early Childhood,Everyone 10+ and Above,Teen)
#'M','AO'= (Mature and Adults only)
#'RP'= nan(Rating Pending)
df_train.Rating.unique()


# In[49]:


from scipy.stats import mode


# In[50]:


df_rate=df_train.pivot_table(values='Rating',index='Genre',aggfunc=(lambda x:mode(x).mode[0]))

bool_rate=df_train['Rating'].isnull()

df_train.loc[bool_rate,'Rating']=df_train.loc[bool_rate,'Genre'].apply(lambda x:df_rate.loc[x])


# In[51]:


df_train.Rating.value_counts()


# ### Treating null_values in 'Developer'

# In[52]:


df_train.Developer.value_counts()


# In[53]:


# Treating all the null values of 'Developer' wrt 'Genre'
df_dev=df_train.pivot_table(values='Developer',index=['Genre'],aggfunc=(lambda x:mode(x).mode[0]))
bool_dev=df_train['Developer'].isnull()
df_train.loc[bool_dev,'Developer']=df_train.loc[bool_dev,'Genre'].apply(lambda x:df_dev.loc[x])


# In[54]:


df_train.isnull().sum()


# In[55]:


fig=plt.figure(figsize=(10,5))
sns.heatmap(df_train.corr(),annot=True)


# In[ ]:





# In[56]:


df_train.shape


# In[57]:


df_train.dtypes


# In[58]:


df_train.Platform.value_counts()


# ### Combining similar values in the column to single unique feature in 'Platform' column

# In[59]:


df_train.Platform.unique()


# Combining similar unique values in the column to single unique feature

# In[60]:


df_train['Platform']=df_train['Platform'].replace(['PSP','PS','PS4','PS2','PSP','PS3','PSV'],'Playstation')
df_train['Platform']=df_train['Platform'].replace(['NES','N64','DS','3DS','3DO','GB','GC','GBA','Wii','3DO','SNES','WiiU'],'Nintendo')
df_train['Platform']=df_train['Platform'].replace(['XB','XOne','X360'],'Xbox')
df_train['Platform']=df_train['Platform'].replace(['PC','PCFX','GEN','TG16'],'PersonalComputer')
df_train['Platform']=df_train['Platform'].replace(['2600','DC','SAT','SCD','WS','NG','GG'],'Others')


# platform=['Wii', 'NES', 'GB', 'DS', 'X360', 'PS3', 'PS2', 'SNES', 'GBA',
#        'PS4', '3DS', 'N64', 'PS', 'XB', 'PC', '2600', 'PSP', 'XOne',
#        'WiiU', 'GC', 'GEN', 'DC', 'PSV', 'SAT', 'SCD', 'WS', 'NG', 'TG16',
#        '3DO', 'GG', 'PCFX']
# Playstation=['PS2','PS3','PSP','PSV','PS4','PS']
# XBox=['XB','X360','XOne']
# Nintendo=['NES','N64','SNES']
# 

# In[61]:


df_train['Platform'].value_counts()


# ### Checking if any Columns can be dropped or not

# In[62]:


df_train['Publisher'].value_counts()


# #### since from the above we an see that there are total of 552 Unique values in 'Publisher' column,So we can drop the column for better model building.

# ## dropped 'Publisher'

# In[63]:


df_train.drop('Publisher',axis=1,inplace=True)## 


# In[64]:


df_train['Developer'].value_counts()


# #### As we can see that there are totally of 1577 Unique values in 'Developer' column,So we can drop the column for better model building.

# ## dropped 'Developer'

# In[65]:


df_train.drop('Developer',axis=1,inplace=True)## dropped Developer


# In[66]:


df_train.info()


# In[67]:


df_train.Rating.value_counts()


# #### So from the above we can combine many unique values as they have the same meaning and Age-types

# In[68]:


df_train['Rating']=df_train['Rating'].replace(['AO','KA','EC'],['M','E','E'])


# In[69]:


df_train.Rating.value_counts()


# In[70]:


df_train.Name.value_counts()


# #### As we can see that there are totally of 10327 Unique values in 'Name' column,So we can drop the column for better model building.

# ## Dropped 'Name' column

# In[71]:


df_train.drop('Name',axis=1,inplace=True)


# In[72]:


df_train


# ## Treating the Outliers

# In[73]:


fig=plt.figure(figsize=(20,10))
df_train.boxplot()


# so from the above we can see that User_Count column has the most number of outliers

# In[74]:


df_train.drop('User_Count',axis=1,inplace=True)


# In[75]:


#fig=plt.figure(figsize=(20,10))
#df_train.boxplot()


# As we can see from above plot that after dropping the User_Count column we are having lesser outliers compare to before one.

# In[76]:


df_train.describe()


# In[77]:


Q1=df_train.quantile(.25)
Q3=df_train.quantile(.75)
IQR=Q3-Q1
# To get the outliers if its outside the interval then can be removed
L=Q1-1.5*IQR # Lower limit
H=Q3+1.5*IQR # upper limit
print(L)
print(H)


# In[78]:


#  To remove the outliers if its the interval ,then can be removed
df_train=df_train[~((df_train < (L) ) |  (df_train > (H) )  ).any(axis=1)]


# In[79]:


df_train.describe()


# In[80]:


df_train.shape


# ### LabelEncoding object data with less unique features

# In[81]:


# importing Label encoder
from sklearn.preprocessing import LabelEncoder


# In[82]:


## LabelEncoding can be done here Since the number of columns are less

le=LabelEncoder()
df_train['Rating']=le.fit_transform(df_train['Rating'])
df_train['Rating'].value_counts()


# In[83]:


df_train


# In[84]:


df_train.columns


# In[85]:


# Creating a copy of the data before One Hot Encoder
df_train1=df_train.copy(deep=True)


# In[ ]:





# In[86]:


fig = plt.figure(figsize=(15,15))
ax= fig.gca()
df_train[['Platform', 'Year_of_Release', 'Genre', 'NA_Sales', 'EU_Sales',
       'JP_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'Rating',
       'Global_Sales']].hist(ax = ax)
plt.show()


# ### One Hot encoding object data with Unique Features

# In[87]:


# Creating Dummies for Categorical data 
df_train1=pd.get_dummies(df_train)
df_train1


# In[88]:


fig=plt.figure(figsize=(20,10))
sns.heatmap(df_train1.corr(),annot=True)


# ### Copying the old test data set to a new one as 'df_1'

# In[89]:


df_1=df_train1.copy(deep=True)


# since all the data has been converted to Numerical data ,now we can further approach for model building.

# # [Model Building] 

# ### Linear Regression

# In[90]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import r2_score,mean_squared_error


# In[91]:


x=df_1.drop('Global_Sales',axis=1)
y=df_1['Global_Sales']


# In[92]:


# considering 100% Data as Train Data

lr=LinearRegression()

# Fitting the model using x & y
lr.fit(x,y)

# predicting the data
train_pred=lr.predict(x)


# checking the RMSE 
print('RMSE for train:',np.sqrt(mean_squared_error(y,train_pred)))

print('r2_score for train:',r2_score(y,train_pred))


#plt.scatter(y_test,test_pred)


# ### MinMaxScaling the data 

# sc=StandardScaler()
# x=df_1.drop('Global_Sales',axis=1)
# x_sc=sc.fit_transform(x)
# x_sc=pd.DataFrame(x_sc)
# x_sc

# x=x_sc
# y=df_1['Global_Sales']

# #x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
# 
# lr=LinearRegression()
# 
# lr.fit(x,y)
# 
# train_pred=lr.predict(x)
# 
# #test_pred=lr.predict(x_test)
# 
# 
# print('RMSE for train:',np.sqrt(mean_squared_error(y,train_pred)))
# 
# print('r2_score for train:',r2_score(y,train_pred))

# ### Random Forest

# In[93]:


from sklearn.ensemble import RandomForestRegressor

x=df_1.drop('Global_Sales',axis=1)
y=df_1['Global_Sales']

RFC=RandomForestRegressor(n_estimators=50,n_jobs=-1,oob_score=True,random_state=42)

RFC.fit(x,y)

train_pred=RFC.predict(x)
#test_pred=RFC.predict(x_test)

print('RMSE for train:',np.sqrt(mean_squared_error(y,train_pred)))


print('r2_score for train:',r2_score(y,train_pred))


# ### GridSearchCV

# In[160]:


from sklearn.model_selection import GridSearchCV

x=df_1.drop('Global_Sales',axis=1)
y=df_1['Global_Sales']

param={'max_depth':[5,10,15,20,30],'min_samples_leaf':[10,15,20,50,100]}

gs=GridSearchCV(estimator=RFC,param_grid=param,cv=4,n_jobs=-1,scoring='r2')

gs.fit(x,y)

gs.best_estimator_
gs_best=gs.best_estimator_

gs_best.fit(x,y)

train_pred=gs_best.predict(x)

#test_pred=gs_best.predict(x_test)


print('RMSE for train:',np.sqrt(mean_squared_error(y,train_pred)))
#print('RMSE for train:',np.sqrt(mean_squared_error(y,train_pred)))


print('r2_score for train:',r2_score(y,train_pred))
#print('r2_score for test:',r2_score(y_test,test_pred))


# ### Decision Tree

# In[94]:


from sklearn.tree import DecisionTreeRegressor


# In[95]:


# considering 100% Data as Train Data

dt=DecisionTreeRegressor()

# Fitting the model using x & y
dt.fit(x,y)

# predicting the data
train_pred=dt.predict(x)


# checking the RMSE 
print('RMSE for train:',np.sqrt(mean_squared_error(y,train_pred)))

print('r2_score for train:',r2_score(y,train_pred))


#plt.scatter(y_test,test_pred)


# In[159]:


global score_card
    score_card = score_card.append({'Model': 'Linear Regression',
                                        'R2 Score': r2_score(y,train_pred),
                                        'Root Mean Square Error':np.sqrt(mean_squared_error(y,train_pred))
                                        }, 
                                        ignore_index = True)


            display(score_card)


# In[ ]:





# ## Importing the "Test data"

# In[96]:


df_test=pd.read_csv('C:/Users/MAVERICK/OneDrive/Desktop/Hack Game Sales/Test.csv')
pd.options.display.max_columns=14


# In[97]:


df_test


# In[98]:


df_test.describe()


# In[99]:


df_test.isnull().sum()


# ### Treating nul value in Year_of_Release

# In[100]:


df_test.Year_of_Release.value_counts()


# In[101]:


df_test_yr=df_test.pivot_table(values='Year_of_Release',index='Genre',aggfunc=np.median)

bool_test_yr=df_test.Year_of_Release.isnull()

## treating the null for 'Year_of_Release' according to 'Genre' ,so that according to game type the 'Year_of_Release' will fill 'NaN'.
df_test.loc[bool_test_yr,'Year_of_Release']=df_test.loc[bool_test_yr,'Genre'].apply(lambda x: df_test_yr.loc[x])


# In[102]:


df_test.Year_of_Release.isnull().sum()


# ### Dropping  Publisher,Name,Developer columns

# In[103]:


df_test.drop(['Publisher','Name','Developer'],axis=1,inplace=True)


# dropped the Publisher,Name,Developer columns since has been dropped in train data due to too much unique values that can'nt be comverted to Numeric data for better model building

# In[104]:


df_test.dtypes


# ### Treating null value in Critic_Score

# In[105]:


df_test_cs=df_test.pivot_table(values='Critic_Score',index='Genre',aggfunc=np.median)

bool_test_cs=df_test.Critic_Score.isnull()

## treating the null for 'Critic_Score' according to 'Genre' ,so that according to game type the 'Critic_Score' will fill 'NaN'.
df_test.loc[bool_test_cs,'Critic_Score']=df_test.loc[bool_test_cs,'Genre'].apply(lambda x: df_test_cs.loc[x])


# In[106]:


df_test.Critic_Score.isnull().sum()


# ### Treating null value in Critic_Count

# In[107]:


df_test_cc=df_test.pivot_table(values='Critic_Count',index='Genre',aggfunc=np.median)

bool_test_cc=df_test.Critic_Count.isnull()

## treating the null for 'Critic_Count' according to 'Genre' ,so that according to game type the 'Critic_Count' will fill 'NaN'.
df_test.loc[bool_test_cc,'Critic_Count']=df_test.loc[bool_test_cc,'Genre'].apply(lambda x: df_test_cc.loc[x])


# In[108]:


df_test.Critic_Count.isnull().sum()


# ### Treating null value in User_Score

# In[109]:


df_test.User_Score.dtypes


# In[110]:


## replacing 'tbd' unique value to 'NaN'
df_test['User_Score']=df_test['User_Score'].replace('tbd',np.nan)

## converting the datatype of User_Score column from 'object' to 'Float64'
df_test['User_Score']=df_test['User_Score'].astype('float64')

## treating the null values according to type of games
df_test_us=df_test.pivot_table(values='User_Score',index='Genre',aggfunc=np.median)

bool_test_us=df_test.User_Score.isnull()

df_test.loc[bool_test_us,'User_Score']=df_test.loc[bool_test_us,'Genre'].apply(lambda x: df_test_us.loc[x])


# In[111]:


df_test.User_Score.isnull().sum()


# ### Treating null value in User_Count

# In[112]:


df_test.User_Count.dtypes


# In[113]:


## treating the null values according to type of games
df_test_uc=df_test.pivot_table(values='User_Count',index='Genre',aggfunc=np.median)

bool_test_uc=df_test.User_Count.isnull()

df_test.loc[bool_test_uc,'User_Count']=df_test.loc[bool_test_uc,'Genre'].apply(lambda x: df_test_uc.loc[x])


# In[114]:


df_test.User_Count.isnull().sum()


# ### Treating null value in Rating

# In[115]:


df_test.Rating.value_counts()


# In[116]:


df_test[['Rating']] = df_test[['Rating']].replace(dict.fromkeys(['E10+'], 'E10'))


# In[117]:


df_test.Rating.value_counts()


# In[118]:


df_test_rate=df_test.pivot_table(values='Rating',index='Genre',aggfunc=(lambda x:mode(x).mode[0]))

bool_test_rate=df_test['Rating'].isnull()

df_test.loc[bool_test_rate,'Rating']=df_test.loc[bool_test_rate,'Genre'].apply(lambda x:df_test_rate.loc[x])


# In[119]:


df_train['Rating']=df_train['Rating'].replace(['AO','KA','EC'],['M','E','E'])


# In[120]:


df_test.Rating.value_counts()


# ### Manually Label Encoding

# In[121]:


df_test['Rating']=df_test['Rating'].replace(['E','T','M','E10','EC'],[0,3,2,1,4])


# In[122]:


df_test.Rating.value_counts()


# In[ ]:





# In[123]:


df_test.isnull().sum()


# ### combining important features in Platform Column

# In[124]:


df_test.Platform.value_counts()


# In[125]:


df_test['Platform']=df_test['Platform'].replace(['PSP','PS','PS4','PS2','PSP','PS3','PSV'],'Playstation')
df_test['Platform']=df_test['Platform'].replace(['NES','N64','DS','3DS','3DO','GB','GC','GBA','Wii','3DO','SNES','WiiU'],'Nintendo')
df_test['Platform']=df_test['Platform'].replace(['XB','XOne','X360'],'Xbox')
df_test['Platform']=df_test['Platform'].replace(['PC','PCFX','GEN','TG16'],'PersonalComputer')
df_test['Platform']=df_test['Platform'].replace(['2600','DC','SAT','SCD','WS','NG','GG'],'Others')


# In[126]:


df_test.Platform.value_counts()


# In[127]:


df_test


# In[128]:


df_test


# ### Treating the outliers

# In[129]:


fig=(plt.figure(figsize=(20,10)))
df_test.boxplot()


# #### as from the above we can see that there are a lot of ouliers in User_Score column so better to drop it

# In[130]:


df_test.drop('User_Score',axis=1,inplace=True)


# In[131]:


Q1=df_test.quantile(.25)
Q3=df_test.quantile(.75)
IQR=Q3-Q1
# To get the outliers if its outside the interval then can be removed
L=Q1-1.5*IQR # Lower limit
H=Q3+1.5*IQR # upper limit
print(L)
print(H)


# In[132]:


df_test=df_test[~((df_test < (L) ) |  (df_test > (H) )  ).any(axis=1)]


# In[133]:


fig=(plt.figure(figsize=(20,10)))
df_test.boxplot()


# In[134]:


df_test.describe()


# In[161]:


df_test.shape


# #### Creating a Copy of df_test as df_test1

# In[135]:


df_test1=df_test.copy(deep=True)


# In[136]:


df_test1


# ### One Hot Encoding the categorical data

# In[137]:


df_2=pd.get_dummies(df_test1)


# In[138]:


df_2


# In[139]:


fig=plt.figure(figsize=(20,10))
sns.heatmap(df_2.corr(),annot=True)


# # Model Building

# In[148]:


x=df_1.drop('Global_Sales',axis=1)
y=df_1['Global_Sales']

x_test=df_2


# In[149]:


# considering 100% Data as Train Data

lr=LinearRegression()

# Fitting the model using x & y
lr.fit(x,y)

# predicting the data
train_pred=lr.predict(x)
test_pred=lr.predict(x_test)


# checking the RMSE 
print('RMSE for train:',np.sqrt(mean_squared_error(y,train_pred)))


print('r2_score for train:',r2_score(y,train_pred))


#plt.scatter(y_test,test_pred)


# In[151]:


test_pred


# In[153]:


df_2.shape


# In[154]:


test_pred=pd.DataFrame(test_pred)


# In[155]:


test_pred


# In[ ]:




