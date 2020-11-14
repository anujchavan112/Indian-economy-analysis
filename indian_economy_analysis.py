#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all dependancies 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np 
#importing csv dataset file 
data=pd.read_csv(r'C:\Users\Anuj\Desktop\indecodata.csv')


# In[2]:


#checking the data
data.head()


# In[3]:


#checking all datatypes 
data.dtypes


# In[4]:


data


# In[5]:


#importing seaborn for data plotting 
import seaborn as sns
#relational plotting
sns.relplot(kind='line',y='Population growth (annual %)',x='GDP (current US$)',data=data)
plt.title("Realation between population growth and GDP")


# In[6]:


#checking what columns are ther in dataset 
data.columns


# In[7]:


sns.relplot(kind="line",y='Inflation, GDP deflator (annual %)',x='GDP (current US$)',data=data)
plt.title("Relation Between Inflation and GDP")
sns.relplot(kind="line",y='GDP (current US$)',x='Population, total',data=data)
plt.title("Realtion between population and GDP")
sns.relplot(kind="line",y='GDP (current US$)',x='Agriculture, forestry, and fishing, value added (% of GDP)',data=data)
plt.title("Realtion between Agricultural valuation in the country and GDP")
sns.relplot(data=data, kind="line",y='Imports of goods and services (% of GDP)',x='GDP (current US$)')
plt.title("Realtion between Import of goods and GDP")
sns.relplot(kind="line",y='Industry (including construction), value added (% of GDP)',x='GDP (current US$)',data=data)
plt.title("Realtion between Industrialization and GDP")
sns.relplot(kind="line",y='Foreign direct investment, net inflows (BoP, current US$)',x='GDP (current US$)',data=data)
plt.title("Realtion between International investment in cuntry and GDP")


# In[8]:


data.columns


# In[9]:


#analyzing data from 1990 to 2019 for all columns 
plt.subplots(figsize=(20,3))
plt.plot(data['Series Name'],data['GDP (current US$)'], '*-')
plt.title("GDP growth")
plt.subplots(figsize=(20,3))
plt.plot(data['Series Name'],data['Life expectancy at birth, total (years)'], '*-')
plt.title("Life expectancy growth")
plt.subplots(figsize=(20,3))
plt.plot(data['Series Name'],data['Population growth (annual %)'], '*-')
plt.title("Population growth %")
plt.subplots(figsize=(20,3))
plt.plot(data['Series Name'],data['Population, total'], '*-')
plt.title("Population growth")
plt.subplots(figsize=(20,3))
plt.plot(data['Series Name'],data['Foreign direct investment, net inflows (BoP, current US$)'], '*-')
plt.title("Foreign investment growth")
plt.subplots(figsize=(20,3))
plt.plot(data['Series Name'],data['Inflation, GDP deflator (annual %)'], '*-')
plt.title("Infletion rate per year")
plt.subplots(figsize=(20,3))
plt.plot(data['Series Name'],data['Industry (including construction), value added (% of GDP)'], '*-')
plt.title("Industrial Growth")
plt.subplots(figsize=(20,3))
plt.plot(data['Series Name'],data['Agriculture, forestry, and fishing, value added (% of GDP)'], '*-')
plt.title("Agricultural Growth")
plt.subplots(figsize=(20,3))
plt.plot(data['Series Name'],data['Exports of goods and services (% of GDP)'], '*-')
plt.title("Export per year")
plt.subplots(figsize=(20,3))
plt.plot(data['Series Name'],data['Imports of goods and services (% of GDP)'], '*-')
plt.title("Import per year")


# In[10]:


#for prediction model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[11]:


#splitting training and testing data 
train =np.asarray(data.drop(['Series Name','Military expenditure (% of GDP)', 'Merchandise trade (% of GDP)'],axis=1))
test=np.asarray(data['Life expectancy at birth, total (years)'])
test_2=np.asarray(data['GDP (current US$)'])


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.33, random_state=2)


# In[13]:


rgr = LinearRegression()
rgr.fit(X_train,y_train)
rgr.score(X_test,y_test)


# In[14]:


pred=rgr.predict(X_test)
X_test


# In[15]:


#checking the predictions 
pred


# In[16]:


y_test


# In[17]:


np.mean((pred-y_test)**2)


# In[18]:


plt.plot(pred)
plt.plot(y_test)
plt.show()


# In[19]:


from sklearn.tree import DecisionTreeRegressor
#decision tree regressor for prediction of age expectancy
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
l=regressor.predict(X_test, check_input=True)
l


# In[20]:


#accuracy of the model
regressor.score(X_test,y_test)


# In[21]:


np.mean((l-y_test)**2)


# In[22]:


#plot between actual and preicted values
plt.title("Actual and preicted values of life expectancy")
plt.plot(l)
plt.plot(y_test)
plt.show()


# In[23]:


#splitting data and training model for GDP prediction 
train1,test1,train2, test2 = train_test_split(train,test_2,test_size=0.33, random_state=2)
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(train1, train2)
t=regressor.predict(test1, check_input=True)
t


# In[24]:


np.mean((t-test2)**2)


# In[25]:


#plot between actual and predicted GDP
plt.title("Actual and preicted values of GDP")
plt.plot(t)
plt.plot(test2)
plt.show()


# In[26]:


#accuracy of the model 
regressor.score(test1,test2)

