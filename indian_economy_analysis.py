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


sns.relplot(kind="line",y='Life expectancy at birth, total (years)',x='GDP (current US$)',data=data)
plt.title("Relation between Life expectancy and GDP")


# In[8]:


sns.relplot(kind="line",y='Inflation, GDP deflator (annual %)',x='Life expectancy at birth, total (years)',data=data)
plt.title("Relation Between Inflation and life expectancy")


# In[9]:


sns.relplot(kind="line",y='Life expectancy at birth, total (years)',x='Population, total',data=data)
plt.title("Realtion between population and life expectancy")


# In[10]:


sns.relplot(kind="line",y='Life expectancy at birth, total (years)',x='Agriculture, forestry, and fishing, value added (% of GDP)',data=data)
plt.title("Realtion between Agricultural valuation in the country and life expectancy")


# In[11]:


sns.relplot(data=data, kind="line",y='Life expectancy at birth, total (years)',x='Imports of goods and services (% of GDP)')
plt.title("Realtion between Import of goods and life expectancy")


# In[12]:


sns.relplot(kind="line",y='Life expectancy at birth, total (years)',x='Exports of goods and services (% of GDP)',data=data)
plt.title("Realtion between Export of goods and life expectancy")


# In[13]:


sns.relplot(kind="line",y='Industry (including construction), value added (% of GDP)',x='Life expectancy at birth, total (years)',data=data)
plt.title("Realtion between Industrialization and life expectancy")


# In[14]:


sns.relplot(kind="line",y='Foreign direct investment, net inflows (BoP, current US$)',x='Life expectancy at birth, total (years)',data=data)
plt.title("Realtion between International investment in cuntry and life expectancy")


# In[15]:


data.columns


# In[16]:


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


# In[17]:


#for prediction model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[18]:


#splitting training and testing data 
train =np.asarray(data.drop(['Series Name','Military expenditure (% of GDP)', 'Merchandise trade (% of GDP)'],axis=1))
test=np.asarray(data['Life expectancy at birth, total (years)'])
test_2=np.asarray(data['GDP (current US$)'])


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.33, random_state=2)


# In[20]:


rgr = LinearRegression()


# In[21]:


rgr.fit(X_train,y_train)


# In[22]:


pred=rgr.predict(X_test)
X_test


# In[23]:


#checking the predictions 
pred


# In[24]:


rgr.score(X_test,y_test)


# In[25]:


rgr.coef_


# In[26]:


pred


# In[27]:


y_test


# In[28]:


np.mean((pred-y_test)**2)


# In[29]:


plt.plot(pred)
plt.plot(y_test)
plt.show()


# In[30]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
#decision tree regressor for prediction of age expectancy
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
l=regressor.predict(X_test, check_input=True)
l


# In[31]:


#accuracy of the model
regressor.score(X_test,y_test)


# In[36]:


#plot between actual and preicted values
plt.title("Actual and preicted values of life expectancy")
plt.plot(l)
plt.plot(y_test)
plt.show()


# In[33]:


#splitting data and training model for GDP prediction 
train1,test1,train2, test2 = train_test_split(train,test_2,test_size=0.33, random_state=2)
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(train1, train2)
t=regressor.predict(test1, check_input=True)
t


# In[37]:


#plot between actual and predicted GDP
plt.title("Actual and preicted values of GDP")
plt.plot(t)
plt.plot(test2)
plt.show()


# In[35]:


#accuracy of the model 
regressor.score(test1,test2)

