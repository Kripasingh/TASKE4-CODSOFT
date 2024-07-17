#!/usr/bin/env python
# coding: utf-8

# # TASK-4: CREDIT CARD FRAUD DETECTION

# IMPORTING LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# IMPORTING DATASETS

# In[2]:


df = pd.read_csv("creditcard.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# Exploration on the datasets

# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.dtypes


# In[9]:


df.columns


# In[10]:


df.info()


# Fraud and Genuune Cases

# In[11]:


fraud_cases = df[df['Class']==1]
print('Number of Fraud Cases:' , fraud_cases)


# In[12]:


non_fraud_cases = df[df['Class']==0]
print('Number of Non Fraud Cases:' , non_fraud_cases)


# In[13]:


fraud = df[df['Class']==1]
genuine = df[df['Class']==0]


# In[14]:


fraud.Amount.describe()


# Exploratory Data Analysis

# In[15]:


df.hist(figsize=(10,10),color='lime')
plt.show()


# In[16]:


from pylab import rcParams
import warnings
warnings.filterwarnings('ignore')


# In[17]:


rcParams['figure.figsize'] = 16,8
f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
f.suptitle(' Time of transaction  vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(genuine.Time , genuine.Amount)
ax2.set_title('Genuine')
plt.xlabel('Time(in Second)')
plt.ylabel('Amount')
plt.show()


# Correlation

# In[18]:


plt.figure(figsize=(10,8))
corr=df.corr()
sns.heatmap(corr,cmap='BuGn')


# In[19]:


corr = df.corr()
sns.heatmap(corr, annot = True)


# In[20]:


sns.relplot(data = df , x= 'Amount' , y="Time" , hue = 'Class')


# Model Building

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X = df.drop(['Class'], axis=1)
y = df.Class


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=123)


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


rfc=RandomForestClassifier()


# In[26]:


Model=rfc.fit(X_train,y_train)


# In[27]:


prediction = Model.predict(X_test)


# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


accuracy_score(y_test,prediction)


# Model 2

# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


X1 = df.drop(['Class'], axis=1)
y1 = df.Class


# In[32]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.30,random_state=123)


# In[33]:


lr=LogisticRegression()


# In[34]:


model2 = lr.fit(X1_train,y1_train)


# In[35]:


print(model2)


# In[36]:


prediction = model2.predict(X1_test)


# In[37]:


accuracy_score(y1_test,prediction)


# Model 3

# In[38]:


from sklearn.tree import DecisionTreeRegressor


# In[39]:


X2 = df.drop(['Class'], axis=1)
y2 = df.Class


# In[40]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size=0.30,random_state=123)


# In[41]:


dt=DecisionTreeRegressor()


# In[42]:


model3 = dt.fit(X2_train,y2_train)


# In[43]:


print(model3)


# In[44]:


prediction = model3.predict(X2_test)


# In[45]:


accuracy_score(y2_test,prediction)

