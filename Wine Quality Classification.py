#!/usr/bin/env python
# coding: utf-8

# ## LIBRARY

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


#load data
train = pd.read_csv('train.csv')
train.head(5)


# In[3]:


test = pd.read_csv('test.csv')
test.head(5)


# ## Data Analysis

# In[4]:


# Correlation between features
plt.figure(figsize=(12,12))
sns.heatmap(data = train.corr(), annot=True)


# In[22]:


#distribution of each features
plt.figure(figsize=(12,12))
for i in range(1,13):
    plt.subplot(3,4,i)
    sns.distplot(train.iloc[:,i])
plt.tight_layout()
plt.show()


# In[23]:


#distribution by 'quality'
numerical_columns = train.select_dtypes(exclude='object').columns.tolist()
numerical_columns.remove('quality')
def show_dist_plot(df, columns):
    for column in columns:
        f, ax = plt.subplots(1,2,figsize=(16,4))
        sns.stripplot(x=df['quality'],y=df[column], ax=ax[0],hue=df['quality'])
        sns.violinplot(data=df, x='quality', y=column, ax=ax[1])
        
show_dist_plot(train, numerical_columns)


# ## Preprocessing

# In[6]:


train.info()


# In[18]:


train.describe(include='all')


# In[7]:


#change object to numeric type ; red = 0 white = 1
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
enc.fit(train['type'])
train['type'] = enc.transform(train['type'])
test['type'] = enc.transform(test['type'])


# In[8]:


train.head()


# In[9]:


train_x = train.drop(['index','quality'],axis=1)
train_y = train['quality']

test_x = test.drop('index', axis = 1)


# ## Modeling

# In[11]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#train the model
model.fit(train_x, train_y)


# In[12]:


y_pred = model.predict(test_x)


# ## Submission

# In[16]:


submission = pd.read_csv('submission.csv')
submission['quality'] = y_pred
submission


# In[17]:


submission.to_csv('baseline.csv', index=False)

