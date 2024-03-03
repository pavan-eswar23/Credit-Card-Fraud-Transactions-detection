#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#loading dataset into pandas data frame


# In[3]:


dt=pd.read_csv('creditcard.csv')


# In[4]:


dt


# In[5]:


dt.describe() #provide statistical analysis of dataset


# In[6]:


dt[dt['Class']==1]


# In[7]:


dt.isna().sum()


# # There are no null values in the dataset

# In[8]:


dt.head(10)


# In[9]:


dt.tail()


# # description of dataset
# #Time- Represents the time in seconds after the first transaction has occured
# 
# #v1,,,,...v28  represents features of each transaction. these details are sensitive so they didn't give more info about these features
# 
# #Amount- represents transcaction amount is us dollars
# 
# #Class- 0-Legit or norml transaction ,1- fraud transaction

# In[11]:


dt.info()


# In[12]:


dt['Class'].value_counts() #Highly unbalanced dataset  more than 99 percent are normal transactions


# In[13]:


normal_tr=dt[dt['Class']==0] #seperating normal and fraud transactions
fraud_tr=dt[dt['Class']==1]
normal_tr.count()


# In[14]:


fraud_tr.count()


# In[15]:


print(normal_tr.shape)
print(fraud_tr.shape)


# In[16]:


normal_tr['Amount'].describe()


# In[17]:


fraud_tr['Amount'].describe()


# In[18]:


dt.groupby('Class').mean()


# #Under Sampling- Build a sample datasset from original dataset which contain similar distribution of fraudelnt and legit transations

# In[19]:


normal_sample=normal_tr.sample(492) #randomly take 492 samples from 284000 samples


# In[20]:


normal_sample


# In[21]:


newdt=pd.concat([normal_sample,fraud_tr],axis=0)


# In[22]:


newdt.shape


# In[23]:


newdt.head()


# In[24]:


newdt.tail()


# In[25]:


newdt.groupby('Class').mean()


# In[26]:


newdt['Class'].value_counts()


# In[27]:


x=newdt.drop('Class',axis=1)
y=newdt['Class']


# In[28]:


scores=[]
for i in range(100):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
    lr=LogisticRegression()
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)
    scores.append(accuracy_score(y_test,y_pred))


# In[29]:


np.argmax(scores)


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)


# In[31]:


x


# In[32]:


y


# In[33]:


x_train


# In[34]:


x_test


# In[35]:


logr=LogisticRegression()


# In[36]:


logr.fit(x_train,y_train)


# In[37]:


y_pred=logr.predict(x_test)


# In[38]:


score=accuracy_score(y_test,y_pred)


# In[39]:


print("Testing Accuracy is :",score)


# In[40]:


train_score=logr.predict(x_train)


# In[41]:


acc2_score=accuracy_score(y_train,train_score)


# In[42]:


print("Training Accuracy is:",acc2_score)


# In[ ]:




