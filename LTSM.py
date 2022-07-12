#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import math
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[ ]:





# In[8]:


df = pd.read_csv('Downloads/PHUN Historical Data.csv')


# In[9]:


df


# In[10]:


plt.plot(df['Price'])
plt.xlabel('Date')
plt.ylabel('close price')


# In[11]:


data = df.filter(['Price'])
dataset = data.values
training_data_len = math.ceil(len(dataset)*.8)
training_data_len


# In[12]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data 


# In[13]:


train_data = scaled_data[0:training_data_len , :]
x_train = []
y_train = []

for i in range (60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
                


# In[14]:


x_train, y_train = np.array(x_train), np.array(y_train)


# In[15]:


x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1] , 1))
x_train.shape


# In[16]:


model = Sequential()
model.add(LSTM(50 , return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50 , return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))


# In[17]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[18]:


model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[19]:


test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    


# In[20]:


x_test = np.array(x_test)


# In[21]:


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[22]:


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[23]:


rmse= np.sqrt(np.mean(predictions - y_test )**2)
rmse


# In[24]:


train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions']= predictions
plt.figure(figsize=(16,8))
plt.title('model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Price'])
plt.plot(valid[['Price', 'Predictions']])           
plt.legend(['Train', 'Val' , 'Predictions'],loc ='lower right')
plt.show()


# In[26]:


quote = pd.read_csv('Downloads/PHUN Historical Data.csv')
new_df = quote.filter(['Price'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days)
X_test = np.array(X_test)   
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
pred_price = model.predict(X_test)
pred_proce = scaler.inverse_transform(pred_price)
print(pred_price)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




