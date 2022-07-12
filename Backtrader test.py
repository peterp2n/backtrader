#!/usr/bin/env python
# coding: utf-8

# In[1]:


import backtrader as bt
import pandas as pd
import yfinance as yf
import matplotlib
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class MyStrategy(bt.Strategy):
  pass


# In[3]:


cerebro = bt.Cerebro()


# In[4]:


df = yf.download('PHUN', start ='2021-07-07')
df


# In[5]:


data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)


# In[6]:


cerebro.run()

cerebro.addstrategy(MyStrategy)
# In[7]:


cerebro.plot()


# In[ ]:




