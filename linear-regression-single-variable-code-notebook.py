#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np


# In[118]:


df = pd.read_csv("homeprices.csv")


# In[119]:


df.info()


# In[120]:


df


# In[121]:


plt.xlabel("area (square ft)")
plt.ylabel("price (US$)")
plt.scatter(df.area,df.price,color="red",marker="+")


# In[122]:


model = linear_model.LinearRegression()


# In[123]:


type(model)


# In[124]:


df[["area"]].values.ndim


# In[125]:


df[["price"]].values.ndim


# In[126]:


model.fit(df[["area"]].values, df[["price"]].values)


# In[127]:


model.coef_


# In[128]:


model.intercept_


# In[129]:


type(np.array([[3300]]))


# In[130]:


np.array([[3300]]).ndim


# In[131]:


np.array([[3300]]).shape


# In[132]:


model.predict(np.array([[3300]]))


# In[133]:


# y=mx+b
(135.78767123*3300)+180616.43835616


# In[134]:


areas_dataframe = pd.read_csv("areas.csv")


# In[135]:


areas_dataframe.head()


# In[136]:


type(areas_dataframe.values)


# In[137]:


prices = model.predict(areas_dataframe.values)


# In[138]:


type(prices)


# In[139]:


areas_dataframe["price"] = prices


# In[140]:


areas_dataframe.head()


# In[141]:


areas_dataframe.to_csv("predictions.csv", index=False)


# In[142]:


df.head()


# In[148]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area (square ft)")
plt.ylabel("price (US$)")
plt.scatter(df.area,df.price,color="red",marker="+")
plt.plot(df[["area"]].values,model.predict(df[["area"]].values))


# In[ ]:




