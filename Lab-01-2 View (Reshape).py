#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np


# In[2]:


t = np.array([[[0,1,2],
               [3,4,5]],
              
              [[6,7,8],
               [9,10,11]]])
ft = torch.FloatTensor(t)
print(t)
print(t.shape)


# In[3]:


print(ft.view([-1,3]))


# In[4]:


print(ft.view([-1,3]).shape)


# In[7]:


print(ft.view([-1,1,3]))
print(ft.view([-1,1,3]).shape)


# In[ ]:




