#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch


# In[2]:


t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)


# In[3]:


print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)


# In[4]:


print('t[0] t[1] t[-1] : ', t[0], t[1], t[-1])
print('t[2:5] t[4:-1] t[:2] : ', t[2:5], t[4:-1], t[:2])


# In[ ]:




