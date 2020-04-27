#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch


# In[2]:


t = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
print(t)


# In[3]:


print(t.dim())
print(t.shape)
print(t.size())


# In[4]:


print(t[0], t[1], t[-1])
print(t[2:5], t[2:-1])
print(t[:3])


# In[ ]:




