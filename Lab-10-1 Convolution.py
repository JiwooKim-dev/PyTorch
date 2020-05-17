#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[2]:


input = torch.Tensor(1, 1, 28, 28)


# In[3]:


conv1 = nn.Conv2d(1, 5, 5)  # output size = 5 X 24
pool = nn.MaxPool2d(2)      # output size = 5 X 12


# In[4]:


out1 = conv1(input)
out2 = pool(out1)


# In[5]:


out1.size()


# In[6]:


out2.size()


# In[ ]:




