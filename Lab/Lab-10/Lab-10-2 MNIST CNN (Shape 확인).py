#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[2]:


inputs = torch.Tensor(1, 1, 28, 28)


# In[3]:


conv1 = nn.Conv2d(1, 32, 3, padding=1)
pool1 = nn.MaxPool2d(2)


# In[4]:


conv2 = nn.Conv2d(32, 64, 3, padding=1)
pool2 = nn.MaxPool2d(2)


# In[5]:


out = conv1(inputs)
out.shape


# In[6]:


out = pool1(out)
out.shape


# In[7]:


out = conv2(out)
out.shape


# In[8]:


out = pool2(out)
out.shape


# In[9]:


out = out.view(out.size(0), -1)   # batch_size(out.size(0))만큼 남기고 flatten
out.shape


# In[10]:


layer1 = nn.Linear(out.size(1), 10)


# In[11]:


out = layer1(out)
out.shape


# In[ ]:




