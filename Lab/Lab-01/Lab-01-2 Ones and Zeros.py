#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


x = torch.FloatTensor([[0,1,2], [2,1,0]])


# In[3]:


print(torch.ones_like(x))


# In[4]:


print(torch.zeros_like(x))


# In[ ]:


# 같은 device(CPU, GPU)에 텐서 선언

