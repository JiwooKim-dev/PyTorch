#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

t = torch.FloatTensor(
    [[1.,2.,3.],
    [4.,5.,6.],
    [7.,8.,9.]]
)

print(t)


# In[2]:


print(t.dim())
print(t.shape)


# In[3]:


print([:, 1])


# In[4]:


print(t[:, 1])


# In[5]:


print(t[:, 1].shape)


# In[6]:


print(t[2, :])


# In[7]:


print(t[2, :].size())


# In[8]:


print(t[:, :-1])


# In[9]:


print(t[:, :-1].shape)


# In[ ]:




