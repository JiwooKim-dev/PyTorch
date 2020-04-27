#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


lt = torch.LongTensor([1,2,3,4])
print(lt)


# In[3]:


print(lt.float())


# In[4]:


bt = torch.ByteTensor([True, False, True, False])  # Boolean
print(bt)


# In[5]:


print(bt.long())
print(bt.float())


# In[ ]:




