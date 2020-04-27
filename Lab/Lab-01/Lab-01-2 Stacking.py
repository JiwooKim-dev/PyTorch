#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[4]:


x = torch.FloatTensor([1,4])
y = torch.FloatTensor([2,5])
z = torch.FloatTensor([3,6])


# In[5]:


print(torch.stack([x,y,z]))


# In[6]:


print(torch.stack([x,y,z], dim=1))  # 쌓이는 3이 dim 1로 간다는 뜻


# In[8]:


print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))


# In[12]:


print(torch.cat([x.view([-1,2]),y.view([-1,2]),z.view([-1,2])], dim=0))


# In[ ]:




