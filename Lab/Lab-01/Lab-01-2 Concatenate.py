#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


x = torch.FloatTensor([[1,2], [3,4]])
y = torch.FloatTensor([[5,6], [7,8]])
print(x)
print(y)


# In[3]:


print(torch.cat([x,y], dim=0))  # dim 0이 늘어난다는 뜻


# In[4]:


print(torch.cat([x,y], dim=1))


# In[ ]:




