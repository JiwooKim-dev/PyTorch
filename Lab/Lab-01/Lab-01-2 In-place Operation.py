#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


x = torch.FloatTensor([[1,2], [3,4]])


# In[3]:


print(x.mul(2.))  # 원래 x와 별개의 메모리에 결괏값 저장


# In[4]:


print(x.mul_(2.))  # 원래 x의 장소에 그대로 결괏값 저장


# In[5]:


print(x)


# In[ ]:




