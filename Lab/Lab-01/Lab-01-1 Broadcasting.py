#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[3]:


# Same Shape
m1 = torch.FloatTensor([[3,3]])
m2 = torch.FloatTensor([[2,2]])
print(m1+m2)


# In[4]:


# Vector + Scalar
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([3])  # 자동으로 Vector 크기로 확장 -> [[3,3]]
print(m1+m2)


# In[5]:


# 2X1 Vector + 1X2 Vector
m1 = torch.FloatTensor([[1,2]])  # (1,2)
m2 = torch.FloatTensor([[3],[4]])  # (2,1)
print(m1+m2)  # (2,2)로 확장


# In[6]:


# Mean
t = torch.FloatTensor([1,2])
print(t.mean())  # only FloatTensor


# In[7]:


t = torch.FloatTensor([[1,2], [3,4]])
print(t)


# In[8]:


print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))


# In[9]:


# Sum
print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))


# In[10]:


# Max
print(t.max())  # 최대 단일값 하나만 return


# In[12]:


print(t.max(dim=0))  # dimension 내 각각의 최댓값 + 인덱스 return
print('Max: ', t.max(dim=0)[0])
print('ArgMax: ', t.max(dim=0)[1])


# In[13]:


print(t.max(dim=1)[0])
print(t.max(dim=1)[1])


# In[ ]:




