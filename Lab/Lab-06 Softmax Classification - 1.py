#!/usr/bin/env python
# coding: utf-8

# In[1]:


# softmax : 각 값이 나올 확률값 추출
# cross entropy : 로그함수 상하반전 -> loss 측정

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


torch.manual_seed(1)


# In[3]:


t = torch.rand(3,5, requires_grad=True)  # t = (3,5)
hypothesis = F.softmax(t, dim=1)
print(hypothesis)


# In[4]:


y = torch.randint(5, (3,)).long()
print(y)


# In[5]:


y_one_hot = torch.zeros_like(hypothesis)  # y_one_hot = (3,5)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)  # y = (3,) -> y = (3,1)


# In[6]:


cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()  # (3,5) -> sum -> (3,1) -> mean -> scalar
print(cost)


# In[8]:


F.cross_entropy(t, y)  # == F.nll_loss(F.log_softmax(t, dim=1), y)


# In[ ]:




