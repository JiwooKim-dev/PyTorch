#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]
y_data = [2,2,2,1,1,1,0,0]

x_train = torch.FloatTensor(x_data)  # (m,4)
y_train = torch.LongTensor(y_data)  # (m,) / index는 정수니까 LongTensor로 받아줘야함


# In[3]:


# 모델 초기화
W = torch.zeros((4,3), requires_grad=True)  # 입력벡터 4 -> 클래스 3 
b = torch.zeros(1, requires_grad=True)


# In[4]:


# optimizer
optimizer = optim.SGD([W,b], lr=0.1)


# In[5]:


nb_epochs = 1001
for epoch in range(nb_epochs):
    
    # hypothesis
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
    
    # cost
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()
    
    # 모델 학습
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d} Cost: {:.5f}'.format(epoch, cost.item()))


# In[ ]:




