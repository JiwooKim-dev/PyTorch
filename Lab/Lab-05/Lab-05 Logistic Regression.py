#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]  # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]  # (6, 1)


# In[3]:


x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


# In[4]:


W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# In[5]:


# hypothesis : 각각의 x에 대해 P(x=1) 확률
# hypothesis = torch.sigmoid(x_train.matmul(W) + b)  # 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
# cost = F.binary_cross_entropy(hypothesis, y_train)  # -(y_train * torch.log(hypothesis)
                                                    # + (1-y_train) * torch.log(1-hypothesis))


# In[6]:


optimizer = optim.SGD([W,b], lr=1)


# In[7]:


nb_epochs = 101
for epoch in range(nb_epochs):
    
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    optimizer.zero_grad()   # 기존의 grad를 0으로 초기화 (안하면 기존의 grad에 더해짐)
    cost.backward()         # cost값에 따른 W,b의 gradient 계산
    optimizer.step()        # cost minimize하는 방향으로 W,b update
    
    if epoch % 10 == 0:
        print('Epoch {:4d} Cost: {:.6f}'.format(epoch, cost.item()))


# In[8]:


# 결과 출력
hypothesis = torch.sigmoid(x_train.matmul(W) + b)  # x_train -> x_test로 변경하면 성능 테스트
print(hypothesis[:6])  


# In[9]:


prediction = hypothesis >= torch.FloatTensor([0.5])  # ByteTensor에 T,F 저장
print(prediction)


# In[ ]:




