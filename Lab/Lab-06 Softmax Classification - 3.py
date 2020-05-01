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

x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)


# In[3]:


W = torch.zeros((4,3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr=0.1)


# In[4]:


nb_epochs = 1001
for epoch in range(nb_epochs):
    
    # cost
    hypothesis = x_train.matmul(W) + b
    cost = F.cross_entropy(hypothesis, y_train)  # scatter를 통해 one_hot 벡터를 구할 필요 없음
    
    # 학습
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d} Cost: {:.5f}'.format(epoch, cost.item()))


# In[ ]:




