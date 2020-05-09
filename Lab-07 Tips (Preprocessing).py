#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


x_train = torch.FloatTensor([[73,80,75],
                            [93,88,93],
                            [89,91,90],
                            [96,98,100],
                            [73,66,70]])   # size : (m, 3)
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])   # size : (m, 1)


# In[3]:


# Data Preprocessing by Standardization ~ N(0, 1)
# Standardization 안하면 아예 NAN 뜸
# 특히 Y가 각 col 값이 차이가 큰 다차원값인 경우 공평하게 학습이 이루어지지 않을 수 있음

mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma


# In[4]:


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.linear(x)


# In[5]:


model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# In[6]:


def train(model, optimizer, x_train, y_train):
    nb_epochs = 201
    for epoch in range(nb_epochs):
        
        # H(x)
        prediction = model(x_train)   # x_train : (m, 3) -> prediction : (m, 1)
        
        # Cost
        cost = F.mse_loss(prediction, y_train)   # y_train : (m, 1)
        
        # 학습
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print('Epoch {:3d} Cost: {:.6f}'.format(epoch, cost.item()))


# In[7]:


train(model, optimizer, norm_x_train, y_train)


# In[ ]:




