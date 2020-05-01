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


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3)  # 4 inputs -> 3 outputs
    
    def forward(self, x):
        return self.linear(x)  # 통과한 결과 값 : (m, 3)


# In[4]:


model = SoftmaxClassifierModel()


# In[5]:


optimizer = optim.SGD(model.parameters(), lr=0.1)


# In[8]:


nb_epochs = 1001
for epoch in range(nb_epochs):
    
    hypothesis = model(x_train)  # x_train = (m, 4)
    cost = F.cross_entropy(hypothesis, y_train)  # hypothesis = (m, 3),  y_train = (m, 1)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d} Cost: {:.5f}'.format(epoch, cost.item()))


# In[ ]:


'''
Binray Classification : BCE, Sigmoid
Multi-class Classification : CE, Softmax
'''

