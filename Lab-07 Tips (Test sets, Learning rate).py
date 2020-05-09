#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


x_train = torch.FloatTensor([[1,2,1],
                            [1,3,2],
                            [1,3,4],
                            [1,5,5],
                            [1,7,5],
                            [1,2,5],
                            [1,6,6],
                            [1,7,7]])    # size : (m, 3)
y_train = torch.LongTensor([2,2,2,1,1,1,0,0])   # size : (m, 1)

x_test = torch.FloatTensor([[2,1,1], [3,1,2], [3,3,4]])
y_test = torch.LongTensor([2,2,2])


# In[3]:


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)   # 3 inputs -> 3 outputs
    def forward(self, x):
        return self.linear(x)   # x = (m, 3) -> (m, 3)


# In[4]:


model = SoftmaxClassifierModel()


# In[5]:


optimizer = optim.SGD(model.parameters(), lr=1e-1)


# In[6]:


# Training
def train(model, optimizer, x_train, y_train):
    nb_epochs = 201
    for epoch in range(nb_epochs):
        
        # H(x)
        prediction = model(x_train)   # x_train : (m, 3) -> prediction : (m, 3)
        
        # Cost
        cost = F.cross_entropy(prediction, y_train)   # y_train : (m, 1)
        
        # 학습
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print('Epoch {:3d} Cost: {:.6f}'.format(epoch, cost.item()))


# In[7]:


# Validation
def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)
    
    print('Accuracy: {}% Cost: {:.6f}'.format(correct_count / len(y_test)*100, cost.item()))


# In[8]:


train(model, optimizer, x_train, y_train)


# In[9]:


test(model, optimizer, x_test, y_test)


# In[ ]:




