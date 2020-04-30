#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch


# In[3]:


# Data set
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# In[4]:


# Model
W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# In[5]:


# Optimizer
optimizer = torch.optim.SGD([W,b], lr=1e-5)


# In[6]:


nb_epochs = 20
for epoch in range(nb_epochs + 1):
    
    # Hypothesis
    hypothesis = x_train.matmul(W) + b
    
    # Cost
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    print('Epoch {:2d}/{} H(x): {} Cost: {:.6f}'.format(epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))
        
    # 학습
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    


# In[ ]:




