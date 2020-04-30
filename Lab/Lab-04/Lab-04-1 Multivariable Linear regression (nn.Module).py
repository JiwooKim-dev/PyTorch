#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


# Data set
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# In[5]:


import torch.nn as nn

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)  # 입력 3, 출력 1의 모델 생성
        
    def forward(self, x):
        return self.linear(x)  # hypotehsis 자동 계산
    
# Model
model = MultivariateLinearRegressionModel()


# In[19]:


# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


# In[25]:


import torch.nn.functional as F

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    
    # H(x)
    hypothesis = model(x_train)
    
    # Cost
    cost = F.mse_loss(hypothesis, y_train)
    
    print('Epoch {:2d}/{} H(x): {} Cost: {:.6f}'.format(epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))

    # Gradient Descent
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


# In[ ]:




