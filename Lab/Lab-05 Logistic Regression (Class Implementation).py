#!/usr/bin/env python
# coding: utf-8

# In[31]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[32]:


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # {W, b}
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))


# In[33]:


x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]  # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]  # (6, 1)

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = BinaryClassifier()


# In[34]:


optimizer = optim.SGD(model.parameters(), lr=1)


# In[35]:


nb_epochs = 101
for epoch in range(nb_epochs):
    
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    optimizer.zero_grad()   # 기존의 grad를 0으로 초기화 (안하면 기존의 grad에 더해짐)
    cost.backward()         # cost값에 따른 W,b의 gradient 계산
    optimizer.step()        # cost minimize하는 방향으로 W,b update
    
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d} Cost: {:.6f} Acc: {:2.2f}'.format(epoch, cost.item(), accuracy * 100))


# In[ ]:




