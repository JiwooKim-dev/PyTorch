#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

X = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]])
Y = torch.FloatTensor([[0], [1], [1], [0]])

linear = torch.nn.Linear(2, 1, bias=True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear, sigmoid)


# In[2]:


criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)


# In[3]:


for step in range(10001):
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(step, cost.item())
    


# In[6]:


with torch.no_grad():
    prediction = model(X)
    
    correct_prediction = torch.argmax(prediction, 1) == Y
    accuracy = correct_prediction.float().mean()
    print('Accuracy: {:.1f}'.format(accuracy.item()))


# In[ ]:




