#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


# 데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])


# In[10]:


# 모델 학습 1

W = torch.zeros(1)
learning_rate = 0.1

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    # Hypothesis 계산
    hypothesis = x_train * W
    # Cost gradient 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((W * x_train - y_train) * x_train)  # W에 대한 cost 미분값
    
    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(epoch, nb_epochs, W.item(), cost.item()))
    
    # 모델 개선
    W -= learning_rate * gradient


# In[11]:


# 모델 학습 2

W = torch.zeros(1, requires_grad=True)
learning_rate = 0.1

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    # Hypothesis 계산
    hypothesis = x_train * W
    # Cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(epoch, nb_epochs, W.item(), cost.item()))
    # Optimizer 설정
    optimizer = torch.optim.SGD([W], learning_rate)
    # 모델 개선
    optimizer.zero_grad()  # gradient 0으로 초기화
    cost.backward()        # gradient 계산 (자동 미분)
    optimizer.step()       # gradient descent (W 조정한다는 뜻)


# In[ ]:




