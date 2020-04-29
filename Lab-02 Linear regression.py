#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch


# In[7]:


# Model 정의
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

W = torch.zeros(1, requires_grad=True)  # zeros : 0으로 초기화
b = torch.zeros(1, requires_grad=True)  # requires_grad=True : 학습 O


# In[13]:


# Model 학습
optimizer = torch.optim.SGD([W,b], lr=0.01)  # optim 라이브러리의 SGD 함수 (학습할 tensor 리스트, learning rate)

nb_epochs = 1000
for epoch in range(1, nb_epochs + 1):
    # Hypothesis 예측
    hypothesis = x_train * W + b
    # Cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)  # 편차 제곱의 평균
    # Optimizer로 학습
    optimizer.zero_grad()  # gradient 초기화
    cost.backward()        # gradient 계산
    optimizer.step()       # W, b 개선

print('W: ',W[0])
print('b: ',b[0])

