#!/usr/bin/env python
# coding: utf-8

# In[1]:


# BackPropagation : 결괏값(Y)에 대한 각 요소들(W, b, X)의 영향력(미분값)을 구하기 위한 과정

import torch
torch.manual_seed(777)


# In[2]:


X = torch.FloatTensor([[0,0], [0,1], [1,0], [0,0]])
Y = torch.FloatTensor([[0], [1], [1], [0]])


# In[3]:


# 2 Layers

'''
# L1
w1 = torch.Tensor(2,2)  # 2 inputs -> 2 outputs
b1 = torch.Tensor(2)    # 2 bias
# L2
w2 = torch.Tensor(2,1)  # 2 inputs -> output
b2 = torch.Tensor(1)    # 1 bias
'''

l1 = torch.nn.Linear(2, 2, bias=True)
l2 = torch.nn.Linear(2, 1, bias=True)


# In[4]:


'''
def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

# Sigmoid 미분
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
'''

sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(l1, sigmoid, l2, sigmoid)


# In[5]:


'''
learning_rate = 1
for step in range(10001):
    # Forward
    l1 = torch.add(torch.matmul(X, w1), b1)
    a1 = sigmoid(l1)
    l2 = torch.add(torch.matmul(a1, w2), b2)
    Y_pred = sigmoid(l2)
    
    cost = -torch.mean(Y * torch.log(Y_pred) + (1 - Y) * torch.log(1 - Y_pred))   # BCELoss
    
    # BackPropagation
    
    # cost 함수 (BCE) 미분
    d_Y_pred = (Y_pred - Y) / (Y_pred * (1.0 - Y_pred) + 1e-7)
    
    # L2
    d_l2 = d_Y_pred * sigmoid_prime(l2)                    # sigmoid 미분 계산
    d_b2 = d_l2                                            # bias 미분 계산
    d_w2 = torch.matmul(torch.transpose(a1, 0, 1), d_b2)   # weight 미분 계산
    
    # L1
    d_a1 = torch.matmul(d_b2, torch.transpose(w2, 0, 1))
    d_l1 = d_a1 * sigmoid_prime(l1)
    d_b1 = d_l1
    d_w1 = torch.matmul(torch.transpose(X, 0, 1), d_b1)
    
    
    # Update Weights and Biases (Gradient Descent)
    w1 = w1 - learning_rate * d_w1
    b1 = b1 - learning_rate * torch.mean(d_b1, 0)
    w2 = w2 - learning_rate * d_w2
    b2 = b2 - learning_rate * torch.mean(d_b2, 0)
    
    
    if step % 1000 == 0:
        print(step, cost.item())
'''

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)


# In[6]:


for step in range(10001):
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if step % 1000 == 0:
        print(step, cost.item())

