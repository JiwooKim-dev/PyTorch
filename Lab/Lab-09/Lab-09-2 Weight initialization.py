#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# In[3]:


# Multi-Layers & ReLU & Adam & Xavier

batch_size = 100
training_epochs = 15
learning_rate = 0.001

# input (28 * 28), output (0~9), 3 Layers
linear1 = torch.nn.Linear(784, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 10, bias=True)
relu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3)   # 마지막에 relu 안해야 CSE 사용 가능

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

mnist_train = datasets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    
    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        X = X.view(-1, 28*28)
        
        hypothesis = model(X)
        cost = loss(hypothesis, Y)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
    
    print('Epoch: {:3d} Cost: {:.9f}'.format(epoch, avg_cost))
    

with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28*28).float()
    Y_test = mnist_test.test_labels
    
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: {:.6f}'.format(accuracy.item()))


# In[ ]:




