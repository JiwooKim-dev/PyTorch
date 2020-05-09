#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch


# In[3]:


batch_size = 100
training_epochs = 15

# input (28 * 28), output (0~9)
linear = torch.nn.Linear(784, 10, bias=True)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

mnist_train = datasets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)


# In[4]:


for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    
    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        X = X.view(-1, 28*28)
        
        hypothesis = linear(X)
        cost = loss(hypothesis, Y)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
    
    print('Epoch: {:4d} Cost: {:.9f}'.format(epoch, avg_cost))


# In[6]:


with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28*28).float()
    Y_test = mnist_test.test_labels
    
    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: {:.6f}'.format(accuracy.item()))


# In[ ]:




